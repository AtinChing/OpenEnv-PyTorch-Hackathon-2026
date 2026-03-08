"""Gymnasium-compatible wrapper for VarahaEnv, ready for Stable-Baselines3."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from varaha_env import VarahaEnv, VarahaConfig


class VarahaSB3Env(gym.Env):
    """Wraps VarahaEnv into a fixed-size continuous obs/action Gymnasium env.

    Observation (40-dim float32):
        [0:3]   drone position   (normalised to world bounds)
        [3:6]   drone velocity   (normalised to max_speed)
        [6]     battery fraction (0-1)
        [7]     carrying_payload flag
        [8:11]  relative vec to base (normalised)
        [11]    time fraction    (step / max_steps)
        [12]    remaining deliveries fraction (0-1)
        [13]    distance to next target (normalised)
        [14:17] next-target pointer: rel vec to nearest undelivered (or base)
        [17:22] target-0: rel_pos(3) + urgency + delivered
        [22:27] target-1: rel_pos(3) + urgency + delivered
        [27:32] target-2: rel_pos(3) + urgency + delivered
        [32:36] hazard-0: rel_pos_horiz(2) + height_norm + severity
        [36:40] hazard-1: rel_pos_horiz(2) + height_norm + severity

    Action (3-dim continuous, range [-1, 1]):
        [0:3]  acceleration axes, scaled to max_acceleration

    Deliver and recharge are **automatic** — the agent always attempts
    both when in range, removing a hard credit-assignment burden.
    """

    metadata = {"render_modes": []}
    N_TARGETS = 3
    N_HAZARDS = 2

    def __init__(self, config: VarahaConfig | None = None):
        super().__init__()
        self.env = VarahaEnv(config)
        cfg = self.env.cfg

        obs_dim = 17 + self.N_TARGETS * 5 + self.N_HAZARDS * 4  # 40
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._cfg = cfg
        self._max_dist = max(cfg.world_x, cfg.world_y)

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.env.reset(seed=seed)
        return self._flatten(obs_dict), {}

    def step(self, action):
        action_dict = self._to_action_dict(action)
        obs_dict, reward, done, info = self.env.step(action_dict)

        terminated = done and (
            not self.env.drone.alive or self.env._is_success()
        )
        truncated = done and not terminated

        return self._flatten(obs_dict), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten(self, obs: dict) -> np.ndarray:
        c = self._cfg
        dp = obs["drone_position"]
        dv = obs["drone_velocity"]

        targets = obs["targets"][: self.N_TARGETS]
        undelivered = [t for t in targets if not t["delivered"]]

        vec = [
            dp["x"] / c.world_x,
            dp["y"] / c.world_y,
            dp["z"] / c.world_z if c.world_z > 0 else 0.0,
            dv["x"] / c.max_speed,
            dv["y"] / c.max_speed,
            dv["z"] / c.max_speed,
            obs["battery"] / c.battery_capacity,
            1.0 if obs["carrying_payload"] else 0.0,
            (self.env.base.position.x - dp["x"]) / self._max_dist,
            (self.env.base.position.y - dp["y"]) / self._max_dist,
            (self.env.base.position.z - dp["z"]) / self._max_dist,
            obs["step"] / c.max_episode_steps,
            len(undelivered) / max(self.N_TARGETS, 1),
        ]

        if undelivered:
            nearest = min(undelivered, key=lambda t: (
                t["relative_position"]["x"] ** 2
                + t["relative_position"]["y"] ** 2
                + t["relative_position"]["z"] ** 2
            ))
            rn = nearest["relative_position"]
            dist = (rn["x"] ** 2 + rn["y"] ** 2 + rn["z"] ** 2) ** 0.5
            vec.append(dist / self._max_dist)
            vec.extend([rn["x"] / self._max_dist, rn["y"] / self._max_dist, rn["z"] / self._max_dist])
        else:
            bx = self.env.base.position.x - dp["x"]
            by = self.env.base.position.y - dp["y"]
            bz = self.env.base.position.z - dp["z"]
            dist = (bx ** 2 + by ** 2 + bz ** 2) ** 0.5
            vec.append(dist / self._max_dist)
            vec.extend([bx / self._max_dist, by / self._max_dist, bz / self._max_dist])

        for t in targets:
            rp = t["relative_position"]
            vec.extend([
                rp["x"] / self._max_dist,
                rp["y"] / self._max_dist,
                rp["z"] / self._max_dist,
                t["urgency"],
                1.0 if t["delivered"] else 0.0,
            ])

        for h in obs.get("hazards", [])[: self.N_HAZARDS]:
            hp = h["relative_position"]
            vec.extend([
                hp["x"] / self._max_dist,
                hp["y"] / self._max_dist,
                h["current_height"] / c.world_z,
                h["severity"],
            ])

        return np.array(vec, dtype=np.float32)

    def _to_action_dict(self, action) -> dict:
        c = self._cfg
        return {
            "ax": float(action[0]) * c.max_acceleration,
            "ay": float(action[1]) * c.max_acceleration,
            "az": float(action[2]) * c.max_acceleration,
            "deliver": True,
            "recharge": True,
        }

    def get_trace(self):
        return self.env.get_trace()
