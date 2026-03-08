"""Gymnasium-compatible wrapper for VarahaEnv, ready for Stable-Baselines3."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from varaha_env import VarahaEnv, VarahaConfig


class VarahaSB3Env(gym.Env):
    """Wraps VarahaEnv into a fixed-size continuous obs/action Gymnasium env.

    Observation (27-dim float32):
        [0:3]   drone position   (normalised to world bounds)
        [3:6]   drone velocity   (normalised to max_speed)
        [6]     battery fraction (0-1)
        [7]     carrying_payload flag
        [8:11]  relative vec to base (normalised)
        [11]    time fraction    (step / max_steps)
        [12:17] target-0: rel_pos(3) + urgency + delivered
        [17:22] target-1: rel_pos(3) + urgency + delivered
        [22:27] target-2: rel_pos(3) + urgency + delivered

    Action (3-dim continuous, range [-1, 1]):
        [0:3]  acceleration axes, scaled to max_acceleration

    Deliver and recharge are **automatic** — the agent always attempts
    both when in range, removing a hard credit-assignment burden.
    """

    metadata = {"render_modes": []}
    N_TARGETS = 3

    def __init__(self, config: VarahaConfig | None = None):
        super().__init__()
        self.env = VarahaEnv(config)
        cfg = self.env.cfg

        obs_dim = 12 + self.N_TARGETS * 5  # 27
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
        ]

        for t in obs["targets"][: self.N_TARGETS]:
            rp = t["relative_position"]
            vec.extend([
                rp["x"] / self._max_dist,
                rp["y"] / self._max_dist,
                rp["z"] / self._max_dist,
                t["urgency"],
                1.0 if t["delivered"] else 0.0,
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
