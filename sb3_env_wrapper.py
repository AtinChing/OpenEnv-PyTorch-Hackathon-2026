"""Gymnasium-compatible wrapper for VarahaEnv, ready for Stable-Baselines3.

Observation space expanded for hardcore environments with obstacles,
cylindrical obstacles, hazards, targets, and first-responder units.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from varaha_env import VarahaEnv, VarahaConfig


class VarahaSB3Env(gym.Env):
    """Wraps VarahaEnv into a fixed-size continuous obs/action Gymnasium env.

    Observation (124-dim float32):
        [0:3]     drone position (normalised to world bounds)
        [3:6]     drone velocity (normalised to max_speed)
        [6]       battery fraction (0-1)
        [7]       carrying_payload flag
        [8:11]    relative vec to base (normalised)
        [11]      time fraction (step / max_steps)
        [12]      remaining deliveries fraction (0-1)
        [13]      distance to next target (normalised)
        [14:17]   next-target pointer: rel vec to nearest undelivered (or base)
        -- 17 core dims --
        [17:42]   5 targets x 5 dims: rel_pos(3) + urgency + delivered
        [42:66]   6 hazards x 4 dims: rel_pos_horiz(2) + height_norm + severity
        [66:106]  8 nearest obstacles x 5 dims: rel_x + rel_y + height_norm + size_x_norm + size_y_norm
        [106:124] 3 responders x 6 dims: status_code + rel_target(2) + intel_dir(2) + intel_severity
        -- total: 124 dims --

    Action (5-dim continuous, range [-1, 1]) when ``explicit_controls=True``:
        [0:3]  acceleration axes, scaled to max_acceleration
        [3]    deliver gate (>0 => attempt delivery)
        [4]    recharge gate (>0 => attempt recharge)

    Action (3-dim) when ``explicit_controls=False``:
        [0:3]  acceleration axes only, with automatic delivery/recharge behavior.
    """

    metadata = {"render_modes": []}
    N_TARGETS = 5
    N_HAZARDS = 6
    N_OBSTACLES = 8
    N_RESPONDERS = 3

    OBS_DIM = 17 + N_TARGETS * 5 + N_HAZARDS * 4 + N_OBSTACLES * 5 + N_RESPONDERS * 6

    # V2 (ultra-hard) slot counts for larger observation space
    V2_TARGETS = 6
    V2_HAZARDS = 8
    V2_OBSTACLES = 12
    V2_RESPONDERS = 5
    V2_OBS_DIM = 17 + V2_TARGETS * 5 + V2_HAZARDS * 4 + V2_OBSTACLES * 5 + V2_RESPONDERS * 6

    @property
    def obs_dim(self) -> int:
        return self.V2_OBS_DIM if self._ultra_hard else self.OBS_DIM

    def __init__(
        self,
        config: VarahaConfig | None = None,
        world_fn=None,
        ultra_hard: bool = False,
        explicit_controls: bool = True,
    ):
        super().__init__()
        self.env = VarahaEnv(config, world_fn=world_fn)
        cfg = self.env.cfg
        self._ultra_hard = ultra_hard
        self._explicit_controls = explicit_controls

        n_t, n_h, n_o, n_r = (
            (self.V2_TARGETS, self.V2_HAZARDS, self.V2_OBSTACLES, self.V2_RESPONDERS)
            if ultra_hard else
            (self.N_TARGETS, self.N_HAZARDS, self.N_OBSTACLES, self.N_RESPONDERS)
        )
        self._n_targets = n_t
        self._n_hazards = n_h
        self._n_obstacles = n_o
        self._n_responders = n_r
        obs_dim = 17 + n_t * 5 + n_h * 4 + n_o * 5 + n_r * 6

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        action_dim = 5 if explicit_controls else 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self._cfg = cfg
        self._max_dist = max(cfg.world_x, cfg.world_y)

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

    def _flatten(self, obs: dict) -> np.ndarray:
        c = self._cfg
        dp = obs["drone_position"]
        dv = obs["drone_velocity"]
        md = self._max_dist

        targets = obs["targets"][:self._n_targets]
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
            (self.env.base.position.x - dp["x"]) / md,
            (self.env.base.position.y - dp["y"]) / md,
            (self.env.base.position.z - dp["z"]) / md,
            obs["step"] / c.max_episode_steps,
            len(undelivered) / max(len(obs["targets"]), 1),
        ]

        if undelivered:
            nearest = min(undelivered, key=lambda t: (
                t["relative_position"]["x"] ** 2
                + t["relative_position"]["y"] ** 2
                + t["relative_position"]["z"] ** 2
            ))
            rn = nearest["relative_position"]
            dist = math.sqrt(rn["x"] ** 2 + rn["y"] ** 2 + rn["z"] ** 2)
            vec.append(dist / md)
            vec.extend([rn["x"] / md, rn["y"] / md, rn["z"] / md])
        else:
            bx = self.env.base.position.x - dp["x"]
            by = self.env.base.position.y - dp["y"]
            bz = self.env.base.position.z - dp["z"]
            dist = math.sqrt(bx ** 2 + by ** 2 + bz ** 2)
            vec.append(dist / md)
            vec.extend([bx / md, by / md, bz / md])

        # --- Targets ---
        for i in range(self._n_targets):
            if i < len(targets):
                rp = targets[i]["relative_position"]
                vec.extend([
                    rp["x"] / md, rp["y"] / md, rp["z"] / md,
                    targets[i]["urgency"],
                    1.0 if targets[i]["delivered"] else 0.0,
                ])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 1.0])

        # --- Hazards ---
        hazards = obs.get("hazards", [])
        for i in range(self._n_hazards):
            if i < len(hazards):
                hp = hazards[i]["relative_position"]
                vec.extend([
                    hp["x"] / md, hp["y"] / md,
                    hazards[i]["current_height"] / c.world_z,
                    hazards[i]["severity"],
                ])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0])

        # --- Obstacles: nearest N ---
        obstacles_sorted = obs.get("obstacles", [])
        for i in range(self._n_obstacles):
            if i < len(obstacles_sorted):
                o = obstacles_sorted[i]
                rp = o["relative_position"]
                vec.extend([
                    rp["x"] / md,
                    rp["y"] / md,
                    o["height"] / c.world_z,
                    o["size_x"] / md,
                    o["size_y"] / md,
                ])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # --- Responders ---
        responders = obs.get("responders", [])
        for i in range(self._n_responders):
            if i < len(responders):
                r = responders[i]
                rp = r["relative_position"]
                idir = r["intel_direction"]
                vec.extend([
                    r["status_code"],
                    rp["x"] / md,
                    rp["y"] / md,
                    idir["x"],
                    idir["y"],
                    r["intel_severity"],
                ])
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(vec, dtype=np.float32)

    def _to_action_dict(self, action) -> dict:
        c = self._cfg
        arr = np.asarray(action, dtype=np.float32).reshape(-1)

        if arr.size < 3:
            raise ValueError(f"Expected action with at least 3 values, got shape {arr.shape}")

        # Legacy 3-dim policies are still supported as a fallback, but new
        # training should use explicit deliver/recharge controls.
        if self._explicit_controls and arr.size >= 5:
            deliver = bool(arr[3] > 0.0)
            recharge = bool(arr[4] > 0.0)
        elif self._explicit_controls:
            deliver = self._can_deliver_now()
            recharge = self._can_recharge_now()
        else:
            deliver = True
            recharge = True

        return {
            "ax": float(arr[0]) * c.max_acceleration,
            "ay": float(arr[1]) * c.max_acceleration,
            "az": float(arr[2]) * c.max_acceleration,
            "deliver": deliver,
            "recharge": recharge,
        }

    def _can_deliver_now(self) -> bool:
        pos = self.env.drone.position
        for tgt in self.env.targets:
            if tgt.delivered:
                continue
            dx = pos.x - tgt.position.x
            dy = pos.y - tgt.position.y
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            alt_above = pos.z - tgt.position.z
            if horiz_dist <= tgt.delivery_radius and -10.0 <= alt_above <= tgt.delivery_radius * 2:
                return True
        return False

    def _can_recharge_now(self) -> bool:
        pos = self.env.drone.position
        base = self.env.base.position
        hdist = math.sqrt((pos.x - base.x) ** 2 + (pos.y - base.y) ** 2)
        return hdist <= self.env.base.recharge_radius

    def get_trace(self):
        return self.env.get_trace()
