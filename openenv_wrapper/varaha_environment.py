"""OpenEnv-compatible Varaha wildfire drone environment."""

from __future__ import annotations

import sys
import os
import uuid
from typing import Any, Callable, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from varaha_env import VarahaConfig, VarahaEnv, build_random_world
from openenv_wrapper.models import VarahaAction, VarahaObservation, VarahaState


class VarahaEnvironment(Environment[VarahaAction, VarahaObservation, VarahaState]):
    """Wildfire logistics drone environment wrapped for OpenEnv.

    Each episode the drone must deliver supplies to responder zones near
    wildfire hazards, then return to base.  Supports domain-randomised
    worlds when ``world_fn`` is provided.
    """

    def __init__(
        self,
        config: Optional[VarahaConfig] = None,
        world_fn: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__()
        self._config = config or VarahaConfig()
        self._world_fn = world_fn
        self._env = VarahaEnv(config=self._config, world_fn=self._world_fn)
        self._episode_id = str(uuid.uuid4())
        self._last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # OpenEnv abstract interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> VarahaObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        obs_dict = self._env.reset(seed=seed)
        self._last_info = {}
        return self._build_observation(obs_dict, reward=0.0, done=False)

    def step(
        self,
        action: VarahaAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> VarahaObservation:
        action_dict = {
            "ax": action.ax,
            "ay": action.ay,
            "az": action.az,
            "deliver": action.deliver,
            "recharge": action.recharge,
            "tool_call": action.tool_call,
        }
        obs_dict, reward, done, info = self._env.step(action_dict)
        self._last_info = info
        return self._build_observation(obs_dict, reward=reward, done=done, info=info)

    @property
    def state(self) -> VarahaState:
        delivered = sum(1 for t in self._env.targets if t.delivered)
        return VarahaState(
            episode_id=self._episode_id,
            step_count=self._env.step_count,
            cumulative_reward=round(self._env.cumulative_reward, 4),
            deliveries_completed=delivered,
            total_targets=len(self._env.targets),
            battery=round(self._env.drone.battery, 4),
            success=self._env._is_success(),
        )

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Varaha Wildfire Logistics",
            description=(
                "A 3D drone delivery environment where an agent must navigate "
                "wildfire hazards and obstacles to deliver supplies to responder "
                "zones, then return to base."
            ),
            version="1.0.0",
            author="Varaha Team",
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        obs_dict: dict[str, Any],
        *,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> VarahaObservation:
        info = info or {}
        trace = self._env.get_trace() if done else None
        return VarahaObservation(
            done=done,
            reward=round(reward, 4),
            metadata={"info": info},
            drone_position=obs_dict["drone_position"],
            drone_velocity=obs_dict["drone_velocity"],
            battery=obs_dict["battery"],
            carrying_payload=obs_dict["carrying_payload"],
            alive=obs_dict["alive"],
            targets=obs_dict["targets"],
            hazards=obs_dict.get("hazards", []),
            mission=obs_dict.get("mission", {}),
            last_tool_result=obs_dict.get("last_tool_result", {}),
            step_num=obs_dict["step"],
            max_steps=obs_dict["max_steps"],
            reward_breakdown=info.get("reward_breakdown", {}),
            success=self._env._is_success(),
            trace=trace,
        )
