"""WebSocket client for the Varaha OpenEnv server."""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from openenv_wrapper.models import VarahaAction, VarahaObservation, VarahaState


class VarahaEnvClient(EnvClient[VarahaAction, VarahaObservation, VarahaState]):
    """Typed client that speaks to a running Varaha OpenEnv server."""

    def _step_payload(self, action: VarahaAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[VarahaObservation]:
        obs_data = payload.get("observation", payload.get("data", payload))
        obs = VarahaObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> VarahaState:
        return VarahaState(**payload)
