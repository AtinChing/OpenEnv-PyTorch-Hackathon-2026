"""Pydantic models for the Varaha OpenEnv environment."""

from typing import Any, Dict, List, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class VarahaAction(Action):
    """Drone acceleration command with automatic delivery/recharge."""

    ax: float = Field(0.0, description="Desired acceleration along x-axis (m/s^2)")
    ay: float = Field(0.0, description="Desired acceleration along y-axis (m/s^2)")
    az: float = Field(0.0, description="Desired acceleration along z-axis (m/s^2)")
    deliver: bool = Field(True, description="Attempt delivery when near a target")
    recharge: bool = Field(True, description="Attempt recharge when near base station")
    tool_call: str = Field(
        "",
        description="Optional tool call: request_intel[:target_id] | battery_forecast | mission_report",
    )


class VarahaObservation(Observation):
    """Full observation returned after each step/reset."""

    drone_position: Dict[str, float] = Field(
        default_factory=dict, description="Drone {x, y, z} in local metres"
    )
    drone_velocity: Dict[str, float] = Field(
        default_factory=dict, description="Drone velocity {x, y, z} in m/s"
    )
    battery: float = Field(0.0, description="Remaining battery units")
    carrying_payload: bool = Field(True, description="Whether the drone still carries payload")
    alive: bool = Field(True, description="Whether the drone is still operational")
    targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-target relative position, urgency, delivered status",
    )
    hazards: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-hazard relative position, current height, severity",
    )
    step_num: int = Field(0, description="Current step number in the episode")
    max_steps: int = Field(2000, description="Maximum allowed steps")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Itemised reward components from the last step"
    )
    mission: Dict[str, Any] = Field(
        default_factory=dict,
        description="Instruction-mode progress, next instruction, and violation counters",
    )
    last_tool_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="Result payload from the most recent tool call",
    )
    success: bool = Field(False, description="Whether the mission is successfully completed")
    trace: Optional[Dict[str, Any]] = Field(
        None, description="Full episode trace (only populated on the final step)"
    )


class VarahaState(State):
    """Internal environment state exposed via the state property."""

    cumulative_reward: float = Field(0.0, description="Total accumulated reward")
    deliveries_completed: int = Field(0, description="Number of targets delivered so far")
    total_targets: int = Field(0, description="Total number of targets in the episode")
    battery: float = Field(0.0, description="Current battery level")
    success: bool = Field(False, description="Whether the mission is complete")
