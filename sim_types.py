"""Varaha simulation types — core data structures for the wildfire logistics environment."""

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------

@dataclass
class Vec3:
    """Lightweight 3-component vector with basic arithmetic helpers."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # --- arithmetic ---

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    # --- magnitude ---

    def norm(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n < 1e-9:
            return Vec3(0.0, 0.0, 0.0)
        return self.scale(1.0 / n)

    def clamp_magnitude(self, max_mag: float) -> "Vec3":
        n = self.norm()
        if n > max_mag and n > 1e-9:
            return self.scale(max_mag / n)
        return Vec3(self.x, self.y, self.z)

    # --- distance ---

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).norm()

    def horizontal_distance_to(self, other: "Vec3") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    # --- serialization ---

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4)}

    def __repr__(self) -> str:
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    """Full kinematic + status state of the drone."""

    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    battery: float = 100.0
    carrying_payload: bool = True
    alive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "battery": round(self.battery, 4),
            "carrying_payload": self.carrying_payload,
            "alive": self.alive,
        }


# ---------------------------------------------------------------------------
# World entities
# ---------------------------------------------------------------------------

@dataclass
class BaseStation:
    """Home base where the drone launches, lands, and recharges."""

    position: Vec3 = field(default_factory=Vec3)
    recharge_radius: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "recharge_radius": self.recharge_radius,
        }


@dataclass
class DeliveryTarget:
    """A responder zone requiring supply delivery."""

    id: str = ""
    position: Vec3 = field(default_factory=Vec3)
    urgency: float = 0.5
    delivered: bool = False
    delivery_radius: float = 15.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "urgency": round(self.urgency, 4),
            "delivered": self.delivered,
            "delivery_radius": self.delivery_radius,
        }


@dataclass
class HazardRegion:
    """Wildfire danger zone modeled as a ground-level dome.

    The hazard has a horizontal radius and a height.  Danger is zero
    above ``height`` and outside ``radius``, allowing drones to fly
    over fires at sufficient altitude.  Within the dome, danger scales
    with proximity to the center both horizontally and vertically.

    ``growth_rate`` controls per-step height increase (metres/step),
    simulating fire growth over an episode.
    """

    id: str = ""
    center: Vec3 = field(default_factory=Vec3)
    radius: float = 50.0
    severity: float = 0.5
    height: float = 80.0
    growth_rate: float = 0.0
    _current_height: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self._current_height = self.height

    def reset(self):
        """Reset dynamic state for a new episode."""
        self._current_height = self.height

    def tick(self):
        """Advance one timestep — grow the fire."""
        if self.growth_rate > 0:
            self._current_height += self.growth_rate

    def contains(self, pos: Vec3) -> bool:
        horiz = ((pos.x - self.center.x) ** 2 + (pos.y - self.center.y) ** 2) ** 0.5
        alt = pos.z - self.center.z
        return horiz <= self.radius and 0 <= alt < self._current_height

    def danger_factor(self, pos: Vec3) -> float:
        """0 outside the dome, scales up toward the ground-level center."""
        horiz = ((pos.x - self.center.x) ** 2 + (pos.y - self.center.y) ** 2) ** 0.5
        if horiz >= self.radius:
            return 0.0
        alt = pos.z - self.center.z
        if alt >= self._current_height or alt < 0:
            return 0.0
        horiz_factor = 1.0 - horiz / self.radius
        vert_factor = 1.0 - alt / self._current_height
        return self.severity * horiz_factor * vert_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "center": self.center.to_dict(),
            "radius": self.radius,
            "severity": self.severity,
            "height": self.height,
            "current_height": round(self._current_height, 2),
            "growth_rate": self.growth_rate,
        }


@dataclass
class ObstacleVolume:
    """Axis-aligned 3D box that the drone must not enter."""

    id: str = ""
    min_corner: Vec3 = field(default_factory=Vec3)
    max_corner: Vec3 = field(default_factory=Vec3)
    kind: str = "building"

    def contains(self, pos: Vec3) -> bool:
        return (
            self.min_corner.x <= pos.x <= self.max_corner.x
            and self.min_corner.y <= pos.y <= self.max_corner.y
            and self.min_corner.z <= pos.z <= self.max_corner.z
        )

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.min_corner.x + self.max_corner.x) / 2,
            (self.min_corner.y + self.max_corner.y) / 2,
            (self.min_corner.z + self.max_corner.z) / 2,
        )

    @property
    def half_size(self) -> Vec3:
        return Vec3(
            (self.max_corner.x - self.min_corner.x) / 2,
            (self.max_corner.y - self.min_corner.y) / 2,
            (self.max_corner.z - self.min_corner.z) / 2,
        )

    @property
    def height(self) -> float:
        return self.max_corner.z

    def nearest_surface_dist(self, pos: Vec3) -> float:
        """Signed distance to the nearest surface (negative = inside)."""
        cx, cy = self.center.x, self.center.y
        hx, hy = self.half_size.x, self.half_size.y
        dx = max(abs(pos.x - cx) - hx, 0.0)
        dy = max(abs(pos.y - cy) - hy, 0.0)
        dz_below = max(self.min_corner.z - pos.z, 0.0)
        dz_above = max(pos.z - self.max_corner.z, 0.0)
        return math.sqrt(dx * dx + dy * dy + (dz_below + dz_above) ** 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "min_corner": self.min_corner.to_dict(),
            "max_corner": self.max_corner.to_dict(),
            "kind": self.kind,
        }


@dataclass
class CylindricalObstacle:
    """Vertical cylinder obstacle — trees, poles, pillars, tanks."""

    id: str = ""
    center: Vec3 = field(default_factory=Vec3)
    radius: float = 10.0
    height: float = 50.0
    kind: str = "tree"

    def contains(self, pos: Vec3) -> bool:
        dx = pos.x - self.center.x
        dy = pos.y - self.center.y
        horiz_dist = math.sqrt(dx * dx + dy * dy)
        return horiz_dist <= self.radius and 0 <= pos.z <= self.height

    def nearest_surface_dist(self, pos: Vec3) -> float:
        dx = pos.x - self.center.x
        dy = pos.y - self.center.y
        horiz_dist = math.sqrt(dx * dx + dy * dy)
        radial_gap = max(horiz_dist - self.radius, 0.0)
        vert_gap = max(pos.z - self.height, 0.0) if pos.z > self.height else max(-pos.z, 0.0)
        return math.sqrt(radial_gap ** 2 + vert_gap ** 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "center": self.center.to_dict(),
            "radius": round(self.radius, 2),
            "height": round(self.height, 2),
            "kind": self.kind,
        }


# ---------------------------------------------------------------------------
# Responder units — dynamic actors that alter mission conditions mid-episode
# ---------------------------------------------------------------------------

RESPONDER_STATUSES = ("stable", "urgent", "critical")
RESPONDER_STATUS_MAP = {"stable": 0.0, "urgent": 0.5, "critical": 1.0}

INTEL_TYPES = (
    "none",
    "blocked_north", "blocked_south", "blocked_east", "blocked_west",
    "safe_north", "safe_south", "safe_east", "safe_west",
    "fire_expanded", "fire_receded",
)

INTEL_DIRECTION_VECS = {
    "none": (0.0, 0.0),
    "blocked_north": (0.0, 1.0), "blocked_south": (0.0, -1.0),
    "blocked_east": (1.0, 0.0), "blocked_west": (-1.0, 0.0),
    "safe_north": (0.0, 1.0), "safe_south": (0.0, -1.0),
    "safe_east": (1.0, 0.0), "safe_west": (-1.0, 0.0),
    "fire_expanded": (0.0, 0.0), "fire_receded": (0.0, 0.0),
}


@dataclass
class ScheduledEvent:
    """A future event a responder will trigger at a specific step."""
    step: int = 0
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    fired: bool = False


@dataclass
class ResponderUnit:
    """First responder on the ground linked to a delivery target.

    Can dynamically alter mission conditions mid-episode:
      1. Update urgency of their linked target
      2. Relocate the drop-zone (move target position)
      3. Broadcast hazard intel (structured approach guidance)
    """

    id: str = ""
    position: Vec3 = field(default_factory=Vec3)
    linked_target_id: str = ""
    status: str = "stable"
    current_need: str = "supplies"
    message: str = ""
    can_update_dropzone: bool = False
    active: bool = True

    latest_intel: str = "none"
    intel_severity: float = 0.0

    scheduled_events: list[ScheduledEvent] = field(default_factory=list)

    def status_code(self) -> float:
        return RESPONDER_STATUS_MAP.get(self.status, 0.0)

    def intel_direction(self) -> tuple[float, float]:
        return INTEL_DIRECTION_VECS.get(self.latest_intel, (0.0, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "linked_target_id": self.linked_target_id,
            "status": self.status,
            "current_need": self.current_need,
            "message": self.message,
            "can_update_dropzone": self.can_update_dropzone,
            "active": self.active,
            "latest_intel": self.latest_intel,
            "intel_severity": round(self.intel_severity, 4),
        }


# ---------------------------------------------------------------------------
# Observation & step diagnostics
# ---------------------------------------------------------------------------

@dataclass
class VarahaObservation:
    """Structured observation returned to the agent each step.

    Kept as a dataclass for documentation; the env also offers a plain-dict
    path via ``get_observation()`` for maximum serialisation flexibility.
    """

    drone_position: Vec3 = field(default_factory=Vec3)
    drone_velocity: Vec3 = field(default_factory=Vec3)
    battery: float = 100.0
    carrying_payload: bool = True
    alive: bool = True
    targets: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 500

    def to_dict(self) -> dict[str, Any]:
        return {
            "drone_position": self.drone_position.to_dict(),
            "drone_velocity": self.drone_velocity.to_dict(),
            "battery": round(self.battery, 4),
            "carrying_payload": self.carrying_payload,
            "alive": self.alive,
            "targets": self.targets,
            "step": self.step,
            "max_steps": self.max_steps,
        }


@dataclass
class MissionInstruction:
    """Single mission instruction used for long-horizon planning mode."""

    id: str = ""
    kind: str = ""
    description: str = ""
    target_id: str = ""
    tool_name: str = ""
    completed: bool = False
    violated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "target_id": self.target_id,
            "tool_name": self.tool_name,
            "completed": self.completed,
            "violated": self.violated,
        }


@dataclass
class TracePoint:
    """Single frame of the drone's recorded trajectory."""

    step: int = 0
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    battery: float = 100.0
    reward: float = 0.0
    cumulative_reward: float = 0.0
    events: list[str] = field(default_factory=list)
    observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "battery": round(self.battery, 4),
            "reward": round(self.reward, 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
            "events": list(self.events),
            "observation": self.observation,
        }


@dataclass
class StepInfo:
    """Per-step diagnostic info returned alongside the reward."""

    collision: bool = False
    delivered_target_ids: list[str] = field(default_factory=list)
    in_hazard: bool = False
    hazard_severity: float = 0.0
    reached_base: bool = False
    distance_traveled: float = 0.0
    tool_call: str = ""
    tool_result: dict[str, Any] = field(default_factory=dict)
    instruction_completed: int = 0
    instruction_total: int = 0
    instruction_violations: int = 0
    reward_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collision": self.collision,
            "delivered_target_ids": list(self.delivered_target_ids),
            "in_hazard": self.in_hazard,
            "hazard_severity": round(self.hazard_severity, 4),
            "reached_base": self.reached_base,
            "distance_traveled": round(self.distance_traveled, 4),
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "instruction_completed": self.instruction_completed,
            "instruction_total": self.instruction_total,
            "instruction_violations": self.instruction_violations,
            "reward_breakdown": {
                k: round(v, 4) for k, v in self.reward_breakdown.items()
            },
        }
