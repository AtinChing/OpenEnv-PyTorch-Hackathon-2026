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
    """Wildfire / no-fly danger zone modeled as a sphere."""

    id: str = ""
    center: Vec3 = field(default_factory=Vec3)
    radius: float = 50.0
    severity: float = 0.5

    def contains(self, pos: Vec3) -> bool:
        return pos.distance_to(self.center) <= self.radius

    def danger_factor(self, pos: Vec3) -> float:
        """0 outside, scales up to *severity* at the center."""
        d = pos.distance_to(self.center)
        if d >= self.radius:
            return 0.0
        return self.severity * (1.0 - d / self.radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "center": self.center.to_dict(),
            "radius": self.radius,
            "severity": self.severity,
        }


@dataclass
class ObstacleVolume:
    """Axis-aligned 3D box that the drone must not enter."""

    id: str = ""
    min_corner: Vec3 = field(default_factory=Vec3)
    max_corner: Vec3 = field(default_factory=Vec3)

    def contains(self, pos: Vec3) -> bool:
        return (
            self.min_corner.x <= pos.x <= self.max_corner.x
            and self.min_corner.y <= pos.y <= self.max_corner.y
            and self.min_corner.z <= pos.z <= self.max_corner.z
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "min_corner": self.min_corner.to_dict(),
            "max_corner": self.max_corner.to_dict(),
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
    reward_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collision": self.collision,
            "delivered_target_ids": list(self.delivered_target_ids),
            "in_hazard": self.in_hazard,
            "hazard_severity": round(self.hazard_severity, 4),
            "reached_base": self.reached_base,
            "distance_traveled": round(self.distance_traveled, 4),
            "reward_breakdown": {
                k: round(v, 4) for k, v in self.reward_breakdown.items()
            },
        }
