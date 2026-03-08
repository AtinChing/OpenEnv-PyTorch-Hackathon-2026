"""Varaha — wildfire logistics simulation environment.

A drone must deliver supplies to responder zones near wildfire hazards in
California-like terrain.  The environment uses lightweight 3D kinematics with
local metre-based coordinates and an optional lat/lon conversion helper for
later Cesium visualisation.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from sim_types import (
    Vec3,
    DroneState,
    BaseStation,
    DeliveryTarget,
    HazardRegion,
    ObstacleVolume,
    StepInfo,
    TracePoint,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VarahaConfig:
    """All tunable environment parameters live here."""

    # World bounds (metres) — 5 km × 5 km operational area
    world_x: float = 5000.0
    world_y: float = 5000.0
    world_z: float = 200.0

    # Drone physics
    battery_capacity: float = 300.0
    max_speed: float = 25.0          # m/s
    max_acceleration: float = 8.0    # m/s²
    dt: float = 0.5                  # seconds per step

    # Episode
    max_episode_steps: int = 2000

    # Battery drain coefficients (tuned for 5 km scale)
    drain_per_meter: float = 0.008
    drain_elevation_factor: float = 0.02
    drain_idle_per_step: float = 0.005
    recharge_rate: float = 5.0       # battery units restored per recharge step

    # Reward knobs
    delivery_reward: float = 200.0
    return_bonus: float = 100.0
    step_penalty: float = 0.05
    battery_cost_factor: float = 0.3
    collision_penalty: float = 300.0
    hazard_penalty: float = 5.0
    failure_penalty: float = 200.0
    distance_shaping_factor: float = 0.05

    # California origin anchor (near Sacramento — wildfire-relevant)
    origin_lat: float = 38.55
    origin_lon: float = -121.47


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class VarahaEnv:
    """Core wildfire logistics simulation.

    Action format (dict)::

        {
            "ax": float,       # desired acceleration x (m/s²)
            "ay": float,       # desired acceleration y
            "az": float,       # desired acceleration z
            "deliver": bool,   # attempt delivery if near a target
            "recharge": bool,  # attempt recharge if near base
        }

    Returns ``(obs_dict, reward, done, info_dict)`` per OpenAI-gym convention.
    """

    def __init__(self, config: Optional[VarahaConfig] = None) -> None:
        self.cfg = config or VarahaConfig()

        self.base: BaseStation
        self.drone: DroneState
        self.targets: list[DeliveryTarget] = []
        self.hazards: list[HazardRegion] = []
        self.obstacles: list[ObstacleVolume] = []

        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.trace: list[TracePoint] = []

        self._prev_nearest_dist: float = 0.0

        self._build_demo_world()

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------

    def _build_demo_world(self) -> None:
        """Hardcoded 5 km demo scenario.

        Layout (top-down, +x → east, +y → north, 5 km × 5 km)::

            T3 (1000,4200)
            ·
            H2 (900,3200)    O2 [500-1500, 2600-3000]
            ·
            ·                   T2 (4100,2900) ← inside H1 fringe
            ·                H1 (3800,2600)
            ·
            ·        O1 [2200-2800, 1000-2200]
            ·
            ·   T1 (1800,600)
            ·
            Base (250,250)

        - T2 sits inside the fringe of hazard H1 → brief hazard exposure required
        - T3 is behind obstacle O2 and near hazard H2
        - O1 blocks direct mid-map routing from T1 to T2
        - Drone can fly over obstacles if altitude > obstacle height
        - Total route ≈ 12 km, battery budget ≈ 300 units
        """
        self.base = BaseStation(position=Vec3(250.0, 250.0, 0.0), recharge_radius=80.0)

        self.targets = [
            DeliveryTarget(
                id="T1", position=Vec3(1800.0, 600.0, 30.0),
                urgency=0.6, delivery_radius=50.0,
            ),
            DeliveryTarget(
                id="T2", position=Vec3(4100.0, 2900.0, 50.0),
                urgency=1.0, delivery_radius=50.0,
            ),
            DeliveryTarget(
                id="T3", position=Vec3(1000.0, 4200.0, 20.0),
                urgency=0.8, delivery_radius=50.0,
            ),
        ]

        self.hazards = [
            HazardRegion(
                id="H1", center=Vec3(3800.0, 2600.0, 0.0),
                radius=500.0, severity=0.9,
                height=70.0, growth_rate=0.005,
            ),
            HazardRegion(
                id="H2", center=Vec3(900.0, 3200.0, 0.0),
                radius=400.0, severity=0.7,
                height=55.0, growth_rate=0.008,
            ),
        ]

        self.obstacles = [
            ObstacleVolume(
                id="O1",
                min_corner=Vec3(2200.0, 1000.0, 0.0),
                max_corner=Vec3(2800.0, 2200.0, 120.0),
            ),
            ObstacleVolume(
                id="O2",
                min_corner=Vec3(500.0, 2600.0, 0.0),
                max_corner=Vec3(1500.0, 3000.0, 90.0),
            ),
        ]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset the environment and return the initial observation."""
        if seed is not None:
            random.seed(seed)

        self.drone = DroneState(
            position=Vec3(self.base.position.x, self.base.position.y, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            battery=self.cfg.battery_capacity,
            carrying_payload=True,
            alive=True,
        )

        for t in self.targets:
            t.delivered = False

        for h in self.hazards:
            h.height = h.height * random.uniform(0.85, 1.15)
            h.severity = max(0.3, min(1.0, h.severity + random.uniform(-0.1, 0.1)))
            h.reset()

        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.trace = []
        self._prev_nearest_dist = self._nearest_target_dist()

        obs = self.get_observation()

        self.trace.append(TracePoint(
            step=0,
            position=Vec3(self.drone.position.x, self.drone.position.y, self.drone.position.z),
            velocity=Vec3(0.0, 0.0, 0.0),
            battery=self.drone.battery,
            reward=0.0,
            cumulative_reward=0.0,
            events=["reset"],
            observation=obs,
        ))

        return obs

    def step(self, action: dict[str, Any]) -> tuple[dict, float, bool, dict]:
        """Advance the simulation by one timestep.

        Returns ``(observation, reward, done, info)``.
        """
        if self.done:
            return self.get_observation(), 0.0, True, StepInfo().to_dict()

        self.step_count += 1

        # --- parse & clamp acceleration ---
        accel = Vec3(
            float(action.get("ax", 0.0)),
            float(action.get("ay", 0.0)),
            float(action.get("az", 0.0)),
        ).clamp_magnitude(self.cfg.max_acceleration)

        # --- kinematics (Euler integration) ---
        self.drone.velocity = (
            self.drone.velocity + accel.scale(self.cfg.dt)
        ).clamp_magnitude(self.cfg.max_speed)

        old_pos = Vec3(self.drone.position.x, self.drone.position.y, self.drone.position.z)
        self.drone.position = self.drone.position + self.drone.velocity.scale(self.cfg.dt)

        # clamp to world bounds
        self.drone.position.x = max(0.0, min(self.cfg.world_x, self.drone.position.x))
        self.drone.position.y = max(0.0, min(self.cfg.world_y, self.drone.position.y))
        self.drone.position.z = max(0.0, min(self.cfg.world_z, self.drone.position.z))

        dist_traveled = old_pos.distance_to(self.drone.position)
        elevation_change = abs(self.drone.position.z - old_pos.z)

        # --- battery ---
        drain = self._compute_battery_drain(dist_traveled, elevation_change)
        self.drone.battery -= drain

        # --- advance dynamic hazards ---
        for h in self.hazards:
            h.tick()

        # --- world interactions ---
        collision = self._check_collisions()
        in_hazard, hazard_sev = self._check_hazards()

        delivered_ids: list[str] = []
        if action.get("deliver", False):
            delivered_ids = self._deliver_targets()

        reached_base = (
            self.drone.position.distance_to(self.base.position) <= self.base.recharge_radius
        )
        if action.get("recharge", False) and reached_base:
            self.drone.battery = min(
                self.cfg.battery_capacity,
                self.drone.battery + self.cfg.recharge_rate,
            )

        if self._all_delivered():
            self.drone.carrying_payload = False

        # --- reward ---
        info = StepInfo(
            collision=collision,
            delivered_target_ids=delivered_ids,
            in_hazard=in_hazard,
            hazard_severity=hazard_sev,
            reached_base=reached_base,
            distance_traveled=dist_traveled,
        )
        reward, breakdown = self._compute_reward(info)
        info.reward_breakdown = breakdown
        self.cumulative_reward += reward

        # --- termination ---
        if collision:
            self.drone.alive = False
            self.done = True
        elif self.drone.battery <= 0.0:
            self.drone.battery = 0.0
            self.drone.alive = False
            self.done = True
        elif self._is_success():
            self.done = True
        elif self.step_count >= self.cfg.max_episode_steps:
            self.done = True

        # record trace
        events: list[str] = []
        for tid in delivered_ids:
            events.append(f"delivered_{tid}")
        if collision:
            events.append("collision")
        if in_hazard:
            events.append(f"hazard_{hazard_sev:.2f}")
        if self.drone.battery <= 0.0 and not collision:
            events.append("battery_dead")
        if self._is_success():
            events.append("success")

        obs = self.get_observation()

        self.trace.append(TracePoint(
            step=self.step_count,
            position=Vec3(self.drone.position.x, self.drone.position.y, self.drone.position.z),
            velocity=Vec3(self.drone.velocity.x, self.drone.velocity.y, self.drone.velocity.z),
            battery=self.drone.battery,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            events=events,
            observation=obs,
        ))

        return obs, reward, self.done, info.to_dict()

    # ------------------------------------------------------------------
    # Observation / render
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Compact, RL-friendly observation dict."""
        targets_obs = []
        for t in self.targets:
            rel = t.position - self.drone.position
            targets_obs.append({
                "id": t.id,
                "relative_position": rel.to_dict(),
                "urgency": t.urgency,
                "delivered": t.delivered,
            })

        hazards_obs = []
        for h in self.hazards:
            rel = h.center - self.drone.position
            hazards_obs.append({
                "id": h.id,
                "relative_position": rel.to_dict(),
                "current_height": h._current_height,
                "severity": h.severity,
            })

        return {
            "drone_position": self.drone.position.to_dict(),
            "drone_velocity": self.drone.velocity.to_dict(),
            "battery": round(self.drone.battery, 4),
            "carrying_payload": self.drone.carrying_payload,
            "alive": self.drone.alive,
            "targets": targets_obs,
            "hazards": hazards_obs,
            "step": self.step_count,
            "max_steps": self.cfg.max_episode_steps,
        }

    def render_state(self) -> dict[str, Any]:
        """Rich state dict for future Cesium / frontend rendering."""
        return {
            "base_station": self.base.to_dict(),
            "drone": self.drone.to_dict(),
            "targets": [t.to_dict() for t in self.targets],
            "hazards": [h.to_dict() for h in self.hazards],
            "obstacles": [o.to_dict() for o in self.obstacles],
            "step": self.step_count,
            "max_steps": self.cfg.max_episode_steps,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "done": self.done,
        }

    def get_trace(self) -> dict[str, Any]:
        """Full episode trace for replay / visualisation."""
        return {
            "world": {
                "bounds": {"x": self.cfg.world_x, "y": self.cfg.world_y, "z": self.cfg.world_z},
                "base_station": self.base.to_dict(),
                "targets": [t.to_dict() for t in self.targets],
                "hazards": [h.to_dict() for h in self.hazards],
                "obstacles": [o.to_dict() for o in self.obstacles],
            },
            "trace": [tp.to_dict() for tp in self.trace],
            "summary": {
                "total_steps": self.step_count,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "delivered": [t.id for t in self.targets if t.delivered],
                "alive": self.drone.alive,
                "final_battery": round(self.drone.battery, 4),
                "success": self._is_success(),
            },
        }

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def local_to_latlon(self, vec: Vec3) -> tuple[float, float, float]:
        """Convert local (x, y, z) metres to (lat, lon, alt).

        Uses a flat-earth approximation centred on ``cfg.origin_lat/lon``.
        Accurate enough for small areas (~tens of km) and Cesium plotting.
        """
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * math.cos(math.radians(self.cfg.origin_lat))

        lat = self.cfg.origin_lat + vec.y / meters_per_deg_lat
        lon = self.cfg.origin_lon + vec.x / meters_per_deg_lon
        alt = vec.z
        return (round(lat, 7), round(lon, 7), round(alt, 2))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_battery_drain(self, dist: float, elevation_change: float) -> float:
        return (
            dist * self.cfg.drain_per_meter
            + elevation_change * self.cfg.drain_elevation_factor
            + self.cfg.drain_idle_per_step
        )

    def _check_collisions(self) -> bool:
        for obs in self.obstacles:
            if obs.contains(self.drone.position):
                return True
        return False

    def _check_hazards(self) -> tuple[bool, float]:
        max_sev = 0.0
        in_hazard = False
        for h in self.hazards:
            df = h.danger_factor(self.drone.position)
            if df > 0.0:
                in_hazard = True
                max_sev = max(max_sev, df)
        return in_hazard, max_sev

    def _deliver_targets(self) -> list[str]:
        delivered: list[str] = []
        for t in self.targets:
            if not t.delivered and self.drone.position.distance_to(t.position) <= t.delivery_radius:
                t.delivered = True
                delivered.append(t.id)
        return delivered

    def _all_delivered(self) -> bool:
        return all(t.delivered for t in self.targets)

    def _is_success(self) -> bool:
        return (
            self._all_delivered()
            and self.drone.position.distance_to(self.base.position) <= self.base.recharge_radius
        )

    def _nearest_target_dist(self) -> float:
        """Distance to closest undelivered target, or to base if all done."""
        dists = [
            self.drone.position.distance_to(t.position)
            for t in self.targets
            if not t.delivered
        ]
        if not dists:
            return self.drone.position.distance_to(self.base.position)
        return min(dists)

    def _compute_reward(self, info: StepInfo) -> tuple[float, dict[str, float]]:
        bd: dict[str, float] = {}
        total = 0.0

        # per-step cost of time
        bd["step_penalty"] = -self.cfg.step_penalty
        total += bd["step_penalty"]

        # battery usage cost (proportional to energy spent)
        bd["battery_cost"] = -(
            info.distance_traveled * self.cfg.drain_per_meter * self.cfg.battery_cost_factor
        )
        total += bd["battery_cost"]

        # delivery rewards (scaled by urgency) + progress bonus
        for tid in info.delivered_target_ids:
            tgt = next(t for t in self.targets if t.id == tid)
            r = self.cfg.delivery_reward * (1.0 + tgt.urgency)
            bd[f"delivery_{tid}"] = r
            total += r

        if info.delivered_target_ids:
            n_remaining = sum(1 for t in self.targets if not t.delivered)
            progress_bonus = 50.0 * (1.0 - n_remaining / len(self.targets))
            bd["progress_bonus"] = progress_bonus
            total += progress_bonus

        # collision
        if info.collision:
            bd["collision"] = -self.cfg.collision_penalty
            total += bd["collision"]

        # hazard exposure (severity-weighted)
        if info.in_hazard:
            bd["hazard"] = -self.cfg.hazard_penalty * info.hazard_severity
            total += bd["hazard"]

        # safe return bonus
        if info.reached_base and self._all_delivered():
            bd["return_bonus"] = self.cfg.return_bonus
            total += bd["return_bonus"]

        # distance shaping — nudge toward nearest undelivered target (or base)
        # Skip shaping on delivery steps to avoid a huge negative spike
        # when the nearest-target reference jumps to a farther target.
        # Double the factor when heading home after all deliveries.
        curr_dist = self._nearest_target_dist()
        if info.delivered_target_ids:
            bd["distance_shaping"] = 0.0
            self._prev_nearest_dist = curr_dist
        else:
            factor = self.cfg.distance_shaping_factor
            if self._all_delivered():
                factor *= 2.0
            shaping = (self._prev_nearest_dist - curr_dist) * factor
            bd["distance_shaping"] = shaping
            total += shaping
            self._prev_nearest_dist = curr_dist

        # failure (battery depletion; collision already penalised above)
        if self.drone.battery <= 0.0 and not info.collision:
            bd["failure"] = -self.cfg.failure_penalty
            total += bd["failure"]

        bd["total"] = total
        return total, bd
