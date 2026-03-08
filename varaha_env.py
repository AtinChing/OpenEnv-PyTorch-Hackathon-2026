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
    CylindricalObstacle,
    ResponderUnit,
    ScheduledEvent,
    RESPONDER_STATUSES,
    INTEL_TYPES,
    StepInfo,
    TracePoint,
    MissionInstruction,
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
    collision_penalty: float = 500.0
    hazard_penalty: float = 5.0
    failure_penalty: float = 200.0
    distance_shaping_factor: float = 0.05
    obstacle_proximity_penalty: float = 1.5
    obstacle_proximity_radius: float = 80.0

    # Long-horizon instruction mode (LLM-oriented)
    instruction_mode: bool = False
    instruction_count: int = 60
    sparse_reward_mode: bool = False
    instruction_completion_reward: float = 0.5
    instruction_terminal_success_bonus: float = 2200.0
    instruction_terminal_progress_bonus: float = 800.0
    instruction_violation_penalty: float = 120.0
    instruction_unfinished_penalty: float = 10.0
    available_tools: tuple[str, ...] = (
        "request_intel",
        "battery_forecast",
        "mission_report",
    )

    # California origin anchor (near Sacramento — wildfire-relevant)
    origin_lat: float = 38.55
    origin_lon: float = -121.47


# ---------------------------------------------------------------------------
# Random world generator for domain randomization
# ---------------------------------------------------------------------------

def build_random_world(env: "VarahaEnv") -> None:
    """Legacy easy world gen — kept for backward compatibility."""
    build_hardcore_world(env)


def _hdist(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def build_hardcore_world(env: "VarahaEnv", ultra_hard: bool = False) -> None:
    """Generate an extremely challenging randomized world for serious RL training.

    Features template-based obstacle placement (urban grid, dense forest,
    corridor maze, river valley, fortress, mixed), cylindrical obstacles,
    responder units with dynamic events, and adversarial target placement.

    When ultra_hard=True: denser obstacles, more hazards, more targets, longer episodes.
    """
    cfg = env.cfg
    rng = random

    wx, wy, wz = cfg.world_x, cfg.world_y, cfg.world_z
    margin = 200.0

    def _rpos(z_lo=10.0, z_hi=60.0):
        return Vec3(rng.uniform(margin, wx - margin),
                    rng.uniform(margin, wy - margin),
                    rng.uniform(z_lo, z_hi))

    def _rpos_ground():
        return Vec3(rng.uniform(margin, wx - margin),
                    rng.uniform(margin, wy - margin), 0.0)

    # --- Base station ---
    base_pos = Vec3(rng.uniform(100, wx - 100), rng.uniform(100, wy - 100), 0.0)
    env.base = BaseStation(position=base_pos, recharge_radius=rng.uniform(60, 100))

    # --- Targets (2-5 normal, 3-6 ultra) ---
    if ultra_hard:
        n_targets = rng.choices([3, 4, 5, 6], weights=[0.15, 0.35, 0.35, 0.15])[0]
    else:
        n_targets = rng.choices([2, 3, 4, 5], weights=[0.15, 0.40, 0.30, 0.15])[0]
    targets = []
    for i in range(n_targets):
        for _ in range(120):
            pos = _rpos(z_lo=5.0, z_hi=60.0)
            if _hdist(pos, base_pos) < 500:
                continue
            if all(_hdist(pos, t.position) > 400 for t in targets):
                break
        targets.append(DeliveryTarget(
            id=f"T{i+1}", position=pos,
            urgency=rng.uniform(0.3, 1.0),
            delivery_radius=rng.uniform(70.0, 130.0),
        ))
    env.targets = targets

    # --- Hazards (3-8 normal, 5-10 ultra) with wild variety ---
    if ultra_hard:
        n_hazards = rng.choices([5, 6, 7, 8, 9, 10], weights=[0.10, 0.20, 0.25, 0.25, 0.15, 0.05])[0]
    else:
        n_hazards = rng.choices([3, 4, 5, 6, 7, 8], weights=[0.10, 0.20, 0.25, 0.25, 0.15, 0.05])[0]
    hazards = []
    for i in range(n_hazards):
        center = _rpos_ground()
        fire_type = rng.choice(["tiny_intense", "massive_low", "tall_mid", "standard"])
        if fire_type == "tiny_intense":
            r, sev, ht, gr = rng.uniform(80, 200), rng.uniform(0.9, 1.0), rng.uniform(140, 195), rng.uniform(0.012, 0.025)
        elif fire_type == "massive_low":
            r, sev, ht, gr = rng.uniform(500, 1000), rng.uniform(0.3, 0.5), rng.uniform(25, 50), rng.uniform(0.001, 0.004)
        elif fire_type == "tall_mid":
            r, sev, ht, gr = rng.uniform(250, 500), rng.uniform(0.7, 0.95), rng.uniform(100, 180), rng.uniform(0.008, 0.015)
        else:
            r, sev, ht, gr = rng.uniform(200, 600), rng.uniform(0.4, 0.9), rng.uniform(40, 120), rng.uniform(0.003, 0.012)
        hazards.append(HazardRegion(id=f"H{i+1}", center=center,
                                     radius=r, severity=sev, height=ht, growth_rate=gr))
    env.hazards = hazards

    # --- Obstacle templates ---
    obstacles: list[ObstacleVolume] = []
    cylinders: list[CylindricalObstacle] = []
    oid = [0]

    def _next_oid(prefix="O"):
        oid[0] += 1
        return f"{prefix}{oid[0]}"

    def _add_box(cx, cy, w, h, zt, kind="building"):
        obstacles.append(ObstacleVolume(
            id=_next_oid(), kind=kind,
            min_corner=Vec3(cx - w / 2, cy - h / 2, 0.0),
            max_corner=Vec3(cx + w / 2, cy + h / 2, zt),
        ))

    def _add_cyl(cx, cy, radius, height, kind="tree"):
        cylinders.append(CylindricalObstacle(
            id=_next_oid("C"), kind=kind,
            center=Vec3(cx, cy, 0.0), radius=radius, height=height,
        ))

    if ultra_hard:
        template = rng.choices(["urban_grid", "dense_forest", "corridor_maze",
                               "river_valley", "fortress", "mixed"],
                              weights=[0.08, 0.12, 0.12, 0.10, 0.10, 0.48])[0]
    else:
        template = rng.choice(["urban_grid", "dense_forest", "corridor_maze",
                               "river_valley", "fortress", "mixed"])

    # ---- URBAN GRID: rows and columns of buildings ----
    if template == "urban_grid" or template == "mixed":
        ox = rng.uniform(500, 1500)
        oy = rng.uniform(500, 1500)
        rows = rng.randint(2, 5) if ultra_hard else rng.randint(2, 4)
        cols = rng.randint(3, 6) if ultra_hard else rng.randint(3, 5)
        spacing = rng.uniform(300, 550) if ultra_hard else rng.uniform(350, 600)
        for r in range(rows):
            for c in range(cols):
                bx = ox + c * spacing + rng.uniform(-80, 80)
                by = oy + r * spacing + rng.uniform(-80, 80)
                if bx < margin or bx > wx - margin or by < margin or by > wy - margin:
                    continue
                bw = rng.uniform(80, 300)
                bh = rng.uniform(80, 300)
                bzt = rng.choice([rng.uniform(30, 60), rng.uniform(100, 195)])
                _add_box(bx, by, bw, bh, bzt)
                if rng.random() < (0.45 if ultra_hard else 0.3):
                    arm_dir = rng.choice(["east", "north"])
                    if arm_dir == "east":
                        _add_box(bx + bw / 2 + 40, by, 80, bh * 0.6, bzt * 0.9)
                    else:
                        _add_box(bx, by + bh / 2 + 40, bw * 0.6, 80, bzt * 0.9)

    # ---- DENSE FOREST: many cylindrical trees ----
    if template == "dense_forest" or template == "mixed":
        forest_cx = rng.uniform(800, wx - 800)
        forest_cy = rng.uniform(800, wy - 800)
        n_trees = rng.randint(25, 60) if ultra_hard else rng.randint(15, 40)
        for _ in range(n_trees):
            tx = forest_cx + rng.gauss(0, 600)
            ty = forest_cy + rng.gauss(0, 600)
            tx = max(margin, min(wx - margin, tx))
            ty = max(margin, min(wy - margin, ty))
            tree_type = rng.choice(["pine", "oak", "palm", "dead"])
            if tree_type == "pine":
                _add_cyl(tx, ty, rng.uniform(8, 20), rng.uniform(40, 100), "tree_pine")
            elif tree_type == "oak":
                _add_cyl(tx, ty, rng.uniform(15, 40), rng.uniform(25, 60), "tree_oak")
            elif tree_type == "palm":
                _add_cyl(tx, ty, rng.uniform(5, 12), rng.uniform(30, 80), "tree_palm")
            else:
                _add_cyl(tx, ty, rng.uniform(10, 25), rng.uniform(20, 50), "tree_dead")

    # ---- CORRIDOR MAZE: parallel walls with gaps ----
    if template == "corridor_maze" or template == "mixed":
        maze_ox = rng.uniform(400, wx / 2)
        maze_oy = rng.uniform(400, wy / 2)
        n_walls = rng.randint(6, 12) if ultra_hard else rng.randint(4, 8)
        wall_dir = rng.choice(["horizontal", "vertical"])
        spacing = rng.uniform(200, 500)
        for w in range(n_walls):
            wl = rng.uniform(400, 1500)
            wt = rng.uniform(40, 80)
            wzt = rng.uniform(100, 195)
            if wall_dir == "horizontal":
                wy_pos = maze_oy + w * spacing
                if wy_pos > wy - margin:
                    continue
                _add_box(maze_ox + wl / 2, wy_pos, wl, wt, wzt, "wall")
                gap_x = maze_ox + rng.uniform(0.2, 0.8) * wl
                _add_box(gap_x, wy_pos, rng.uniform(80, 200), wt, 0, "gap")
            else:
                wx_pos = maze_ox + w * spacing
                if wx_pos > wx - margin:
                    continue
                _add_box(wx_pos, maze_oy + wl / 2, wt, wl, wzt, "wall")

    # ---- RIVER VALLEY: chain of low flat boxes + scattered trees ----
    if template == "river_valley" or (template == "mixed" and rng.random() < (0.7 if ultra_hard else 0.5)):
        river_start_x = rng.uniform(margin, wx / 3)
        river_y = rng.uniform(wy * 0.3, wy * 0.7)
        n_segs = rng.randint(10, 18) if ultra_hard else rng.randint(6, 12)
        for seg in range(n_segs):
            seg_x = river_start_x + seg * rng.uniform(200, 400)
            seg_y = river_y + rng.gauss(0, 150)
            if seg_x > wx - margin:
                break
            seg_y = max(margin, min(wy - margin, seg_y))
            _add_box(seg_x, seg_y, rng.uniform(200, 400), rng.uniform(60, 150),
                     rng.uniform(3, 10), "river")
            for _ in range(rng.randint(2, 6) if ultra_hard else rng.randint(1, 4)):
                bank_offset = rng.choice([-1, 1]) * rng.uniform(100, 300)
                _add_cyl(seg_x + rng.uniform(-100, 100),
                          seg_y + bank_offset,
                          rng.uniform(8, 20), rng.uniform(30, 80), "tree_bank")

    # ---- FORTRESS: walls surrounding a target area ----
    if template == "fortress" or (template == "mixed" and rng.random() < (0.6 if ultra_hard else 0.4)):
        if targets:
            fort_target = rng.choice(targets)
            ftx, fty = fort_target.position.x, fort_target.position.y
            wall_half = rng.uniform(250, 500)
            wall_zt = rng.uniform(120, 190)
            wall_thick = rng.uniform(50, 80)
            _add_box(ftx, fty - wall_half, wall_half * 2, wall_thick, wall_zt, "fortress_wall")
            _add_box(ftx, fty + wall_half, wall_half * 2, wall_thick, wall_zt, "fortress_wall")
            _add_box(ftx - wall_half, fty, wall_thick, wall_half * 2, wall_zt, "fortress_wall")
            _add_box(ftx + wall_half, fty, wall_thick, wall_half * 2, wall_zt, "fortress_wall")

    # ---- Always scatter some light poles and random pillars ----
    n_poles = rng.randint(6, 18) if ultra_hard else rng.randint(3, 10)
    for _ in range(n_poles):
        px = rng.uniform(margin, wx - margin)
        py = rng.uniform(margin, wy - margin)
        _add_cyl(px, py, rng.uniform(2, 6), rng.uniform(30, 80), "light_pole")

    n_pillars = rng.randint(4, 12) if ultra_hard else rng.randint(2, 6)
    for _ in range(n_pillars):
        px = rng.uniform(margin, wx - margin)
        py = rng.uniform(margin, wy - margin)
        _add_cyl(px, py, rng.uniform(15, 50), rng.uniform(80, 195), "pillar")

    obstacles = [o for o in obstacles if o.max_corner.z > 1.0]
    env.obstacles = obstacles
    env.cylinders = cylinders

    # --- Responder units (1 per target, up to 5 in ultra) ---
    responders = []
    max_resp = 5 if ultra_hard else 4
    for i, tgt in enumerate(targets[:max_resp]):
        r = ResponderUnit(
            id=f"R{i+1}",
            position=Vec3(tgt.position.x + rng.uniform(-50, 50),
                          tgt.position.y + rng.uniform(-50, 50), 0.0),
            linked_target_id=tgt.id,
            status="stable",
            current_need=rng.choice(["supplies", "medical", "evacuation", "water"]),
            can_update_dropzone=rng.random() < 0.5,
            active=True,
        )
        events = []

        if rng.random() < 0.7:
            events.append(ScheduledEvent(
                step=rng.randint(100, 600),
                event_type="urgency_update",
                payload={"new_urgency": rng.uniform(0.5, 1.0)},
            ))

        if r.can_update_dropzone and rng.random() < 0.5:
            events.append(ScheduledEvent(
                step=rng.randint(200, 800),
                event_type="dropzone_relocation",
                payload={"dx": rng.uniform(-200, 200), "dy": rng.uniform(-200, 200)},
            ))

        if rng.random() < 0.6:
            intel = rng.choice([
                "blocked_north", "blocked_south", "blocked_east", "blocked_west",
                "safe_north", "safe_south", "safe_east", "safe_west",
                "fire_expanded", "fire_receded",
            ])
            events.append(ScheduledEvent(
                step=rng.randint(50, 500),
                event_type="hazard_intel",
                payload={"intel": intel, "severity": rng.uniform(0.3, 1.0)},
            ))

        r.scheduled_events = events
        responders.append(r)
    env.responders = responders


def build_hardcore_world_v2(env: "VarahaEnv") -> None:
    """Ultra-hard variant: denser obstacles, more hazards, more targets."""
    build_hardcore_world(env, ultra_hard=True)


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
            "tool_call": str,  # optional: request_intel | battery_forecast | mission_report
        }

    Returns ``(obs_dict, reward, done, info_dict)`` per OpenAI-gym convention.
    """

    def __init__(self, config: Optional[VarahaConfig] = None,
                 world_fn: Optional[Any] = None) -> None:
        self.cfg = config or VarahaConfig()
        self._world_fn = world_fn

        self.base: BaseStation
        self.drone: DroneState
        self.targets: list[DeliveryTarget] = []
        self.hazards: list[HazardRegion] = []
        self.obstacles: list[ObstacleVolume] = []
        self.cylinders: list[CylindricalObstacle] = []
        self.responders: list[ResponderUnit] = []

        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.trace: list[TracePoint] = []

        self._prev_nearest_dist: float = 0.0
        self._hazard_base_heights: list[float] = []
        self._hazard_base_severities: list[float] = []
        self.instructions: list[MissionInstruction] = []
        self._instruction_cursor: int = 0
        self._instruction_violations: int = 0
        self._tool_history: list[str] = []
        self._last_tool_result: dict[str, Any] = {}
        self._instruction_progress_reward: float = 0.0

        self._rebuild_world()

    def _rebuild_world(self):
        if self._world_fn is not None:
            self._world_fn(self)
        else:
            self._build_demo_world()
        self._hazard_base_heights = [h.height for h in self.hazards]
        self._hazard_base_severities = [h.severity for h in self.hazards]

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
                urgency=0.6, delivery_radius=80.0,
            ),
            DeliveryTarget(
                id="T2", position=Vec3(4100.0, 2900.0, 50.0),
                urgency=1.0, delivery_radius=120.0,
            ),
            DeliveryTarget(
                id="T3", position=Vec3(1000.0, 4200.0, 20.0),
                urgency=0.8, delivery_radius=100.0,
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

        if self._world_fn is not None:
            self._rebuild_world()

        self.drone = DroneState(
            position=Vec3(self.base.position.x, self.base.position.y, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            battery=self.cfg.battery_capacity,
            carrying_payload=True,
            alive=True,
        )

        for t in self.targets:
            t.delivered = False

        for i, h in enumerate(self.hazards):
            h.height = self._hazard_base_heights[i] * random.uniform(0.85, 1.15)
            h.severity = max(0.3, min(1.0, self._hazard_base_severities[i] + random.uniform(-0.1, 0.1)))
            h.reset()

        for r in self.responders:
            r.active = True
            r.status = "stable"
            r.latest_intel = "none"
            r.intel_severity = 0.0
            r.message = ""
            for ev in r.scheduled_events:
                ev.fired = False

        self._target_base_positions = {
            t.id: Vec3(t.position.x, t.position.y, t.position.z)
            for t in self.targets
        }
        self._build_instruction_program()
        self._instruction_progress_reward = 0.0
        self._last_tool_result = {}
        self._tool_history = []

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

        # --- advance responder events ---
        self._tick_responders()

        # --- world interactions ---
        collision = self._check_collisions()
        in_hazard, hazard_sev = self._check_hazards()

        tool_call = ""
        tool_result: dict[str, Any] = {}
        raw_tool_call = action.get("tool_call")
        if raw_tool_call is not None and str(raw_tool_call).strip():
            tool_call, tool_result = self._execute_tool_call(str(raw_tool_call).strip())

        prev_instruction_cursor = self._instruction_cursor
        delivered_ids: list[str] = []
        if action.get("deliver", False):
            delivered_ids = self._deliver_targets()

        reached_base = (
            ((self.drone.position.x - self.base.position.x) ** 2
             + (self.drone.position.y - self.base.position.y) ** 2) ** 0.5
            <= self.base.recharge_radius
        )
        if action.get("recharge", False) and reached_base:
            self.drone.battery = min(
                self.cfg.battery_capacity,
                self.drone.battery + self.cfg.recharge_rate,
            )

        self._update_instruction_progress(
            delivered_ids=delivered_ids,
            reached_base=reached_base,
            tool_call=tool_call,
        )
        completed_now = max(0, self._instruction_cursor - prev_instruction_cursor)

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
            tool_call=tool_call,
            tool_result=tool_result,
            instruction_completed=self._instruction_cursor,
            instruction_total=len(self.instructions),
            instruction_violations=self._instruction_violations,
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
        if tool_call:
            events.append(f"tool_{tool_call}")
        if completed_now > 0:
            events.append(f"instruction+{completed_now}")

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
        dp = self.drone.position

        targets_obs = []
        for t in self.targets:
            rel = t.position - dp
            targets_obs.append({
                "id": t.id,
                "relative_position": rel.to_dict(),
                "urgency": t.urgency,
                "delivered": t.delivered,
            })

        hazards_obs = []
        for h in self.hazards:
            rel = h.center - dp
            hazards_obs.append({
                "id": h.id,
                "relative_position": rel.to_dict(),
                "current_height": h._current_height,
                "severity": h.severity,
            })

        obstacles_obs = []
        for obs in self.obstacles:
            c = obs.center
            hs = obs.half_size
            rel = c - dp
            dist = dp.horizontal_distance_to(c)
            obstacles_obs.append({
                "type": "box",
                "relative_position": rel.to_dict(),
                "height": obs.height,
                "size_x": hs.x * 2,
                "size_y": hs.y * 2,
                "distance": dist,
                "kind": obs.kind,
            })
        for cyl in self.cylinders:
            rel = cyl.center - dp
            dist = dp.horizontal_distance_to(cyl.center)
            obstacles_obs.append({
                "type": "cylinder",
                "relative_position": rel.to_dict(),
                "height": cyl.height,
                "size_x": cyl.radius * 2,
                "size_y": cyl.radius * 2,
                "distance": dist,
                "kind": cyl.kind,
            })
        obstacles_obs.sort(key=lambda o: o["distance"])

        responders_obs = []
        for r in self.responders:
            if not r.active:
                continue
            rel = r.position - dp
            intel_dir = r.intel_direction()
            responders_obs.append({
                "id": r.id,
                "relative_position": rel.to_dict(),
                "linked_target_id": r.linked_target_id,
                "status": r.status,
                "status_code": r.status_code(),
                "latest_intel": r.latest_intel,
                "intel_direction": {"x": intel_dir[0], "y": intel_dir[1]},
                "intel_severity": r.intel_severity,
            })

        mission_obs = self._instruction_snapshot()
        return {
            "drone_position": dp.to_dict(),
            "drone_velocity": self.drone.velocity.to_dict(),
            "battery": round(self.drone.battery, 4),
            "carrying_payload": self.drone.carrying_payload,
            "alive": self.drone.alive,
            "targets": targets_obs,
            "hazards": hazards_obs,
            "obstacles": obstacles_obs,
            "responders": responders_obs,
            "mission": mission_obs,
            "last_tool_result": self._last_tool_result,
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
            "cylinders": [c.to_dict() for c in self.cylinders],
            "responders": [r.to_dict() for r in self.responders],
            "mission": self._instruction_snapshot(include_full=True),
            "tool_history": list(self._tool_history),
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
                "cylinders": [c.to_dict() for c in self.cylinders],
                "responders": [r.to_dict() for r in self.responders],
                "mission": self._instruction_snapshot(include_full=True),
            },
            "trace": [tp.to_dict() for tp in self.trace],
            "summary": {
                "total_steps": self.step_count,
                "cumulative_reward": round(self.cumulative_reward, 4),
                "delivered": [t.id for t in self.targets if t.delivered],
                "alive": self.drone.alive,
                "final_battery": round(self.drone.battery, 4),
                "success": self._is_success(),
                "instruction_completed": self._instruction_cursor,
                "instruction_total": len(self.instructions),
                "instruction_violations": self._instruction_violations,
                "tool_calls": list(self._tool_history),
            },
        }

    # ------------------------------------------------------------------
    # Long-horizon instruction mode
    # ------------------------------------------------------------------

    def _build_instruction_program(self) -> None:
        self.instructions = []
        self._instruction_cursor = 0
        self._instruction_violations = 0

        if not self.cfg.instruction_mode or not self.targets:
            return

        ordered_targets = sorted(self.targets, key=lambda t: (-t.urgency, t.id))
        target_count = len(ordered_targets)
        desired_len = self.cfg.instruction_count if self.cfg.instruction_count > 0 else (target_count * 3 + 1)
        desired_len = max(desired_len, target_count * 2 + 1)

        instructions: list[MissionInstruction] = []
        inst_idx = 1
        cycle = 0
        while len(instructions) < max(desired_len - 1, 1):
            for tgt in ordered_targets:
                if len(instructions) >= max(desired_len - 1, 1):
                    break
                instructions.append(
                    MissionInstruction(
                        id=f"I{inst_idx}",
                        kind="deliver_target",
                        description=f"Cycle {cycle + 1}: deliver to {tgt.id} in order.",
                        target_id=tgt.id,
                    )
                )
                inst_idx += 1
                if len(instructions) >= max(desired_len - 1, 1):
                    break
                tool = "request_intel" if (cycle % 2 == 0) else "battery_forecast"
                instructions.append(
                    MissionInstruction(
                        id=f"I{inst_idx}",
                        kind="tool_call",
                        description=f"Call {tool} after servicing {tgt.id}.",
                        target_id=tgt.id,
                        tool_name=tool,
                    )
                )
                inst_idx += 1
            cycle += 1

        instructions.append(
            MissionInstruction(
                id=f"I{inst_idx}",
                kind="return_base",
                description="Return to base only after all deliveries are completed.",
            )
        )
        self.instructions = instructions

    def _current_instruction(self) -> Optional[MissionInstruction]:
        if self._instruction_cursor >= len(self.instructions):
            return None
        return self.instructions[self._instruction_cursor]

    def _instruction_snapshot(self, include_full: bool = False) -> dict[str, Any]:
        total = len(self.instructions)
        completed = min(self._instruction_cursor, total)
        next_instruction = self._current_instruction()
        out: dict[str, Any] = {
            "enabled": self.cfg.instruction_mode,
            "total": total,
            "completed": completed,
            "remaining": max(total - completed, 0),
            "progress": (completed / total) if total > 0 else 1.0,
            "violations": self._instruction_violations,
            "next_instruction": next_instruction.to_dict() if next_instruction else None,
        }
        if include_full:
            out["instructions"] = [inst.to_dict() for inst in self.instructions]
        return out

    def _complete_current_instruction(self) -> None:
        inst = self._current_instruction()
        if inst is None:
            return
        inst.completed = True
        self._instruction_cursor += 1
        self._instruction_progress_reward += self.cfg.instruction_completion_reward

    def _record_instruction_violation(self) -> None:
        self._instruction_violations += 1
        inst = self._current_instruction()
        if inst is not None:
            inst.violated = True

    def _tool_matches_instruction(self, tool_call: str, inst: MissionInstruction) -> bool:
        base, _, arg = tool_call.partition(":")
        if base != inst.tool_name:
            return False
        if inst.target_id and arg and arg != inst.target_id:
            return False
        return True

    def _update_instruction_progress(
        self,
        delivered_ids: list[str],
        reached_base: bool,
        tool_call: str,
    ) -> None:
        if not self.cfg.instruction_mode or not self.instructions:
            return

        inst = self._current_instruction()
        if inst and inst.kind == "deliver_target":
            for tid in delivered_ids:
                if tid != inst.target_id:
                    self._record_instruction_violation()

        while True:
            inst = self._current_instruction()
            if inst is None:
                break

            if inst.kind == "deliver_target":
                if inst.target_id in delivered_ids:
                    self._complete_current_instruction()
                    continue
                break

            if inst.kind == "tool_call":
                if not tool_call:
                    break
                if self._tool_matches_instruction(tool_call, inst):
                    self._complete_current_instruction()
                else:
                    self._record_instruction_violation()
                break

            if inst.kind == "return_base":
                if reached_base and self._all_delivered():
                    self._complete_current_instruction()
                break

            break

    def _execute_tool_call(self, tool_call: str) -> tuple[str, dict[str, Any]]:
        raw = tool_call.strip().lower()
        if not raw:
            return "", {}

        tool_name, _, arg = raw.partition(":")
        normalized_call = f"{tool_name}:{arg}" if arg else tool_name

        if tool_name not in self.cfg.available_tools:
            result = {"ok": False, "error": f"unsupported_tool:{tool_name}"}
            self._tool_history.append(normalized_call)
            self._last_tool_result = result
            return normalized_call, result

        if tool_name == "request_intel":
            responder = None
            if arg:
                responder = next(
                    (r for r in self.responders if r.active and r.linked_target_id.lower() == arg.lower()),
                    None,
                )
            if responder is None:
                responder = next((r for r in self.responders if r.active), None)
            if responder is None:
                result = {"ok": True, "intel": "none", "message": "no_active_responders"}
            else:
                result = {
                    "ok": True,
                    "intel": responder.latest_intel,
                    "intel_severity": round(responder.intel_severity, 3),
                    "responder_id": responder.id,
                    "target_id": responder.linked_target_id,
                    "message": responder.message,
                }
        elif tool_name == "battery_forecast":
            burn = max(self.cfg.drain_per_meter, 1e-6)
            est_range = self.drone.battery / burn
            result = {
                "ok": True,
                "battery": round(self.drone.battery, 3),
                "estimated_range_m": round(est_range, 1),
            }
        else:  # mission_report
            result = {
                "ok": True,
                "delivered": [t.id for t in self.targets if t.delivered],
                "remaining": [t.id for t in self.targets if not t.delivered],
                "instruction_progress": round(self._instruction_snapshot()["progress"], 3),
                "violations": self._instruction_violations,
            }

        self._tool_history.append(normalized_call)
        self._last_tool_result = result
        return normalized_call, result

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
        for cyl in self.cylinders:
            if cyl.contains(self.drone.position):
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
        """Cylindrical delivery check — drone must be within horizontal radius
        and above the target (within a generous altitude window for drops)."""
        delivered: list[str] = []
        for t in self.targets:
            if t.delivered:
                continue
            dx = self.drone.position.x - t.position.x
            dy = self.drone.position.y - t.position.y
            horiz_dist = (dx * dx + dy * dy) ** 0.5
            alt_above = self.drone.position.z - t.position.z
            if horiz_dist <= t.delivery_radius and -10.0 <= alt_above <= t.delivery_radius * 2:
                t.delivered = True
                delivered.append(t.id)
        return delivered

    def _all_delivered(self) -> bool:
        return all(t.delivered for t in self.targets)

    def _is_success(self) -> bool:
        hdist = ((self.drone.position.x - self.base.position.x) ** 2
                 + (self.drone.position.y - self.base.position.y) ** 2) ** 0.5
        return self._all_delivered() and hdist <= self.base.recharge_radius

    def _nearest_target_dist(self) -> float:
        """Horizontal distance to closest undelivered target, or to base if all done."""
        dists = [
            ((self.drone.position.x - t.position.x) ** 2
             + (self.drone.position.y - t.position.y) ** 2) ** 0.5
            for t in self.targets
            if not t.delivered
        ]
        if not dists:
            return ((self.drone.position.x - self.base.position.x) ** 2
                    + (self.drone.position.y - self.base.position.y) ** 2) ** 0.5
        return min(dists)

    def _tick_responders(self) -> None:
        """Process scheduled responder events for the current step."""
        for r in self.responders:
            if not r.active:
                continue
            for ev in r.scheduled_events:
                if ev.fired or ev.step != self.step_count:
                    continue
                ev.fired = True
                etype = ev.event_type

                if etype == "urgency_update":
                    tgt = self._find_target(r.linked_target_id)
                    if tgt and not tgt.delivered:
                        tgt.urgency = max(0.1, min(1.0, ev.payload.get("new_urgency", tgt.urgency)))
                        r.status = "critical" if tgt.urgency >= 0.9 else "urgent" if tgt.urgency >= 0.6 else "stable"
                        r.message = f"urgency->{tgt.urgency:.1f}"

                elif etype == "dropzone_relocation":
                    tgt = self._find_target(r.linked_target_id)
                    if tgt and not tgt.delivered and r.can_update_dropzone:
                        dx = ev.payload.get("dx", 0.0)
                        dy = ev.payload.get("dy", 0.0)
                        tgt.position.x = max(50, min(self.cfg.world_x - 50, tgt.position.x + dx))
                        tgt.position.y = max(50, min(self.cfg.world_y - 50, tgt.position.y + dy))
                        r.position = Vec3(tgt.position.x, tgt.position.y, 0.0)
                        r.message = f"dropzone moved ({dx:+.0f},{dy:+.0f})"
                        self._prev_nearest_dist = self._nearest_target_dist()

                elif etype == "hazard_intel":
                    r.latest_intel = ev.payload.get("intel", "none")
                    r.intel_severity = ev.payload.get("severity", 0.5)
                    r.message = f"intel: {r.latest_intel}"

    def _find_target(self, tid: str) -> Optional[DeliveryTarget]:
        for t in self.targets:
            if t.id == tid:
                return t
        return None

    def _obstacle_proximity_penalty(self) -> float:
        """Graduated penalty for flying close to any obstacle surface."""
        min_dist = float("inf")
        pos = self.drone.position
        for obs in self.obstacles:
            d = obs.nearest_surface_dist(pos)
            if d < min_dist:
                min_dist = d
        for cyl in self.cylinders:
            d = cyl.nearest_surface_dist(pos)
            if d < min_dist:
                min_dist = d
        if min_dist >= self.cfg.obstacle_proximity_radius:
            return 0.0
        factor = 1.0 - min_dist / self.cfg.obstacle_proximity_radius
        return self.cfg.obstacle_proximity_penalty * factor * factor

    def _compute_reward(self, info: StepInfo) -> tuple[float, dict[str, float]]:
        if self.cfg.instruction_mode and self.cfg.sparse_reward_mode:
            return self._compute_sparse_instruction_reward(info)

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

        if self._instruction_progress_reward > 0.0:
            bd["instruction_progress"] = self._instruction_progress_reward
            total += bd["instruction_progress"]
            self._instruction_progress_reward = 0.0

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

        # obstacle proximity (graduated — discourages flying close)
        prox = self._obstacle_proximity_penalty()
        if prox > 0:
            bd["obstacle_proximity"] = -prox
            total -= prox

        # failure (battery depletion; collision already penalised above)
        if self.drone.battery <= 0.0 and not info.collision:
            bd["failure"] = -self.cfg.failure_penalty
            total += bd["failure"]

        bd["total"] = total
        return total, bd

    def _compute_sparse_instruction_reward(self, info: StepInfo) -> tuple[float, dict[str, float]]:
        bd: dict[str, float] = {}
        total = 0.0

        # Keep shaping intentionally small in sparse mode.
        bd["step_penalty"] = -(self.cfg.step_penalty * 0.25)
        total += bd["step_penalty"]

        if self._instruction_progress_reward > 0.0:
            bd["instruction_progress"] = self._instruction_progress_reward
            total += bd["instruction_progress"]
            self._instruction_progress_reward = 0.0

        if info.in_hazard:
            bd["hazard"] = -(self.cfg.hazard_penalty * 0.2 * info.hazard_severity)
            total += bd["hazard"]

        terminal = (
            info.collision
            or self.drone.battery <= 0.0
            or self._is_success()
            or self.step_count >= self.cfg.max_episode_steps
        )
        if terminal:
            total_instr = len(self.instructions)
            progress = (self._instruction_cursor / total_instr) if total_instr > 0 else 1.0
            bd["terminal_progress"] = self.cfg.instruction_terminal_progress_bonus * progress
            total += bd["terminal_progress"]

            if self._is_success():
                bd["terminal_success"] = self.cfg.instruction_terminal_success_bonus
                total += bd["terminal_success"]
            else:
                bd["terminal_failure"] = -self.cfg.failure_penalty
                total += bd["terminal_failure"]

            remaining = max(total_instr - self._instruction_cursor, 0)
            if remaining > 0:
                bd["unfinished_penalty"] = -remaining * self.cfg.instruction_unfinished_penalty
                total += bd["unfinished_penalty"]

            if self._instruction_violations > 0:
                bd["instruction_violations"] = (
                    -self._instruction_violations * self.cfg.instruction_violation_penalty
                )
                total += bd["instruction_violations"]

            if info.collision:
                bd["collision"] = -self.cfg.collision_penalty
                total += bd["collision"]

        bd["total"] = total
        return total, bd
