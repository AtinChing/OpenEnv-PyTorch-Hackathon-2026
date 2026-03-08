#!/usr/bin/env python3
"""Evaluate trained PPO agent on held-out test environments."""

import argparse
import heapq
import json
import os
from typing import Any, Optional

import numpy as np
from stable_baselines3 import PPO

from sim_types import Vec3, BaseStation, DeliveryTarget, HazardRegion, ObstacleVolume
from sb3_env_wrapper import VarahaSB3Env


# ---------------------------------------------------------------------------
# Test map definitions
# ---------------------------------------------------------------------------

def map_reversed(env):
    """Map 1 — Reversed: base top-right, targets in bottom-left quadrant."""
    env.base = BaseStation(position=Vec3(4700.0, 4700.0, 0.0), recharge_radius=80.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(3200.0, 3800.0, 25.0),
                       urgency=0.7, delivery_radius=90.0),
        DeliveryTarget(id="T2", position=Vec3(1000.0, 2000.0, 40.0),
                       urgency=1.0, delivery_radius=110.0),
        DeliveryTarget(id="T3", position=Vec3(3500.0, 800.0, 15.0),
                       urgency=0.5, delivery_radius=100.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(1200.0, 2200.0, 0.0),
                     radius=450.0, severity=0.8, height=65.0, growth_rate=0.006),
        HazardRegion(id="H2", center=Vec3(3000.0, 1200.0, 0.0),
                     radius=350.0, severity=0.6, height=50.0, growth_rate=0.004),
    ]
    env.obstacles = [
        ObstacleVolume(id="O1",
                       min_corner=Vec3(2000.0, 2800.0, 0.0),
                       max_corner=Vec3(2600.0, 3400.0, 100.0)),
        ObstacleVolume(id="O2",
                       min_corner=Vec3(3800.0, 1500.0, 0.0),
                       max_corner=Vec3(4400.0, 2000.0, 80.0)),
    ]


def map_fire_corridor(env):
    """Map 2 — Fire Corridor: two fires form a narrow gap the drone must navigate."""
    env.base = BaseStation(position=Vec3(200.0, 2500.0, 0.0), recharge_radius=80.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(1500.0, 2500.0, 30.0),
                       urgency=0.6, delivery_radius=90.0),
        DeliveryTarget(id="T2", position=Vec3(3500.0, 2500.0, 45.0),
                       urgency=0.9, delivery_radius=100.0),
        DeliveryTarget(id="T3", position=Vec3(4800.0, 2500.0, 20.0),
                       urgency=1.0, delivery_radius=110.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(2500.0, 1800.0, 0.0),
                     radius=600.0, severity=0.9, height=75.0, growth_rate=0.007),
        HazardRegion(id="H2", center=Vec3(2500.0, 3200.0, 0.0),
                     radius=600.0, severity=0.85, height=70.0, growth_rate=0.005),
    ]
    env.obstacles = [
        ObstacleVolume(id="O1",
                       min_corner=Vec3(3000.0, 2200.0, 0.0),
                       max_corner=Vec3(3300.0, 2800.0, 110.0)),
    ]


def map_dense_obstacles(env):
    """Map 3 — Dense Urban: many obstacles creating a maze-like environment."""
    env.base = BaseStation(position=Vec3(250.0, 250.0, 0.0), recharge_radius=80.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(2000.0, 1500.0, 35.0),
                       urgency=0.8, delivery_radius=100.0),
        DeliveryTarget(id="T2", position=Vec3(3500.0, 3500.0, 40.0),
                       urgency=0.7, delivery_radius=100.0),
        DeliveryTarget(id="T3", position=Vec3(4500.0, 1000.0, 25.0),
                       urgency=1.0, delivery_radius=110.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(3200.0, 3200.0, 0.0),
                     radius=400.0, severity=0.7, height=55.0, growth_rate=0.003),
        HazardRegion(id="H2", center=Vec3(4200.0, 800.0, 0.0),
                     radius=350.0, severity=0.8, height=60.0, growth_rate=0.006),
    ]
    env.obstacles = [
        ObstacleVolume(id="O1",
                       min_corner=Vec3(1000.0, 800.0, 0.0),
                       max_corner=Vec3(1400.0, 1200.0, 130.0)),
        ObstacleVolume(id="O2",
                       min_corner=Vec3(2500.0, 2000.0, 0.0),
                       max_corner=Vec3(3000.0, 2500.0, 100.0)),
        ObstacleVolume(id="O3",
                       min_corner=Vec3(1800.0, 3000.0, 0.0),
                       max_corner=Vec3(2200.0, 3600.0, 90.0)),
        ObstacleVolume(id="O4",
                       min_corner=Vec3(3800.0, 1500.0, 0.0),
                       max_corner=Vec3(4200.0, 2000.0, 110.0)),
    ]


def map_long_range(env):
    """Map 4 — Long Range: targets at extreme corners, battery management critical."""
    env.base = BaseStation(position=Vec3(2500.0, 2500.0, 0.0), recharge_radius=100.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(300.0, 300.0, 20.0),
                       urgency=0.5, delivery_radius=100.0),
        DeliveryTarget(id="T2", position=Vec3(4700.0, 4700.0, 35.0),
                       urgency=1.0, delivery_radius=100.0),
        DeliveryTarget(id="T3", position=Vec3(4700.0, 300.0, 30.0),
                       urgency=0.9, delivery_radius=100.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(1500.0, 1500.0, 0.0),
                     radius=500.0, severity=0.8, height=80.0, growth_rate=0.004),
        HazardRegion(id="H2", center=Vec3(3800.0, 3800.0, 0.0),
                     radius=450.0, severity=0.75, height=60.0, growth_rate=0.005),
    ]
    env.obstacles = [
        ObstacleVolume(id="O1",
                       min_corner=Vec3(2000.0, 800.0, 0.0),
                       max_corner=Vec3(3000.0, 1200.0, 100.0)),
    ]


def map_clustered_fires(env):
    """Map 5 — Clustered Fires: overlapping hazards, high-urgency target deep inside."""
    env.base = BaseStation(position=Vec3(100.0, 100.0, 0.0), recharge_radius=80.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(1200.0, 1200.0, 25.0),
                       urgency=0.4, delivery_radius=90.0),
        DeliveryTarget(id="T2", position=Vec3(2800.0, 2800.0, 50.0),
                       urgency=1.0, delivery_radius=130.0),
        DeliveryTarget(id="T3", position=Vec3(4000.0, 1500.0, 30.0),
                       urgency=0.7, delivery_radius=100.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(2500.0, 2500.0, 0.0),
                     radius=500.0, severity=1.0, height=85.0, growth_rate=0.008),
        HazardRegion(id="H2", center=Vec3(3000.0, 3000.0, 0.0),
                     radius=400.0, severity=0.9, height=75.0, growth_rate=0.006),
    ]
    env.obstacles = [
        ObstacleVolume(id="O1",
                       min_corner=Vec3(1800.0, 1800.0, 0.0),
                       max_corner=Vec3(2200.0, 2200.0, 120.0)),
        ObstacleVolume(id="O2",
                       min_corner=Vec3(3500.0, 900.0, 0.0),
                       max_corner=Vec3(4000.0, 1300.0, 80.0)),
    ]


def map_single_target(env):
    """Map 6 — Single Target: one delivery through a fire zone."""
    env.base = BaseStation(position=Vec3(500.0, 500.0, 0.0), recharge_radius=90.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(4000.0, 4000.0, 40.0),
                       urgency=1.0, delivery_radius=110.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(2500.0, 2500.0, 0.0),
                     radius=600.0, severity=0.85, height=70.0, growth_rate=0.005),
    ]
    env.obstacles = [
        ObstacleVolume(id="O1",
                       min_corner=Vec3(1500.0, 1500.0, 0.0),
                       max_corner=Vec3(2000.0, 2000.0, 100.0)),
    ]


def map_two_targets(env):
    """Map 7 — Two Targets: quick sortie with return."""
    env.base = BaseStation(position=Vec3(2500.0, 100.0, 0.0), recharge_radius=80.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(800.0, 3000.0, 30.0),
                       urgency=0.6, delivery_radius=100.0),
        DeliveryTarget(id="T2", position=Vec3(4200.0, 3500.0, 35.0),
                       urgency=0.9, delivery_radius=100.0),
    ]
    env.hazards = [
        HazardRegion(id="H1", center=Vec3(1500.0, 2000.0, 0.0),
                     radius=400.0, severity=0.7, height=60.0, growth_rate=0.004),
        HazardRegion(id="H2", center=Vec3(3500.0, 2800.0, 0.0),
                     radius=350.0, severity=0.8, height=55.0, growth_rate=0.006),
    ]
    env.obstacles = []


def map_final_boss(env):
    """Map 8 — Final Boss: irregular city maze with edge punishment."""
    env.base = BaseStation(position=Vec3(320.0, 320.0, 0.0), recharge_radius=85.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(1650.0, 4350.0, 38.0),
                       urgency=0.88, delivery_radius=100.0),
        DeliveryTarget(id="T2", position=Vec3(4550.0, 3600.0, 55.0),
                       urgency=1.0, delivery_radius=120.0),
        DeliveryTarget(id="T3", position=Vec3(4300.0, 700.0, 45.0),
                       urgency=0.92, delivery_radius=110.0),
    ]
    env.hazards = [
        # Four boundary domes to discourage edge-hugging routes.
        HazardRegion(id="H1", center=Vec3(2500.0, 260.0, 0.0),
                     radius=560.0, severity=0.88, height=150.0, growth_rate=0.0070),
        HazardRegion(id="H2", center=Vec3(4740.0, 2500.0, 0.0),
                     radius=540.0, severity=0.90, height=155.0, growth_rate=0.0065),
        HazardRegion(id="H3", center=Vec3(2500.0, 4740.0, 0.0),
                     radius=560.0, severity=0.89, height=160.0, growth_rate=0.0070),
        HazardRegion(id="H4", center=Vec3(260.0, 2500.0, 0.0),
                     radius=520.0, severity=0.87, height=145.0, growth_rate=0.0065),
        # Internal domes forcing vertical/route decisions through the city core.
        HazardRegion(id="H5", center=Vec3(1300.0, 1350.0, 0.0),
                     radius=420.0, severity=0.78, height=120.0, growth_rate=0.0050),
        HazardRegion(id="H6", center=Vec3(2550.0, 2050.0, 0.0),
                     radius=500.0, severity=0.94, height=175.0, growth_rate=0.0085),
        HazardRegion(id="H7", center=Vec3(3600.0, 3150.0, 0.0),
                     radius=450.0, severity=0.86, height=145.0, growth_rate=0.0060),
        HazardRegion(id="H8", center=Vec3(4300.0, 1150.0, 0.0),
                     radius=390.0, severity=0.91, height=150.0, growth_rate=0.0075),
    ]

    def _obs(i: int, x1: float, y1: float, x2: float, y2: float, z: float) -> ObstacleVolume:
        return ObstacleVolume(
            id=f"O{i}",
            min_corner=Vec3(x1, y1, 0.0),
            max_corner=Vec3(x2, y2, z),
        )

    # Engine supports axis-aligned boxes; combine them to form irregular
    # L/U/T-like footprints and narrow street corridors.
    env.obstacles = [
        # L-block (south-west)
        _obs(1, 900.0, 650.0, 1220.0, 2400.0, 170.0),
        _obs(2, 900.0, 650.0, 1980.0, 980.0, 170.0),
        # U-block (north-central)
        _obs(3, 1900.0, 1800.0, 2220.0, 3600.0, 178.0),
        _obs(4, 2780.0, 1800.0, 3100.0, 3600.0, 178.0),
        _obs(5, 1900.0, 3270.0, 3100.0, 3600.0, 178.0),
        # Hook shape (east-mid)
        _obs(6, 3320.0, 850.0, 3600.0, 2500.0, 182.0),
        _obs(7, 3600.0, 2200.0, 4450.0, 2500.0, 182.0),
        # L-block (north-east)
        _obs(8, 3450.0, 2800.0, 3770.0, 4550.0, 168.0),
        _obs(9, 3450.0, 4220.0, 4700.0, 4550.0, 168.0),
        # Central choke islands
        _obs(10, 2400.0, 900.0, 2920.0, 1420.0, 185.0),
        _obs(11, 2400.0, 2550.0, 2920.0, 3050.0, 185.0),
        # North-west comb
        _obs(12, 650.0, 3000.0, 1030.0, 4700.0, 158.0),
        _obs(13, 1030.0, 4050.0, 1700.0, 4700.0, 158.0),
        # Local blockers near expected shortcuts
        _obs(14, 1200.0, 1200.0, 1700.0, 1650.0, 162.0),
        _obs(15, 3900.0, 1200.0, 4550.0, 1700.0, 162.0),
    ]


def map_altitude_gauntlet(env):
    """Map 9 — Altitude Gauntlet: stacked obstacles force under/over routing."""
    env.cfg.max_episode_steps = max(env.cfg.max_episode_steps, 3000)
    env.base = BaseStation(position=Vec3(320.0, 260.0, 0.0), recharge_radius=90.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(1400.0, 4380.0, 55.0),
                       urgency=0.82, delivery_radius=105.0),
        # Keep T2 difficult but reachable; original placement intersected a suspended canopy volume.
        DeliveryTarget(id="T2", position=Vec3(3050.0, 2100.0, 120.0),
                       urgency=1.0, delivery_radius=120.0),
        DeliveryTarget(id="T3", position=Vec3(4560.0, 3980.0, 145.0),
                       urgency=0.96, delivery_radius=115.0),
    ]
    env.hazards = [
        # Boundary fire belt makes edge-running expensive.
        HazardRegion(id="H1", center=Vec3(2500.0, 250.0, 0.0),
                     radius=610.0, severity=0.90, height=170.0, growth_rate=0.0070),
        HazardRegion(id="H2", center=Vec3(4750.0, 2500.0, 0.0),
                     radius=560.0, severity=0.90, height=170.0, growth_rate=0.0065),
        HazardRegion(id="H3", center=Vec3(2500.0, 4750.0, 0.0),
                     radius=590.0, severity=0.91, height=175.0, growth_rate=0.0070),
        HazardRegion(id="H4", center=Vec3(250.0, 2500.0, 0.0),
                     radius=540.0, severity=0.88, height=165.0, growth_rate=0.0065),
        # Interior domes create vertical decision points.
        HazardRegion(id="H5", center=Vec3(1300.0, 1500.0, 0.0),
                     radius=430.0, severity=0.78, height=120.0, growth_rate=0.0050),
        HazardRegion(id="H6", center=Vec3(2500.0, 2000.0, 0.0),
                     radius=520.0, severity=0.95, height=185.0, growth_rate=0.0085),
        HazardRegion(id="H7", center=Vec3(3350.0, 2750.0, 0.0),
                     radius=460.0, severity=0.88, height=160.0, growth_rate=0.0060),
        HazardRegion(id="H8", center=Vec3(4200.0, 1100.0, 0.0),
                     radius=360.0, severity=0.90, height=150.0, growth_rate=0.0070),
        HazardRegion(id="H9", center=Vec3(1700.0, 3600.0, 0.0),
                     radius=420.0, severity=0.82, height=140.0, growth_rate=0.0055),
        HazardRegion(id="H10", center=Vec3(3650.0, 4050.0, 0.0),
                     radius=420.0, severity=0.87, height=155.0, growth_rate=0.0060),
    ]

    def _obs(i: int, x1: float, y1: float, z1: float,
             x2: float, y2: float, z2: float) -> ObstacleVolume:
        return ObstacleVolume(
            id=f"O{i}",
            min_corner=Vec3(x1, y1, z1),
            max_corner=Vec3(x2, y2, z2),
        )

    env.obstacles = [
        # Grounded megastructures.
        _obs(1, 700.0, 700.0, 0.0, 1050.0, 1900.0, 175.0),
        _obs(2, 1050.0, 700.0, 0.0, 1700.0, 980.0, 175.0),
        _obs(3, 1850.0, 1300.0, 0.0, 2350.0, 2500.0, 188.0),
        _obs(4, 2900.0, 1600.0, 0.0, 3400.0, 2850.0, 186.0),
        _obs(5, 3650.0, 2500.0, 0.0, 4300.0, 3100.0, 170.0),
        _obs(6, 2300.0, 3600.0, 0.0, 3200.0, 3950.0, 165.0),
        _obs(7, 600.0, 2850.0, 0.0, 950.0, 4550.0, 160.0),
        _obs(8, 950.0, 4150.0, 0.0, 1550.0, 4550.0, 160.0),
        _obs(9, 3850.0, 600.0, 0.0, 4550.0, 1350.0, 168.0),
        _obs(10, 350.0, 2100.0, 0.0, 900.0, 2650.0, 158.0),
        # Suspended skybridges / canopies (must go below or detour).
        _obs(11, 1100.0, 1050.0, 70.0, 2600.0, 1350.0, 195.0),
        _obs(12, 2100.0, 2250.0, 82.0, 3700.0, 2550.0, 198.0),
        _obs(13, 3000.0, 3320.0, 88.0, 4550.0, 3600.0, 196.0),
        _obs(14, 900.0, 3000.0, 95.0, 2000.0, 3320.0, 190.0),
        _obs(15, 2450.0, 2920.0, 110.0, 3150.0, 3500.0, 198.0),
        _obs(16, 3400.0, 1450.0, 92.0, 4300.0, 1800.0, 192.0),
        _obs(17, 1450.0, 550.0, 78.0, 2100.0, 820.0, 188.0),
        # Hanging walls (low-level tunnels underneath).
        _obs(18, 1750.0, 1650.0, 65.0, 2050.0, 3350.0, 200.0),
        _obs(19, 3200.0, 950.0, 72.0, 3500.0, 2400.0, 200.0),
        _obs(20, 2550.0, 3550.0, 80.0, 2850.0, 4700.0, 200.0),
    ]


TEST_MAPS = {
    "reversed":        ("Reversed (base top-right)", map_reversed),
    "fire_corridor":   ("Fire Corridor (narrow gap)", map_fire_corridor),
    "dense_obstacles": ("Dense Urban (4 obstacles)", map_dense_obstacles),
    "long_range":      ("Long Range (corners)", map_long_range),
    "clustered_fires": ("Clustered Fires (overlapping)", map_clustered_fires),
    "single_target":   ("Single Target (1 drop)", map_single_target),
    "two_targets":     ("Two Targets (2 drops)", map_two_targets),
    "final_boss":      ("Final Boss (15 obstacles, 8 fires)", map_final_boss),
    "altitude_gauntlet": ("Altitude Gauntlet (stacked airspace)", map_altitude_gauntlet),
}


# ---------------------------------------------------------------------------
# Stress-map planner baseline (explicit controls)
# ---------------------------------------------------------------------------

class HighAltitudePlanner:
    """Simple grid planner that flies high and routes around blocking footprints."""

    def __init__(
        self,
        env: VarahaSB3Env,
        cruise_z: Optional[float] = None,
        grid_res: float = 60.0,
        clearance: float = 80.0,
    ) -> None:
        self.envw = env
        self.cruise_z = cruise_z if cruise_z is not None else self._auto_cruise_z()
        self.grid_res = grid_res
        self.clearance = max(clearance, 180.0) if self.cruise_z < 100.0 else clearance
        self.goal_id = ""
        self.path_xy: list[tuple[float, float]] = []
        self._steps_since_plan = 0

    def _auto_cruise_z(self) -> float:
        env = self.envw.env
        return min(env.cfg.world_z - 1.0, 198.0)

    def _select_goal(self) -> tuple[str, float, float]:
        env = self.envw.env
        pos = env.drone.position
        pending = [t for t in env.targets if not t.delivered]
        if pending:
            tgt = min(
                pending,
                key=lambda t: (pos.x - t.position.x) ** 2 + (pos.y - t.position.y) ** 2,
            )
            return (f"target:{tgt.id}", tgt.position.x, tgt.position.y)
        return ("base", env.base.position.x, env.base.position.y)

    def _build_blocked_grid(self) -> tuple[np.ndarray, float]:
        env = self.envw.env
        cfg = env.cfg
        res = self.grid_res
        nx = int(cfg.world_x / res) + 1
        ny = int(cfg.world_y / res) + 1
        blocked = np.zeros((nx, ny), dtype=np.bool_)

        def clamp_cell(x: float, y: float) -> tuple[int, int]:
            ix = int(max(0, min(nx - 1, x / res)))
            iy = int(max(0, min(ny - 1, y / res)))
            return ix, iy

        for obs in env.obstacles:
            if obs.min_corner.z - 1.0 <= self.cruise_z <= obs.max_corner.z + 1.0:
                x1 = obs.min_corner.x - self.clearance
                x2 = obs.max_corner.x + self.clearance
                y1 = obs.min_corner.y - self.clearance
                y2 = obs.max_corner.y + self.clearance
                ix1, iy1 = clamp_cell(x1, y1)
                ix2, iy2 = clamp_cell(x2, y2)
                blocked[ix1:ix2 + 1, iy1:iy2 + 1] = True

        for cyl in env.cylinders:
            if self.cruise_z <= cyl.height + 1.0:
                r = cyl.radius + self.clearance
                ix1, iy1 = clamp_cell(cyl.center.x - r, cyl.center.y - r)
                ix2, iy2 = clamp_cell(cyl.center.x + r, cyl.center.y + r)
                for ix in range(ix1, ix2 + 1):
                    for iy in range(iy1, iy2 + 1):
                        wx = ix * res
                        wy = iy * res
                        if (wx - cyl.center.x) ** 2 + (wy - cyl.center.y) ** 2 <= r * r:
                            blocked[ix, iy] = True

        return blocked, res

    def _a_star(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        blocked: np.ndarray,
        res: float,
    ) -> list[tuple[float, float]]:
        nx, ny = blocked.shape

        def to_cell(x: float, y: float) -> tuple[int, int]:
            return (
                int(max(0, min(nx - 1, x / res))),
                int(max(0, min(ny - 1, y / res))),
            )

        start = to_cell(*start_xy)
        goal = to_cell(*goal_xy)
        blocked[start] = False
        blocked[goal] = False

        pq: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(pq, (0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        gscore = {start: 0.0}

        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while pq:
            _, node = heapq.heappop(pq)
            if node == goal:
                break
            for dx, dy in neighbors:
                nxn = node[0] + dx
                nyn = node[1] + dy
                if nxn < 0 or nxn >= nx or nyn < 0 or nyn >= ny:
                    continue
                if blocked[nxn, nyn]:
                    continue
                if dx != 0 and dy != 0:
                    # Avoid cutting through obstacle corners on diagonals.
                    if blocked[node[0] + dx, node[1]] or blocked[node[0], node[1] + dy]:
                        continue
                step_cost = 1.4142 if dx != 0 and dy != 0 else 1.0
                cand = gscore[node] + step_cost
                nxt = (nxn, nyn)
                if cand < gscore.get(nxt, float("inf")):
                    gscore[nxt] = cand
                    came_from[nxt] = node
                    hx = goal[0] - nxn
                    hy = goal[1] - nyn
                    h = (hx * hx + hy * hy) ** 0.5
                    heapq.heappush(pq, (cand + h, nxt))

        if goal not in came_from and goal != start:
            return [goal_xy]

        node = goal
        cells = [node]
        while node != start:
            node = came_from[node]
            cells.append(node)
        cells.reverse()
        if len(cells) > 1:
            cells = cells[1:]
        return [(ix * res, iy * res) for ix, iy in cells]

    def _can_deliver_now(self) -> bool:
        pos = self.envw.env.drone.position
        for tgt in self.envw.env.targets:
            if tgt.delivered:
                continue
            dx = pos.x - tgt.position.x
            dy = pos.y - tgt.position.y
            horiz_dist = (dx * dx + dy * dy) ** 0.5
            alt_above = pos.z - tgt.position.z
            if horiz_dist <= tgt.delivery_radius and -10.0 <= alt_above <= tgt.delivery_radius * 2:
                return True
        return False

    def _can_recharge_now(self) -> bool:
        env = self.envw.env
        pos = env.drone.position
        base = env.base.position
        hdist = ((pos.x - base.x) ** 2 + (pos.y - base.y) ** 2) ** 0.5
        return hdist <= env.base.recharge_radius

    def act(self) -> np.ndarray:
        env = self.envw.env
        cfg = env.cfg
        goal_id, gx, gy = self._select_goal()
        pos = env.drone.position
        target_obj = None
        if goal_id.startswith("target:"):
            tid = goal_id.split(":", 1)[1]
            target_obj = next((t for t in env.targets if t.id == tid), None)

        self._steps_since_plan += 1
        if goal_id != self.goal_id or not self.path_xy or self._steps_since_plan >= 8:
            blocked, res = self._build_blocked_grid()
            self.path_xy = self._a_star((pos.x, pos.y), (gx, gy), blocked, res)
            self.goal_id = goal_id
            self._steps_since_plan = 0

        while self.path_xy:
            wx, wy = self.path_xy[0]
            if ((pos.x - wx) ** 2 + (pos.y - wy) ** 2) ** 0.5 < self.grid_res * 0.35:
                self.path_xy.pop(0)
            else:
                break

        tx, ty = (gx, gy) if not self.path_xy else self.path_xy[0]
        vel = env.drone.velocity

        dx = tx - pos.x
        dy = ty - pos.y
        dxy = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        tz = min(cfg.world_z - 1.0, self.cruise_z)
        if target_obj is not None and dxy < 350.0:
            tz = min(cfg.world_z - 1.0, max(self.cruise_z, target_obj.position.z + 25.0))

        if self.cruise_z < 100.0:
            desired_speed = min(10.0, max(3.0, dxy * 0.35))
        else:
            desired_speed = min(20.0, max(5.0, dxy * 0.5))
        desired_vx = desired_speed * dx / dxy
        desired_vy = desired_speed * dy / dxy
        desired_vz = max(-cfg.max_speed, min(cfg.max_speed, (tz - pos.z) * 1.0))

        ax = (desired_vx - vel.x) / cfg.dt
        ay = (desired_vy - vel.y) / cfg.dt
        az = (desired_vz - vel.z) / cfg.dt

        ax_n = float(max(-1.0, min(1.0, ax / cfg.max_acceleration)))
        ay_n = float(max(-1.0, min(1.0, ay / cfg.max_acceleration)))
        az_n = float(max(-1.0, min(1.0, az / cfg.max_acceleration)))

        # One-step safety shield against control overshoot into an obstacle volume.
        accel_vec = Vec3(
            ax_n * cfg.max_acceleration,
            ay_n * cfg.max_acceleration,
            az_n * cfg.max_acceleration,
        ).clamp_magnitude(cfg.max_acceleration)
        pred_pos = Vec3(pos.x, pos.y, pos.z)
        pred_vel = Vec3(vel.x, vel.y, vel.z)
        collision_lookahead = False
        for _ in range(4):
            pred_vel = (pred_vel + accel_vec.scale(cfg.dt)).clamp_magnitude(cfg.max_speed)
            pred_pos = pred_pos + pred_vel.scale(cfg.dt)
            if any(o.contains(pred_pos) for o in env.obstacles) or any(c.contains(pred_pos) for c in env.cylinders):
                collision_lookahead = True
                break
        if collision_lookahead:
            ax_n = float(max(-1.0, min(1.0, -vel.x / (cfg.max_acceleration + 1e-6))))
            ay_n = float(max(-1.0, min(1.0, -vel.y / (cfg.max_acceleration + 1e-6))))
            if self.cruise_z < 100.0:
                az_n = -1.0
            else:
                az_n = -1.0 if pos.z > 140.0 else 1.0

        deliver = self._can_deliver_now()
        recharge = self._can_recharge_now() and (
            all(t.delivered for t in env.targets) or env.drone.battery < cfg.battery_capacity * 0.8
        )
        return np.array(
            [ax_n, ay_n, az_n, 1.0 if deliver else -1.0, 1.0 if recharge else -1.0],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_map(
    model: Optional[PPO],
    map_name: str,
    world_fn,
    n_episodes: int = 10,
    save_dir: str = "./results/generalization",
    controller: str = "ppo",
    seed_offset: int = 10_000,
    explicit_controls: bool = True,
):
    rewards, deliveries, successes, lengths = [], [], [], []
    best_trace = None
    best_reward = -float("inf")

    for ep in range(n_episodes):
        env = VarahaSB3Env(world_fn=world_fn, explicit_controls=explicit_controls)
        obs, _ = env.reset(seed=seed_offset + ep * 131)
        total_r = 0.0
        done = False
        planner = HighAltitudePlanner(env) if controller == "planner" else None
        while not done:
            if controller == "planner":
                action = planner.act()
            else:
                if model is None:
                    raise ValueError("controller='ppo' requires a loaded model")
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except ValueError as exc:
                    raise RuntimeError(
                        "PPO model/environment shape mismatch. "
                        "Use a checkpoint trained with the current wrapper "
                        "(obs_dim=124, explicit_controls action_dim=5), "
                        "or run with --controller planner."
                    ) from exc
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated

        trace = env.get_trace()
        rewards.append(total_r)
        deliveries.append(len(trace["summary"]["delivered"]))
        successes.append(trace["summary"]["success"])
        lengths.append(trace["summary"]["total_steps"])

        if total_r > best_reward:
            best_reward = total_r
            best_trace = trace

    # Save best trace
    os.makedirs(save_dir, exist_ok=True)
    trace_path = os.path.join(save_dir, f"trace_{map_name}_{controller}.json")
    with open(trace_path, "w") as f:
        json.dump(best_trace, f, indent=2)

    return {
        "map": map_name,
        "mean_reward": round(float(np.mean(rewards)), 1),
        "std_reward": round(float(np.std(rewards)), 1),
        "mean_deliveries": round(float(np.mean(deliveries)), 2),
        "max_deliveries": int(max(deliveries)),
        "success_rate": round(float(np.mean(successes)), 2),
        "mean_length": round(float(np.mean(lengths)), 0),
        "best_reward": round(best_reward, 1),
        "best_delivered": best_trace["summary"]["delivered"],
        "trace_file": trace_path,
        "controller": controller,
        "seed_offset": seed_offset,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO or planner on Varaha generalization maps.")
    p.add_argument("--model-path", default="./results/ppo_varaha.zip", help="Path to PPO checkpoint")
    p.add_argument("--controller", choices=["ppo", "planner"], default="ppo")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-offset", type=int, default=10_000, help="Unseen seed offset")
    p.add_argument("--save-dir", default="./results/generalization")
    p.add_argument("--maps", default="", help="Comma list of map keys; empty = all default maps")
    p.add_argument(
        "--stress-only",
        action="store_true",
        help="Evaluate only final_boss + altitude_gauntlet (recommended for stress testing).",
    )
    p.add_argument(
        "--implicit-controls",
        action="store_true",
        help="Use legacy 3D action space with auto deliver/recharge.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    explicit_controls = not args.implicit_controls

    model: Optional[PPO] = None
    if args.controller == "ppo":
        print(f"Loading PPO model from {args.model_path}...")
        model = PPO.load(args.model_path)

    print("\n" + "=" * 72)
    print(f"  GENERALIZATION EVALUATION ({args.controller.upper()})")
    print("=" * 72)
    print(f"  episodes={args.episodes}  seed_offset={args.seed_offset}  explicit_controls={explicit_controls}")

    selected_maps: list[tuple[str, Any]]
    if args.stress_only:
        selected_maps = [
            ("final_boss", TEST_MAPS["final_boss"][1]),
            ("altitude_gauntlet", TEST_MAPS["altitude_gauntlet"][1]),
        ]
    elif args.maps.strip():
        keys = [k.strip() for k in args.maps.split(",") if k.strip()]
        selected_maps = [(k, TEST_MAPS[k][1]) for k in keys]
    else:
        selected_maps = list(TEST_MAPS.items())

    all_results = []

    # Include training map baseline unless explicitly stress-only.
    if not args.stress_only:
        print("\n  [baseline] training...")
        baseline = evaluate_on_map(
            model=model,
            map_name="training",
            world_fn=None,
            n_episodes=args.episodes,
            save_dir=args.save_dir,
            controller=args.controller,
            seed_offset=args.seed_offset,
            explicit_controls=explicit_controls,
        )
        all_results.append(baseline)
        print(
            f"    reward={baseline['mean_reward']:.1f} ± {baseline['std_reward']:.1f}  "
            f"deliveries={baseline['mean_deliveries']:.2f}  success={baseline['success_rate']:.0%}"
        )

    for item in selected_maps:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple):
            key = item[0]
            world_fn = item[1][1]
            desc = item[1][0]
        else:
            key, world_fn = item
            desc = TEST_MAPS[key][0]
        print(f"\n  [{key}] {desc}...")
        result = evaluate_on_map(
            model=model,
            map_name=key,
            world_fn=world_fn,
            n_episodes=args.episodes,
            save_dir=args.save_dir,
            controller=args.controller,
            seed_offset=args.seed_offset,
            explicit_controls=explicit_controls,
        )
        all_results.append(result)
        print(
            f"    reward={result['mean_reward']:.1f} ± {result['std_reward']:.1f}  "
            f"deliveries={result['mean_deliveries']:.2f}  success={result['success_rate']:.0%}  "
            f"best={result['best_delivered']}"
        )

    print("\n" + "=" * 72)
    print(f"  {'Map':<22} {'Reward':>10} {'Deliveries':>12} {'Success':>9}")
    print("-" * 72)
    for r in all_results:
        print(
            f"  {r['map']:<22} {r['mean_reward']:>9.1f} "
            f"{r['mean_deliveries']:>11.2f} {r['success_rate']:>8.0%}"
        )
    print("=" * 72)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = "stress_results" if args.stress_only else "results"
    save_path = os.path.join(args.save_dir, f"{tag}_{args.controller}.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {save_path}")
    print(f"  Traces saved → {args.save_dir}/trace_*.json")


if __name__ == "__main__":
    main()
