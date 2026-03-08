#!/usr/bin/env python3
"""Evaluate trained PPO agent on held-out test environments."""

import json
import os

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
    env.base = BaseStation(position=Vec3(320.0, 260.0, 0.0), recharge_radius=90.0)
    env.targets = [
        DeliveryTarget(id="T1", position=Vec3(1400.0, 4380.0, 55.0),
                       urgency=0.82, delivery_radius=105.0),
        DeliveryTarget(id="T2", position=Vec3(3050.0, 2480.0, 120.0),
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_map(model, map_name: str, world_fn, n_episodes: int = 10,
                    save_dir: str = "./results/generalization"):
    rewards, deliveries, successes, lengths = [], [], [], []
    best_trace = None
    best_reward = -float("inf")

    for ep in range(n_episodes):
        env = VarahaSB3Env(world_fn=world_fn)
        obs, _ = env.reset(seed=ep * 31)
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
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
    trace_path = os.path.join(save_dir, f"trace_{map_name}.json")
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
    }


def main():
    model_path = "./results/ppo_varaha.zip"
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Also eval on training map for baseline comparison
    print("\n" + "=" * 70)
    print("  GENERALIZATION EVALUATION")
    print("=" * 70)

    # Training map baseline
    print("\n  [baseline] Training Map...")
    baseline = evaluate_on_map(model, "training", world_fn=None, n_episodes=10)
    all_results = [baseline]
    print(f"    reward={baseline['mean_reward']:.1f} ± {baseline['std_reward']:.1f}  "
          f"deliveries={baseline['mean_deliveries']:.2f}  "
          f"best={baseline['best_delivered']}")

    # Test maps
    for key, (desc, world_fn) in TEST_MAPS.items():
        print(f"\n  [{key}] {desc}...")
        result = evaluate_on_map(model, key, world_fn, n_episodes=10)
        all_results.append(result)
        print(f"    reward={result['mean_reward']:.1f} ± {result['std_reward']:.1f}  "
              f"deliveries={result['mean_deliveries']:.2f}  "
              f"best={result['best_delivered']}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'Map':<22} {'Reward':>10} {'Deliveries':>12} {'Max':>5} {'Success':>9}")
    print("-" * 70)
    for r in all_results:
        tag = "★" if r["map"] == "training" else " "
        print(f"  {tag} {r['map']:<20} {r['mean_reward']:>9.1f} "
              f"{r['mean_deliveries']:>11.2f} {r['max_deliveries']:>5d} "
              f"{r['success_rate']:>8.0%}")
    print("=" * 70)

    avg_test_del = np.mean([r["mean_deliveries"] for r in all_results if r["map"] != "training"])
    print(f"\n  Training map deliveries:  {baseline['mean_deliveries']:.2f}")
    print(f"  Avg test map deliveries:  {avg_test_del:.2f}")
    print(f"  Generalization ratio:     {avg_test_del / max(baseline['mean_deliveries'], 0.01):.1%}")

    # Save all results
    save_path = "./results/generalization/results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {save_path}")
    print(f"  Traces saved → ./results/generalization/trace_*.json")


if __name__ == "__main__":
    main()
