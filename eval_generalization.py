#!/usr/bin/env python3
"""Evaluate trained PPO agent on held-out test environments.

Creates 5 novel maps the agent has never seen and measures generalization.
"""

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


TEST_MAPS = {
    "reversed":        ("Reversed (base top-right)", map_reversed),
    "fire_corridor":   ("Fire Corridor (narrow gap)", map_fire_corridor),
    "dense_obstacles": ("Dense Urban (4 obstacles)", map_dense_obstacles),
    "long_range":      ("Long Range (corners)", map_long_range),
    "clustered_fires": ("Clustered Fires (overlapping)", map_clustered_fires),
    "single_target":   ("Single Target (1 drop)", map_single_target),
    "two_targets":     ("Two Targets (2 drops)", map_two_targets),
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
