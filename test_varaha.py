#!/usr/bin/env python3
"""Varaha environment test harness.

Three tests:
  1. Random rollout   — smoke-test that the env runs without crashing
  2. Heuristic rollout — simple greedy baseline to prove navigability
  3. Render preview    — dump a sample render_state JSON for frontend work
"""

import json
import random
from typing import Any

from sim_types import Vec3
from varaha_env import VarahaEnv


# -----------------------------------------------------------------------
# 1. Random rollout
# -----------------------------------------------------------------------

def random_rollout(env: VarahaEnv, max_steps: int = 200) -> None:
    """Sample random actions and print concise per-step logs."""
    obs = env.reset(seed=42)
    print(f"  Initial pos: {obs['drone_position']}  battery: {obs['battery']}")

    total_reward = 0.0
    for i in range(max_steps):
        action = {
            "ax": random.uniform(-5, 5),
            "ay": random.uniform(-5, 5),
            "az": random.uniform(-2, 2),
            "deliver": random.random() < 0.1,
            "recharge": random.random() < 0.05,
        }
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if i % 20 == 0 or done:
            pos = obs["drone_position"]
            print(
                f"  step {i:>4d} | "
                f"pos ({pos['x']:7.1f},{pos['y']:7.1f},{pos['z']:6.1f}) | "
                f"bat {obs['battery']:6.1f} | "
                f"r {reward:+8.2f} | "
                f"{'DONE' if done else ''}"
            )

        if done:
            break

    delivered = [t["id"] for t in obs["targets"] if t["delivered"]]
    print(f"  --- Episode finished after {env.step_count} steps ---")
    print(f"  Total reward : {total_reward:+.2f}")
    print(f"  Delivered    : {delivered}")
    print(f"  Alive        : {obs['alive']}")
    print(f"  Battery left : {obs['battery']:.2f}")

    with open("trace_random.json", "w") as f:
        json.dump(env.get_trace(), f, indent=2)
    print("  Wrote trace_random.json")


# -----------------------------------------------------------------------
# 2. Heuristic rollout
# -----------------------------------------------------------------------

def _heuristic_action(obs: dict[str, Any], env: VarahaEnv) -> dict[str, Any]:
    """Simple PD-control heuristic: fly to nearest target, then return home."""
    drone_pos = Vec3(**obs["drone_position"])
    drone_vel = Vec3(**obs["drone_velocity"])

    undelivered = [t for t in obs["targets"] if not t["delivered"]]

    if undelivered:
        nearest = min(undelivered, key=lambda t: Vec3(**t["relative_position"]).norm())
        rel = Vec3(**nearest["relative_position"])
        in_range = rel.norm() <= 15.0
        goal_dir = rel
    else:
        goal_dir = env.base.position - drone_pos
        in_range = False

    # PD control: proportional toward goal, derivative damping
    kp, kd = 0.3, 0.8
    accel = goal_dir.scale(kp) - drone_vel.scale(kd)
    accel = accel.clamp_magnitude(env.cfg.max_acceleration)

    near_base = drone_pos.distance_to(env.base.position) <= env.base.recharge_radius

    return {
        "ax": accel.x,
        "ay": accel.y,
        "az": accel.z,
        "deliver": in_range,
        "recharge": near_base and obs["battery"] < 40,
    }


def heuristic_rollout(env: VarahaEnv, max_steps: int = 500) -> None:
    """Run the greedy heuristic and report results."""
    obs = env.reset(seed=7)
    print(f"  Initial pos: {obs['drone_position']}  battery: {obs['battery']}")

    total_reward = 0.0
    for i in range(max_steps):
        action = _heuristic_action(obs, env)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        flag = ""
        if info.get("delivered_target_ids"):
            flag = f" DELIVERED {info['delivered_target_ids']}"
        if info.get("in_hazard"):
            flag += f" HAZARD(sev={info['hazard_severity']:.2f})"
        if info.get("collision"):
            flag += " COLLISION"

        if i % 25 == 0 or done or flag:
            pos = obs["drone_position"]
            print(
                f"  step {i:>4d} | "
                f"pos ({pos['x']:7.1f},{pos['y']:7.1f},{pos['z']:6.1f}) | "
                f"bat {obs['battery']:6.1f} | "
                f"r {reward:+8.2f}"
                f"{flag}"
            )

        if done:
            break

    delivered = [t["id"] for t in obs["targets"] if t["delivered"]]
    print(f"  --- Episode finished after {env.step_count} steps ---")
    print(f"  Total reward : {total_reward:+.2f}")
    print(f"  Delivered    : {delivered}")
    print(f"  Alive        : {obs['alive']}")
    print(f"  Battery left : {obs['battery']:.2f}")
    print(f"  Success      : {env._is_success()}")

    with open("trace_heuristic.json", "w") as f:
        json.dump(env.get_trace(), f, indent=2)
    print("  Wrote trace_heuristic.json")


# -----------------------------------------------------------------------
# 3. Render preview
# -----------------------------------------------------------------------

def render_preview(env: VarahaEnv) -> None:
    """Print a human-readable summary and dump JSON for frontend work."""
    env.reset(seed=0)
    # run a few steps so the state is non-trivial
    for _ in range(10):
        env.step({"ax": 2.0, "ay": 1.0, "az": 0.5, "deliver": False, "recharge": False})

    state = env.render_state()

    print("  Base station :", state["base_station"]["position"])
    print("  Drone pos    :", state["drone"]["position"])
    print(f"  Drone battery: {state['drone']['battery']:.2f}")
    print(f"  Step         : {state['step']} / {state['max_steps']}")
    print(f"  Cum. reward  : {state['cumulative_reward']:.2f}")
    print("  Targets:")
    for t in state["targets"]:
        print(f"    {t['id']}  pos={t['position']}  urg={t['urgency']}  delivered={t['delivered']}")
    print("  Hazards:")
    for h in state["hazards"]:
        print(f"    {h['id']}  center={h['center']}  r={h['radius']}  sev={h['severity']}")
    print("  Obstacles:")
    for o in state["obstacles"]:
        print(f"    {o['id']}  {o['min_corner']} → {o['max_corner']}")

    # lat/lon conversion demo
    drone_latlon = env.local_to_latlon(env.drone.position)
    print(f"  Drone lat/lon: {drone_latlon}")

    # dump JSON
    out_path = "sample_render_state.json"
    with open(out_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n  Wrote {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    env = VarahaEnv()

    print("=" * 64)
    print("  TEST 1 : Random Rollout")
    print("=" * 64)
    random_rollout(env)

    print()
    print("=" * 64)
    print("  TEST 2 : Heuristic Rollout")
    print("=" * 64)
    heuristic_rollout(env)

    print()
    print("=" * 64)
    print("  TEST 3 : Render Preview")
    print("=" * 64)
    render_preview(env)

    print()
    print("All tests completed.")
