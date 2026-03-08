#!/usr/bin/env python3
"""Generate side-by-side path comparison: untrained vs trained PPO.

Usage:
    python compare_paths.py                              # uses ./results/ppo_varaha.zip
    python compare_paths.py --model results/ppo_varaha   # explicit model path
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from stable_baselines3 import PPO

from sb3_env_wrapper import VarahaSB3Env


# -----------------------------------------------------------------------
# World drawing helpers
# -----------------------------------------------------------------------

def draw_world(ax, world, title=""):
    """Render base, targets, hazards, obstacles on a matplotlib axis."""
    bx = world["bounds"]["x"]
    by = world["bounds"]["y"]

    # Hazards
    for hz in world["hazards"]:
        circle = plt.Circle(
            (hz["center"]["x"], hz["center"]["y"]),
            hz["radius"],
            facecolor=(1, 0.3, 0.2, 0.12),
            edgecolor=(1, 0.3, 0.2, 0.5),
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(circle)
        ax.text(
            hz["center"]["x"], hz["center"]["y"],
            f'{hz["id"]}\nsev={hz["severity"]}',
            ha="center", va="center", fontsize=7, color=(1, 0.3, 0.2, 0.7),
        )

    # Obstacles
    for ob in world["obstacles"]:
        w = ob["max_corner"]["x"] - ob["min_corner"]["x"]
        h = ob["max_corner"]["y"] - ob["min_corner"]["y"]
        rect = mpatches.FancyBboxPatch(
            (ob["min_corner"]["x"], ob["min_corner"]["y"]),
            w, h,
            boxstyle="round,pad=0",
            facecolor=(0.5, 0.5, 0.55, 0.25),
            edgecolor=(0.5, 0.5, 0.55, 0.6),
            linewidth=1.5,
        )
        ax.add_patch(rect)
        cx = ob["min_corner"]["x"] + w / 2
        cy = ob["min_corner"]["y"] + h / 2
        ax.text(cx, cy, ob["id"], ha="center", va="center", fontsize=8, color="0.5")

    # Base
    bs = world["base_station"]
    ax.plot(bs["position"]["x"], bs["position"]["y"], "s",
            color="dodgerblue", markersize=9, zorder=5)
    ax.annotate("BASE", (bs["position"]["x"], bs["position"]["y"]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=8, fontweight="bold", color="dodgerblue")

    # Targets
    for tgt in world["targets"]:
        delivered = tgt.get("delivered", False)
        color = "limegreen" if delivered else "orange"
        ax.plot(tgt["position"]["x"], tgt["position"]["y"], "^",
                color=color, markersize=9, zorder=5)
        ax.annotate(
            f'{tgt["id"]} (u={tgt["urgency"]})',
            (tgt["position"]["x"], tgt["position"]["y"]),
            textcoords="offset points", xytext=(12, -10),
            fontsize=7, color=color,
        )

    ax.set_xlim(-150, bx + 150)
    ax.set_ylim(-150, by + 150)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def draw_path(ax, trace, summary):
    """Overlay the drone flight path on an existing world axis."""
    xs = [p["position"]["x"] for p in trace]
    ys = [p["position"]["y"] for p in trace]

    n = len(xs)
    for i in range(1, n):
        t = i / n
        events = trace[i].get("events", [])
        if any(e == "collision" for e in events):
            color = "red"
        elif any(e.startswith("hazard") for e in events):
            color = "orange"
        elif any(e.startswith("delivered") for e in events):
            color = "limegreen"
        else:
            color = plt.cm.cool(t)
        ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                color=color, linewidth=1.2, alpha=0.4 + 0.6 * t)

    # Event markers
    for p in trace:
        px, py = p["position"]["x"], p["position"]["y"]
        for ev in p.get("events", []):
            if ev.startswith("delivered"):
                ax.plot(px, py, "o", color="limegreen", markersize=7, zorder=6)
            elif ev == "collision":
                ax.plot(px, py, "x", color="red", markersize=10,
                        markeredgewidth=2.5, zorder=6)
            elif ev == "success":
                ax.plot(px, py, "o", color="dodgerblue", markersize=10,
                        markeredgewidth=2, fillstyle="none", zorder=6)

    # Start marker
    ax.plot(xs[0], ys[0], "D", color="white", markersize=6, markeredgecolor="black",
            markeredgewidth=1, zorder=7)

    # Summary text
    d = summary
    txt = (
        f"steps={d['total_steps']}  reward={d['cumulative_reward']:.0f}\n"
        f"delivered={d['delivered']}  "
        f"{'SUCCESS' if d['success'] else 'alive' if d['alive'] else 'DEAD'}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            color="white")


# -----------------------------------------------------------------------
# Rollout runners
# -----------------------------------------------------------------------

def run_untrained_rollout(seed=42):
    env = VarahaSB3Env()
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_trace()


def run_trained_rollout(model, seed=42):
    env = VarahaSB3Env()
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_trace()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def compare(model_path: str = "./results/ppo_varaha", save_dir: str = "./results"):
    os.makedirs(save_dir, exist_ok=True)

    print("Running untrained rollout...")
    untrained_data = run_untrained_rollout(seed=42)
    print(f"  steps={untrained_data['summary']['total_steps']}  "
          f"reward={untrained_data['summary']['cumulative_reward']:.1f}  "
          f"delivered={untrained_data['summary']['delivered']}")

    print(f"Loading trained model from {model_path}...")
    model = PPO.load(model_path)

    print("Running trained rollout...")
    trained_data = run_trained_rollout(model, seed=42)
    print(f"  steps={trained_data['summary']['total_steps']}  "
          f"reward={trained_data['summary']['cumulative_reward']:.1f}  "
          f"delivered={trained_data['summary']['delivered']}")

    # --- Individual plots ---
    for name, data in [("path_untrained", untrained_data), ("path_trained", trained_data)]:
        fig, ax = plt.subplots(figsize=(8, 8))
        label = "Untrained (random)" if "untrained" in name else "Trained PPO"
        draw_world(ax, data["world"], title=label)
        draw_path(ax, data["trace"], data["summary"])
        fig.tight_layout()
        p = os.path.join(save_dir, f"{name}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved {p}")

    # --- Side-by-side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    draw_world(ax1, untrained_data["world"], title="Untrained (random actions)")
    draw_path(ax1, untrained_data["trace"], untrained_data["summary"])
    draw_world(ax2, trained_data["world"], title="Trained PPO")
    draw_path(ax2, trained_data["trace"], trained_data["summary"])
    fig.suptitle("Varaha — Path Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(save_dir, "path_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved {p}")

    # Save trace JSONs for HTML visualiser
    for name, data in [("trace_untrained_cmp", untrained_data), ("trace_trained_cmp", trained_data)]:
        p = os.path.join(save_dir, f"{name}.json")
        with open(p, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare untrained vs trained paths")
    parser.add_argument("--model", type=str, default="./results/ppo_varaha")
    parser.add_argument("--save-dir", type=str, default="./results")
    args = parser.parse_args()
    compare(model_path=args.model, save_dir=args.save_dir)
