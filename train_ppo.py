#!/usr/bin/env python3
"""Train a PPO policy on VarahaEnv using Stable-Baselines3.

Usage:
    python train_ppo.py                         # 500K steps, default
    python train_ppo.py --timesteps 1000000     # 1M steps
    python train_ppo.py --timesteps 200000 --n-envs 8
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_env_wrapper import VarahaSB3Env


# -----------------------------------------------------------------------
# Callback — records episode-level metrics from Monitor wrapper
# -----------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_timestamps: list[int] = []
        self._step_count = 0

    def _on_step(self) -> bool:
        self._step_count += self.training_env.num_envs
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
                self.episode_lengths.append(ep["l"])
                self.episode_timestamps.append(self._step_count)
        return True


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_rewards(rewards: list[float], timestamps: list[int], save_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, rewards, alpha=0.25, color="steelblue", linewidth=0.8)

    window = max(1, min(50, len(rewards) // 5))
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            timestamps[window - 1:],
            smoothed,
            color="steelblue",
            linewidth=2,
            label=f"rolling avg (w={window})",
        )

    ax.set_xlabel("Total Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("PPO Training — Varaha Wildfire Logistics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reward curve → {save_path}")


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(model, n_episodes: int = 20, deterministic: bool = True):
    rewards, lengths, deliveries, successes = [], [], [], []

    for ep in range(n_episodes):
        env = VarahaSB3Env()
        obs, _ = env.reset(seed=ep * 77)
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated

        trace = env.get_trace()
        rewards.append(total_r)
        lengths.append(trace["summary"]["total_steps"])
        deliveries.append(len(trace["summary"]["delivered"]))
        successes.append(trace["summary"]["success"])

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_deliveries": float(np.mean(deliveries)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": n_episodes,
    }


# -----------------------------------------------------------------------
# Trajectory saving (for visualiser)
# -----------------------------------------------------------------------

def save_trajectories(model, save_dir: str, n: int = 3, prefix: str = "trace_trained"):
    paths = []
    for i in range(n):
        env = VarahaSB3Env()
        obs, _ = env.reset(seed=i * 42)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        out = os.path.join(save_dir, f"{prefix}_{i}.json")
        with open(out, "w") as f:
            json.dump(env.get_trace(), f, indent=2)
        t = env.get_trace()["summary"]
        print(f"  {out}  steps={t['total_steps']}  reward={t['cumulative_reward']:.1f}  "
              f"delivered={t['delivered']}  success={t['success']}")
        paths.append(out)
    return paths


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train(total_timesteps: int = 10_000_000, n_envs: int = 32, save_dir: str = "./results"):
    os.makedirs(save_dir, exist_ok=True)

    def make_env(rank):
        def _init():
            env = VarahaSB3Env()
            env = Monitor(env)
            return env
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
    )

    callback = MetricsCallback()

    print("=" * 60)
    print(f"  PPO Training — {total_timesteps:,} timesteps, {n_envs} envs (SubprocVecEnv)")
    print("=" * 60)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s  ({len(callback.episode_rewards)} episodes)")

    # Save model
    model_path = os.path.join(save_dir, "ppo_varaha")
    model.save(model_path)
    print(f"  Model saved → {model_path}.zip")

    # Reward curve
    if callback.episode_rewards:
        plot_rewards(
            callback.episode_rewards,
            callback.episode_timestamps,
            os.path.join(save_dir, "reward_curve.png"),
        )

    # Evaluate
    print("\n  Evaluating trained policy (20 episodes, deterministic)...")
    metrics = evaluate(model, n_episodes=20)
    metrics["training_timesteps"] = total_timesteps
    metrics["training_seconds"] = round(elapsed, 1)
    metrics["training_episodes"] = len(callback.episode_rewards)
    print(f"    mean_reward     = {metrics['mean_reward']:.1f} ± {metrics['std_reward']:.1f}")
    print(f"    mean_deliveries = {metrics['mean_deliveries']:.2f}")
    print(f"    success_rate    = {metrics['success_rate']:.2%}")
    print(f"    mean_length     = {metrics['mean_length']:.0f}")

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # Save trajectories for visualiser
    print("\n  Saving trained trajectories...")
    save_trajectories(model, save_dir, n=3, prefix="trace_trained")

    # Also save an untrained trajectory for comparison
    print("\n  Saving untrained trajectory for comparison...")
    untrained = PPO("MlpPolicy", DummyVecEnv([make_env(0)]), device="cpu")
    save_trajectories(untrained, save_dir, n=1, prefix="trace_untrained")

    print("\n" + "=" * 60)
    print("  Done. Results in:", save_dir)
    print("=" * 60)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Varaha")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="./results")
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, n_envs=args.n_envs, save_dir=args.save_dir)
