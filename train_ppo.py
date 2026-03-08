#!/usr/bin/env python3
"""Train a PPO policy on VarahaEnv (hardcore mode) using Stable-Baselines3.

Usage:
    python train_ppo.py                                  # default hardcore
    python train_ppo.py --timesteps 50000000 --n-envs 64
    python train_ppo.py --time-limit 5400                # 90 minutes
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
from varaha_env import VarahaConfig, build_hardcore_world, build_hardcore_world_v2
from world_loader import world_fn_from_json


# -----------------------------------------------------------------------
# Config for hardcore environments
# -----------------------------------------------------------------------

HARDCORE_CONFIG = VarahaConfig(
    max_episode_steps=3000,
    collision_penalty=500.0,
    obstacle_proximity_penalty=1.5,
    obstacle_proximity_radius=80.0,
    distance_shaping_factor=0.05,
    hazard_penalty=5.0,
)

HARDCORE_V2_CONFIG = VarahaConfig(
    max_episode_steps=3500,
    collision_penalty=500.0,
    obstacle_proximity_penalty=1.5,
    obstacle_proximity_radius=80.0,
    distance_shaping_factor=0.05,
    hazard_penalty=5.0,
)


# -----------------------------------------------------------------------
# Callback — records episode-level metrics from Monitor wrapper
# -----------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    def __init__(self, time_limit_s: float = 0, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_timestamps: list[int] = []
        self._step_count = 0
        self._t0 = time.time()
        self._time_limit = time_limit_s

    def _on_step(self) -> bool:
        self._step_count += self.training_env.num_envs
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
                self.episode_lengths.append(ep["l"])
                self.episode_timestamps.append(self._step_count)

        if self._time_limit > 0 and (time.time() - self._t0) >= self._time_limit:
            print(f"\n  TIME LIMIT REACHED ({self._time_limit:.0f}s). Stopping training.")
            return False
        return True


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_rewards(rewards: list[float], timestamps: list[int], save_path: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, rewards, alpha=0.15, color="steelblue", linewidth=0.5)

    window = max(1, min(100, len(rewards) // 5))
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
    ax.set_title("PPO Hardcore Training — Varaha Wildfire Logistics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reward curve -> {save_path}")


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(model, n_episodes: int = 30, deterministic: bool = True,
             config=None, world_fn=None, ultra_hard: bool = False):
    config = config or HARDCORE_CONFIG
    world_fn = world_fn or build_hardcore_world
    rewards, lengths, deliveries, successes = [], [], [], []

    for ep in range(n_episodes):
        env = VarahaSB3Env(config=config, world_fn=world_fn, ultra_hard=ultra_hard)
        obs, _ = env.reset(seed=ep * 77 + 9999)
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

def save_trajectories(model, save_dir: str, n: int = 5, prefix: str = "trace_hardcore",
                      config=None, world_fn=None, ultra_hard: bool = False):
    config = config or HARDCORE_CONFIG
    world_fn = world_fn or build_hardcore_world
    paths = []
    for i in range(n):
        env = VarahaSB3Env(config=config, world_fn=world_fn, ultra_hard=ultra_hard)
        obs, _ = env.reset(seed=i * 42 + 7777)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        out = os.path.join(save_dir, f"{prefix}_{i}.json")
        with open(out, "w") as f:
            json.dump(env.get_trace(), f)
        t = env.get_trace()["summary"]
        print(f"  {out}  steps={t['total_steps']}  reward={t['cumulative_reward']:.1f}  "
              f"delivered={t['delivered']}  success={t['success']}")
        paths.append(out)
    return paths


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train(total_timesteps: int = 50_000_000, n_envs: int = 64,
          save_dir: str = "./results_hardcore", time_limit_s: float = 5100,
          v2: bool = False, custom_world_fn=None, custom_world_path: str = ""):
    os.makedirs(save_dir, exist_ok=True)

    if v2:
        config = HARDCORE_V2_CONFIG
        world_fn = build_hardcore_world_v2
        ultra_hard = True
        net_arch = [1024, 512]
        n_steps = 4096
        batch_size = 2048
        obs_dim = VarahaSB3Env.V2_OBS_DIM
    else:
        config = HARDCORE_CONFIG
        world_fn = build_hardcore_world
        ultra_hard = False
        net_arch = [512, 512]
        n_steps = 4096
        batch_size = 2048
        obs_dim = VarahaSB3Env.OBS_DIM

    if custom_world_fn is not None:
        world_fn = custom_world_fn
        ultra_hard = False

    def make_env(rank):
        def _init():
            env = VarahaSB3Env(config=config, world_fn=world_fn, ultra_hard=ultra_hard)
            env = Monitor(env)
            return env
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=net_arch),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
    )

    callback = MetricsCallback(time_limit_s=time_limit_s)

    print("=" * 70)
    print(f"  PPO HARDCORE Training {'(V2 ULTRA)' if v2 else ''}")
    print(f"  Network: {net_arch}  |  Obs dim: {obs_dim}")
    print(f"  Timesteps: {total_timesteps:,}  |  Envs: {n_envs}  |  Time limit: {time_limit_s:.0f}s")
    print(f"  batch_size={batch_size}  n_steps={n_steps}  ent_coef=0.02")
    print(f"  Config: max_steps={config.max_episode_steps}  collision_penalty=500")
    if custom_world_path:
        print(f"  Custom world: {custom_world_path}")
    print("=" * 70)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.1f}s  ({len(callback.episode_rewards)} episodes)")

    model_path = os.path.join(save_dir, "ppo_varaha_hardcore")
    model.save(model_path)
    print(f"  Model saved -> {model_path}.zip")

    if callback.episode_rewards:
        plot_rewards(
            callback.episode_rewards,
            callback.episode_timestamps,
            os.path.join(save_dir, "reward_curve_hardcore.png"),
        )

    print("\n  Evaluating trained policy (30 episodes, deterministic)...")
    metrics = evaluate(model, n_episodes=30, config=config, world_fn=world_fn, ultra_hard=ultra_hard)
    metrics["training_timesteps"] = callback._step_count
    metrics["training_seconds"] = round(elapsed, 1)
    metrics["training_episodes"] = len(callback.episode_rewards)
    metrics["obs_dim"] = obs_dim
    metrics["net_arch"] = net_arch
    print(f"    mean_reward     = {metrics['mean_reward']:.1f} +/- {metrics['std_reward']:.1f}")
    print(f"    mean_deliveries = {metrics['mean_deliveries']:.2f}")
    print(f"    success_rate    = {metrics['success_rate']:.2%}")
    print(f"    mean_length     = {metrics['mean_length']:.0f}")

    metrics_path = os.path.join(save_dir, "metrics_hardcore.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {metrics_path}")

    print("\n  Saving hardcore trajectories...")
    save_trajectories(model, save_dir, n=5, prefix="trace_hardcore",
                      config=config, world_fn=world_fn, ultra_hard=ultra_hard)

    print("\n  Saving untrained trajectory for comparison...")
    untrained = PPO("MlpPolicy", DummyVecEnv([make_env(0)]),
                    policy_kwargs=dict(net_arch=net_arch), device="cpu")
    save_trajectories(untrained, save_dir, n=1, prefix="trace_untrained_hardcore",
                      config=config, world_fn=world_fn, ultra_hard=ultra_hard)

    vec_env.close()

    print("\n" + "=" * 70)
    print(f"  DONE. Results in: {save_dir}")
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total timesteps: {callback._step_count:,}")
    print("=" * 70)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Hardcore on Varaha")
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="./results_hardcore")
    parser.add_argument("--time-limit", type=float, default=5100,
                        help="Max training time in seconds (default 5100 = 85 min)")
    parser.add_argument("--v2", action="store_true",
                        help="Ultra-hard: [1024,512] net, 96 envs, 60M steps, 169-dim obs, 3500 max steps")
    parser.add_argument(
        "--world-json",
        type=str,
        default="",
        help="Optional custom world JSON (trace/render_state/object export).",
    )
    args = parser.parse_args()

    if args.v2:
        args.timesteps = 60_000_000
        args.n_envs = 96
        args.save_dir = "./results_hardcore_v2"
        args.time_limit = 5600

    custom_world_fn = world_fn_from_json(args.world_json) if args.world_json else None
    train(total_timesteps=args.timesteps, n_envs=args.n_envs,
          save_dir=args.save_dir, time_limit_s=args.time_limit, v2=args.v2,
          custom_world_fn=custom_world_fn, custom_world_path=args.world_json)
