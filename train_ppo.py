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
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from eval_generalization import map_altitude_gauntlet, map_final_boss
from hard_map_mix import build_hard_mix_world_fn
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
# Stress-map eval helper
# -----------------------------------------------------------------------

def evaluate_stress_maps(model, save_dir: str, episodes: int = 20):
    stress_specs = [
        ("final_boss", map_final_boss),
        ("altitude_gauntlet", map_altitude_gauntlet),
    ]
    out = []
    for name, fn in stress_specs:
        m = evaluate(
            model,
            n_episodes=episodes,
            config=HARDCORE_V2_CONFIG,
            world_fn=fn,
            ultra_hard=True,
        )
        m["map"] = name
        out.append(m)
        print(
            f"    [{name}] reward={m['mean_reward']:.1f} "
            f"success={m['success_rate']:.0%} deliveries={m['mean_deliveries']:.2f}"
        )
    out_path = os.path.join(save_dir, "stress_metrics_hardmix.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Stress metrics saved -> {out_path}")
    return out


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train(total_timesteps: int = 50_000_000, n_envs: int = 64,
          save_dir: str = "./results_hardcore", time_limit_s: float = 5100,
          v2: bool = False, custom_world_fn=None, custom_world_path: str = "",
          hard_mix: bool = False, hard_example_glob: str = "results/hard_examples/*.json",
          vec_backend: str = "subproc", device: str = "cuda", require_cuda: bool = False,
          quick_15m: bool = False, eval_episodes: int = 30,
          trace_count: int = 5, stress_eval_episodes: int = 20):
    os.makedirs(save_dir, exist_ok=True)

    if hard_mix:
        config = HARDCORE_V2_CONFIG
        world_fn = build_hard_mix_world_fn(hard_example_glob=hard_example_glob)
        ultra_hard = True
        net_arch = [1024, 512]
        n_steps = 4096
        batch_size = 2048
        n_epochs = 10
        learning_rate = 3e-4
        obs_dim = VarahaSB3Env.V2_OBS_DIM
    elif v2:
        config = HARDCORE_V2_CONFIG
        world_fn = build_hardcore_world_v2
        ultra_hard = True
        net_arch = [1024, 512]
        n_steps = 4096
        batch_size = 2048
        n_epochs = 10
        learning_rate = 3e-4
        obs_dim = VarahaSB3Env.V2_OBS_DIM
    else:
        config = HARDCORE_CONFIG
        world_fn = build_hardcore_world
        ultra_hard = False
        net_arch = [512, 512]
        n_steps = 4096
        batch_size = 2048
        n_epochs = 10
        learning_rate = 3e-4
        obs_dim = VarahaSB3Env.OBS_DIM

    if custom_world_fn is not None:
        world_fn = custom_world_fn
        ultra_hard = False

    if quick_15m:
        # Fast preset to keep full pipeline near a 15-minute budget.
        if ultra_hard:
            net_arch = [512, 256]
        else:
            net_arch = [384, 256]
        n_steps = 1024
        batch_size = min(1024, max(64, n_steps * max(1, n_envs) // 2))
        n_epochs = 4
        learning_rate = 5e-4

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            msg = (
                "CUDA requested but torch.cuda.is_available() is False. "
                "If you expect H100, verify container GPU pass-through/driver visibility."
            )
            if require_cuda:
                raise RuntimeError(msg)
            print(f"  {msg} Falling back to CPU.")
            resolved_device = "cpu"
    else:
        resolved_device = device

    def make_env(rank):
        def _init():
            env = VarahaSB3Env(config=config, world_fn=world_fn, ultra_hard=ultra_hard)
            env = Monitor(env)
            return env
        return _init

    if vec_backend == "dummy":
        vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    else:
        try:
            vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        except PermissionError as exc:
            print(f"  SubprocVecEnv unavailable ({exc}). Falling back to DummyVecEnv.")
            vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=net_arch),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=resolved_device,
    )

    callback = MetricsCallback(time_limit_s=time_limit_s)

    print("=" * 70)
    mode_tag = "(HARD-MIX V2)" if hard_mix else "(V2 ULTRA)" if v2 else ""
    if quick_15m:
        mode_tag = f"{mode_tag} (QUICK-15M)".strip()
    print(f"  PPO HARDCORE Training {mode_tag}")
    print(f"  Network: {net_arch}  |  Obs dim: {obs_dim}")
    print(f"  Timesteps: {total_timesteps:,}  |  Envs: {n_envs}  |  Time limit: {time_limit_s:.0f}s")
    print(f"  batch_size={batch_size}  n_steps={n_steps}  n_epochs={n_epochs}  ent_coef=0.02")
    print(f"  Config: max_steps={config.max_episode_steps}  collision_penalty=500")
    print(f"  Device: {resolved_device}")
    if hard_mix:
        print(f"  Hard mix examples glob: {hard_example_glob}")
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

    print(f"\n  Evaluating trained policy ({eval_episodes} episodes, deterministic)...")
    metrics = evaluate(model, n_episodes=eval_episodes, config=config, world_fn=world_fn, ultra_hard=ultra_hard)
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

    if hard_mix:
        print("\n  Evaluating stress maps (final_boss + altitude_gauntlet)...")
        evaluate_stress_maps(model, save_dir=save_dir, episodes=stress_eval_episodes)

    print("\n  Saving hardcore trajectories...")
    save_trajectories(model, save_dir, n=trace_count, prefix="trace_hardcore",
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
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail immediately if CUDA is unavailable (recommended for H100 runs).",
    )
    parser.add_argument(
        "--quick-15m",
        action="store_true",
        help="Fast preset: targets ~15-minute runs by reducing rollout/model/eval sizes.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Evaluation episodes after training.",
    )
    parser.add_argument(
        "--trace-count",
        type=int,
        default=5,
        help="Number of trained trajectories to save.",
    )
    parser.add_argument(
        "--stress-episodes",
        type=int,
        default=20,
        help="Stress-map evaluation episodes used with --hard-mix.",
    )
    parser.add_argument(
        "--vec-backend",
        choices=["subproc", "dummy"],
        default="subproc",
        help="Vectorized env backend. Use dummy in restricted/sandboxed environments.",
    )
    parser.add_argument(
        "--hard-mix",
        action="store_true",
        help="Train on weighted hard map mix (focuses final_boss + altitude_gauntlet + hard examples).",
    )
    parser.add_argument(
        "--hard-example-glob",
        type=str,
        default="results/hard_examples/*.json",
        help="Glob for additional JSON worlds used in --hard-mix.",
    )
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

    if args.hard_mix and not args.quick_15m:
        # Defaults tuned for a stronger stress-map training pass.
        if args.timesteps == 50_000_000:
            args.timesteps = 80_000_000
        if args.n_envs == 64:
            args.n_envs = 96
        if args.save_dir == "./results_hardcore":
            args.save_dir = "./results_hardmix_v3"
        if args.time_limit == 5100:
            args.time_limit = 9000

    if args.quick_15m:
        if args.timesteps == 50_000_000:
            args.timesteps = 12_000_000
        if args.n_envs == 64:
            args.n_envs = 24 if not args.hard_mix else 32
        if args.save_dir == "./results_hardcore":
            args.save_dir = "./results_quick15m"
        if args.time_limit == 5100:
            args.time_limit = 900
        if args.eval_episodes == 30:
            args.eval_episodes = 8
        if args.trace_count == 5:
            args.trace_count = 2
        if args.stress_episodes == 20:
            args.stress_episodes = 6

    custom_world_fn = world_fn_from_json(args.world_json) if args.world_json else None
    train(total_timesteps=args.timesteps, n_envs=args.n_envs,
          save_dir=args.save_dir, time_limit_s=args.time_limit, v2=args.v2,
          custom_world_fn=custom_world_fn, custom_world_path=args.world_json,
          hard_mix=args.hard_mix, hard_example_glob=args.hard_example_glob,
          vec_backend=args.vec_backend, device=args.device, require_cuda=args.require_cuda,
          quick_15m=args.quick_15m, eval_episodes=args.eval_episodes,
          trace_count=args.trace_count, stress_eval_episodes=args.stress_episodes)
