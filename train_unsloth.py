#!/usr/bin/env python3
"""Minimal Unsloth SFT pipeline for Varaha long-horizon planning.

Pipeline:
1. Roll out an instruction-aware expert to create SFT data.
2. Fine-tune a small instruct model with Unsloth (LoRA).
3. Evaluate reward before vs after fine-tuning on held-out seeds.

Example:
    python train_unsloth.py \
      --output-dir ./results_unsloth \
      --dataset-episodes 12 \
      --dataset-max-records 3000 \
      --train-steps 120 \
      --eval-episodes 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from varaha_env import VarahaConfig, VarahaEnv, build_hardcore_world, build_hardcore_world_v2
from world_loader import world_fn_from_json


SYSTEM_PROMPT = (
    "You are a wildfire drone planner. Output ONLY compact JSON with keys "
    "ax, ay, az, deliver, recharge, tool_call. ax/ay/az must be normalized "
    "floats in [-1, 1]."
)


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    deliveries: int
    steps: int


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _fmt_vec(v: dict[str, Any]) -> str:
    return f"{float(v.get('x', 0.0)):.1f},{float(v.get('y', 0.0)):.1f},{float(v.get('z', 0.0)):.1f}"


def observation_to_prompt(obs: dict[str, Any]) -> str:
    """Compact prompt text describing the current state."""
    mission = obs.get("mission", {})
    next_inst = mission.get("next_instruction") or {}

    lines: list[str] = []
    lines.append(
        f"step={obs.get('step', 0)}/{obs.get('max_steps', 0)} "
        f"battery={float(obs.get('battery', 0.0)):.2f} "
        f"payload={int(bool(obs.get('carrying_payload', False)))} "
        f"alive={int(bool(obs.get('alive', True)))}"
    )
    lines.append(f"drone_pos={_fmt_vec(obs.get('drone_position', {}))}")
    lines.append(f"drone_vel={_fmt_vec(obs.get('drone_velocity', {}))}")
    lines.append(
        "mission="
        f"enabled:{int(bool(mission.get('enabled', False)))},"
        f"progress:{float(mission.get('progress', 0.0)):.3f},"
        f"violations:{int(mission.get('violations', 0))},"
        f"next_kind:{next_inst.get('kind', 'none')},"
        f"next_target:{next_inst.get('target_id', '')},"
        f"next_tool:{next_inst.get('tool_name', '')}"
    )

    targets = obs.get("targets", [])[:4]
    for t in targets:
        lines.append(
            "target "
            f"id={t.get('id','')} "
            f"rel={_fmt_vec(t.get('relative_position', {}))} "
            f"urg={float(t.get('urgency', 0.0)):.2f} "
            f"del={int(bool(t.get('delivered', False)))}"
        )

    hazards = obs.get("hazards", [])[:3]
    for h in hazards:
        lines.append(
            "hazard "
            f"id={h.get('id','')} "
            f"rel={_fmt_vec(h.get('relative_position', {}))} "
            f"height={float(h.get('current_height', 0.0)):.1f} "
            f"sev={float(h.get('severity', 0.0)):.2f}"
        )

    obstacles = obs.get("obstacles", [])[:5]
    for o in obstacles:
        lines.append(
            "obstacle "
            f"type={o.get('type','')} "
            f"kind={o.get('kind','')} "
            f"dist={float(o.get('distance', 0.0)):.1f} "
            f"rel={_fmt_vec(o.get('relative_position', {}))} "
            f"h={float(o.get('height', 0.0)):.1f}"
        )

    responders = obs.get("responders", [])[:2]
    for r in responders:
        idir = r.get("intel_direction", {})
        lines.append(
            "responder "
            f"id={r.get('id','')} "
            f"target={r.get('linked_target_id','')} "
            f"status={r.get('status','')} "
            f"intel={r.get('latest_intel','none')} "
            f"idir={float(idir.get('x', 0.0)):.1f},{float(idir.get('y', 0.0)):.1f} "
            f"isev={float(r.get('intel_severity', 0.0)):.2f}"
        )

    return "\n".join(lines)


class InstructionAwareExpert:
    """A simple scripted planner that follows mission instructions."""

    def __init__(self, env: VarahaEnv):
        self.env = env
        self.cruise_z = min(env.cfg.world_z - 3.0, 165.0)
        self.kp = 0.35
        self.kd = 0.78

    def _target_for_instruction(self, obs: dict[str, Any]) -> Any:
        mission = obs.get("mission", {})
        next_inst = mission.get("next_instruction") or {}
        kind = next_inst.get("kind", "")
        target_id = next_inst.get("target_id", "")
        if kind == "deliver_target" and target_id:
            for t in self.env.targets:
                if t.id == target_id and not t.delivered:
                    return t
        return None

    def _pick_goal(self, obs: dict[str, Any]) -> tuple[Any, Any]:
        mission = obs.get("mission", {})
        next_inst = mission.get("next_instruction") or {}
        kind = next_inst.get("kind", "")

        inst_target = self._target_for_instruction(obs)
        if inst_target is not None:
            return inst_target.position, inst_target

        if kind == "return_base":
            return self.env.base.position, None

        pending = [t for t in self.env.targets if not t.delivered]
        if pending:
            pos = self.env.drone.position
            nearest = min(
                pending,
                key=lambda t: (t.position.x - pos.x) ** 2 + (t.position.y - pos.y) ** 2,
            )
            return nearest.position, nearest
        return self.env.base.position, None

    def _tool_call(self, obs: dict[str, Any]) -> str:
        mission = obs.get("mission", {})
        next_inst = mission.get("next_instruction") or {}
        if next_inst.get("kind") != "tool_call":
            return ""
        tool_name = str(next_inst.get("tool_name") or "mission_report")
        target_id = str(next_inst.get("target_id") or "")
        if tool_name == "request_intel" and target_id:
            return f"{tool_name}:{target_id}"
        return tool_name

    def _can_deliver_now(self) -> bool:
        pos = self.env.drone.position
        for tgt in self.env.targets:
            if tgt.delivered:
                continue
            dx = pos.x - tgt.position.x
            dy = pos.y - tgt.position.y
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            alt_above = pos.z - tgt.position.z
            if horiz_dist <= tgt.delivery_radius and -10.0 <= alt_above <= tgt.delivery_radius * 2:
                return True
        return False

    def _can_recharge_now(self) -> bool:
        pos = self.env.drone.position
        base = self.env.base.position
        hdist = math.sqrt((pos.x - base.x) ** 2 + (pos.y - base.y) ** 2)
        return hdist <= self.env.base.recharge_radius

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        cfg = self.env.cfg
        pos = self.env.drone.position
        vel = self.env.drone.velocity

        goal, target_obj = self._pick_goal(obs)
        dx = goal.x - pos.x
        dy = goal.y - pos.y
        dxy = max(math.sqrt(dx * dx + dy * dy), 1e-6)

        target_z = self.cruise_z
        if target_obj is not None and dxy < 420.0:
            target_z = min(cfg.world_z - 2.0, max(target_obj.position.z + 22.0, self.cruise_z * 0.55))
        desired_speed = min(cfg.max_speed * 0.8, max(4.0, dxy * 0.42))

        desired_vx = desired_speed * dx / dxy
        desired_vy = desired_speed * dy / dxy
        desired_vz = _clamp((target_z - pos.z) * 1.05, -cfg.max_speed, cfg.max_speed)

        ax = (desired_vx - vel.x) / cfg.dt
        ay = (desired_vy - vel.y) / cfg.dt
        az = (desired_vz - vel.z) / cfg.dt

        nearest = (obs.get("obstacles") or [{}])[0]
        if nearest and float(nearest.get("distance", 999999.0)) < 55.0:
            rel = nearest.get("relative_position", {})
            # Simple repulsion from the closest obstacle footprint.
            ax -= math.copysign(cfg.max_acceleration * 0.7, float(rel.get("x", 0.0) or 1.0))
            ay -= math.copysign(cfg.max_acceleration * 0.7, float(rel.get("y", 0.0) or 1.0))
            if pos.z < self.cruise_z:
                az += cfg.max_acceleration * 0.4

        accel = np.array([ax, ay, az], dtype=np.float32)
        norm = np.linalg.norm(accel)
        if norm > cfg.max_acceleration and norm > 1e-8:
            accel *= float(cfg.max_acceleration / norm)

        all_delivered = all(t.delivered for t in self.env.targets)
        recharge = self._can_recharge_now() and (all_delivered or self.env.drone.battery < cfg.battery_capacity * 0.65)

        return {
            "ax": float(accel[0]),
            "ay": float(accel[1]),
            "az": float(accel[2]),
            "deliver": self._can_deliver_now(),
            "recharge": recharge,
            "tool_call": self._tool_call(obs),
        }


def action_to_json(action: dict[str, Any], cfg: VarahaConfig) -> str:
    payload = {
        "ax": round(_clamp(float(action.get("ax", 0.0)) / cfg.max_acceleration, -1.0, 1.0), 4),
        "ay": round(_clamp(float(action.get("ay", 0.0)) / cfg.max_acceleration, -1.0, 1.0), 4),
        "az": round(_clamp(float(action.get("az", 0.0)) / cfg.max_acceleration, -1.0, 1.0), 4),
        "deliver": bool(action.get("deliver", False)),
        "recharge": bool(action.get("recharge", False)),
        "tool_call": str(action.get("tool_call", "")),
    }
    return json.dumps(payload, separators=(",", ":"))


def parse_action_json(text: str, cfg: VarahaConfig) -> dict[str, Any]:
    """Parse model output into environment action dict (with safe fallback)."""
    # Fallback: hover in place.
    default = {
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
        "deliver": False,
        "recharge": False,
        "tool_call": "",
    }

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return default

    raw = match.group(0).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            return default

    if not isinstance(data, dict):
        return default

    ax_n = _clamp(float(data.get("ax", data.get("ax_norm", 0.0))), -1.0, 1.0)
    ay_n = _clamp(float(data.get("ay", data.get("ay_norm", 0.0))), -1.0, 1.0)
    az_n = _clamp(float(data.get("az", data.get("az_norm", 0.0))), -1.0, 1.0)

    return {
        "ax": ax_n * cfg.max_acceleration,
        "ay": ay_n * cfg.max_acceleration,
        "az": az_n * cfg.max_acceleration,
        "deliver": _boolish(data.get("deliver", False)),
        "recharge": _boolish(data.get("recharge", False)),
        "tool_call": str(data.get("tool_call", "")).strip(),
    }


def generate_expert_dataset(
    cfg: VarahaConfig,
    world_fn: Any,
    episodes: int,
    max_steps: int,
    max_records: int,
    seed_offset: int,
) -> tuple[list[dict[str, str]], list[EpisodeResult]]:
    records: list[dict[str, str]] = []
    episode_results: list[EpisodeResult] = []

    for ep in range(episodes):
        env = VarahaEnv(config=cfg, world_fn=world_fn)
        obs = env.reset(seed=seed_offset + ep * 173)
        planner = InstructionAwareExpert(env)
        total_reward = 0.0

        for _ in range(max_steps):
            action = planner.act(obs)
            records.append(
                {
                    "prompt": observation_to_prompt(obs),
                    "response": action_to_json(action, cfg),
                }
            )
            obs, reward, done, _info = env.step(action)
            total_reward += reward
            if done or len(records) >= max_records:
                break

        trace = env.get_trace()
        episode_results.append(
            EpisodeResult(
                reward=total_reward,
                success=bool(trace["summary"]["success"]),
                deliveries=len(trace["summary"]["delivered"]),
                steps=int(trace["summary"]["total_steps"]),
            )
        )
        if len(records) >= max_records:
            break

    return records, episode_results


def _apply_chat_template(tokenizer: Any, prompt: str, response: str = "") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if response:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(response == ""),
    )


def _extract_completion(full_text: str, prompt_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text):].strip()
    return full_text.strip()


def run_lm_episode(
    model: Any,
    tokenizer: Any,
    cfg: VarahaConfig,
    world_fn: Any,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    import torch

    env = VarahaEnv(config=cfg, world_fn=world_fn)
    obs = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(max_steps):
        prompt_body = observation_to_prompt(obs)
        prompt_text = _apply_chat_template(tokenizer, prompt_body, response="")
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        completion = _extract_completion(decoded, prompt_text)
        action = parse_action_json(completion, cfg)

        obs, reward, done, _info = env.step(action)
        total_reward += reward
        if done:
            break

    trace = env.get_trace()
    return EpisodeResult(
        reward=total_reward,
        success=bool(trace["summary"]["success"]),
        deliveries=len(trace["summary"]["delivered"]),
        steps=int(trace["summary"]["total_steps"]),
    )


def evaluate_lm_policy(
    model: Any,
    tokenizer: Any,
    cfg: VarahaConfig,
    world_fn: Any,
    episodes: int,
    seed_offset: int,
    max_steps: int,
) -> dict[str, float]:
    results: list[EpisodeResult] = []
    for i in range(episodes):
        results.append(
            run_lm_episode(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                world_fn=world_fn,
                seed=seed_offset + i * 131,
                max_steps=max_steps,
            )
        )

    rewards = [r.reward for r in results]
    deliveries = [r.deliveries for r in results]
    successes = [1.0 if r.success else 0.0 for r in results]
    steps = [r.steps for r in results]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_deliveries": float(np.mean(deliveries)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(steps)),
        "episodes": float(episodes),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth fine-tuning for Varaha planning")
    p.add_argument("--output-dir", default="./results_unsloth")
    p.add_argument("--model-name", default="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    p.add_argument("--max-seq-length", type=int, default=1024)

    p.add_argument("--dataset-episodes", type=int, default=12)
    p.add_argument("--dataset-max-steps", type=int, default=600)
    p.add_argument("--dataset-max-records", type=int, default=3000)
    p.add_argument("--seed-offset", type=int, default=40_000)

    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)

    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--eval-max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--world-json", default="", help="Optional custom world JSON for training/eval.")
    p.add_argument("--v2", action="store_true", help="Use ultra-hard world generator by default.")
    p.add_argument("--instruction-count", type=int, default=60)
    p.add_argument("--dense-reward", action="store_true", help="Disable sparse instruction rewards.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = VarahaConfig(
        max_episode_steps=max(2600, args.eval_max_steps),
        collision_penalty=500.0,
        obstacle_proximity_penalty=1.5,
        obstacle_proximity_radius=80.0,
        distance_shaping_factor=0.05,
        hazard_penalty=5.0,
        instruction_mode=True,
        instruction_count=args.instruction_count,
        sparse_reward_mode=not args.dense_reward,
    )

    if args.world_json:
        world_fn = world_fn_from_json(args.world_json)
        world_name = args.world_json
    else:
        world_fn = build_hardcore_world_v2 if args.v2 else build_hardcore_world
        world_name = "build_hardcore_world_v2" if args.v2 else "build_hardcore_world"

    print("=" * 72)
    print("Generating expert dataset...")
    records, expert_eps = generate_expert_dataset(
        cfg=cfg,
        world_fn=world_fn,
        episodes=args.dataset_episodes,
        max_steps=args.dataset_max_steps,
        max_records=args.dataset_max_records,
        seed_offset=args.seed_offset,
    )
    if not records:
        raise RuntimeError("Dataset generation produced zero records.")

    dataset_path = out_dir / "unsloth_sft_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  records={len(records)}  episodes={len(expert_eps)}  world={world_name}")
    print(f"  dataset -> {dataset_path}")

    expert_stats = {
        "mean_reward": float(np.mean([e.reward for e in expert_eps])),
        "success_rate": float(np.mean([1.0 if e.success else 0.0 for e in expert_eps])),
        "mean_deliveries": float(np.mean([e.deliveries for e in expert_eps])),
    }

    try:
        import torch
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError(
            "Missing Unsloth training dependencies. Install with:\n"
            "  pip install unsloth trl transformers datasets accelerate bitsandbytes peft"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Unsloth training script.")

    print("=" * 72)
    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=not args.no_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Evaluating base model on held-out seeds...")
    FastLanguageModel.for_inference(model)
    baseline = evaluate_lm_policy(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        world_fn=world_fn,
        episodes=args.eval_episodes,
        seed_offset=args.seed_offset + 900_000,
        max_steps=args.eval_max_steps,
    )
    print(
        f"  baseline reward={baseline['mean_reward']:.1f} "
        f"success={baseline['success_rate']:.0%} deliveries={baseline['mean_deliveries']:.2f}"
    )

    print("=" * 72)
    print("Preparing SFT dataset...")
    hf_ds = Dataset.from_list(records)

    def _format_row(row: dict[str, str]) -> dict[str, str]:
        return {"text": _apply_chat_template(tokenizer, row["prompt"], row["response"])}

    hf_ds = hf_ds.map(_format_row, remove_columns=hf_ds.column_names)
    if len(hf_ds) >= 20:
        split = hf_ds.train_test_split(test_size=min(0.05, 200 / len(hf_ds)), seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = hf_ds
        eval_ds = None
    print(f"  train_rows={len(train_ds)} eval_rows={len(eval_ds) if eval_ds is not None else 0}")

    print("=" * 72)
    print("Applying LoRA adapters and training with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.train_steps,
        warmup_steps=max(1, args.train_steps // 20),
        logging_steps=max(1, args.train_steps // 10),
        save_strategy="no",
        report_to="none",
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        weight_decay=0.01,
        seed=args.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=train_args,
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  adapter -> {adapter_dir}")

    print("=" * 72)
    print("Evaluating fine-tuned model...")
    FastLanguageModel.for_inference(model)
    finetuned = evaluate_lm_policy(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        world_fn=world_fn,
        episodes=args.eval_episodes,
        seed_offset=args.seed_offset + 900_000,
        max_steps=args.eval_max_steps,
    )
    print(
        f"  finetuned reward={finetuned['mean_reward']:.1f} "
        f"success={finetuned['success_rate']:.0%} deliveries={finetuned['mean_deliveries']:.2f}"
    )

    improvement = finetuned["mean_reward"] - baseline["mean_reward"]
    print(f"  reward improvement = {improvement:+.1f}")

    metrics = {
        "world": world_name,
        "expert": expert_stats,
        "baseline": baseline,
        "finetuned": finetuned,
        "reward_improvement": float(improvement),
        "config": {
            "model_name": args.model_name,
            "dataset_records": len(records),
            "dataset_episodes": len(expert_eps),
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "instruction_mode": cfg.instruction_mode,
            "sparse_reward_mode": cfg.sparse_reward_mode,
        },
    }

    metrics_path = out_dir / "unsloth_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  metrics -> {metrics_path}")

    # Optional plot for quick submission screenshot.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["baseline", "finetuned"], [baseline["mean_reward"], finetuned["mean_reward"]], color=["#9aa7bf", "#4caf50"])
        ax.set_ylabel("Mean Reward")
        ax.set_title("Unsloth Reward Improvement")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        plot_path = out_dir / "reward_improvement.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"  plot -> {plot_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
