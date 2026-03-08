# Varaha Training Paths

This repo now has two runnable training tracks:

1. PPO (continuous-control RL, Stable-Baselines3)
2. Unsloth SFT (LLM planner for long-horizon instruction mode)

## PPO (existing path)

Default hardcore training:

```bash
python train_ppo.py --timesteps 50000000 --n-envs 64 --save-dir ./results_hardcore
```

Quick ~15-minute training pass:

```bash
python train_ppo.py --quick-15m --hard-mix --device cuda --require-cuda
```

Ultra-hard V2:

```bash
python train_ppo.py --v2
```

Train on a custom render/trace/object world JSON:

```bash
python train_ppo.py --world-json ./sample_render_state.json --timesteps 10000000 --n-envs 32
```

`--world-json` accepts:
- `env.get_trace()` JSON (top-level `world`)
- `env.render_state()` JSON
- A scene export with top-level `objects` (box/cylinder style entries)

## Unsloth (new long-horizon planning path)

Install deps:

```bash
pip install -r requirements-unsloth.txt
```

Run a minimal end-to-end Unsloth training + reward eval:

```bash
python train_unsloth.py \
  --output-dir ./results_unsloth \
  --dataset-episodes 12 \
  --dataset-max-records 3000 \
  --train-steps 120 \
  --eval-episodes 3
```

Artifacts:
- `results_unsloth/unsloth_sft_dataset.jsonl`
- `results_unsloth/adapter/`
- `results_unsloth/unsloth_metrics.json`
- `results_unsloth/reward_improvement.png`

To train/eval Unsloth on a render-derived world:

```bash
python train_unsloth.py --world-json ./sample_render_state.json
```

## Colab notebook (minimal demo)

Use:

```text
colab_unsloth_minimal.ipynb
```

It runs:
1. dependency install
2. `train_unsloth.py`
3. reward before/after visualization

## Can we train on actual rendered buildings/objects?

Yes, with one important caveat:

- Training uses simulator geometry (`obstacles`, `cylinders`) for physics/reward.
- Cesium/Google photorealistic tiles are visual unless exported to obstacle JSON.

So the practical workflow is:
1. export buildings/objects from your render pipeline into JSON
2. load them through `--world-json`
3. train PPO or Unsloth on that geometry
