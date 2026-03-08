"""Hard-map world sampling for PPO training.

Provides a weighted curriculum-style world_fn that samples from:
- Hardcore random worlds (v2)
- Stress test maps (final_boss, altitude_gauntlet)
- Other generalization maps
- Optional JSON hard examples from disk
"""

from __future__ import annotations

import glob
import random
from typing import Any, Callable

from eval_generalization import (
    map_altitude_gauntlet,
    map_clustered_fires,
    map_dense_obstacles,
    map_final_boss,
    map_fire_corridor,
    map_long_range,
    map_reversed,
)
from varaha_env import build_hardcore_world_v2
from world_loader import world_fn_from_json


def build_hard_mix_world_fn(
    hard_example_glob: str = "results/hard_examples/*.json",
) -> Callable[[Any], None]:
    """Create a weighted mixed world generator focused on hardest maps."""
    pool: list[tuple[str, Callable[[Any], None], float]] = [
        ("hardcore_v2", build_hardcore_world_v2, 3.0),
        ("final_boss", map_final_boss, 4.0),
        ("altitude_gauntlet", map_altitude_gauntlet, 4.0),
        ("dense_obstacles", map_dense_obstacles, 1.5),
        ("clustered_fires", map_clustered_fires, 1.5),
        ("fire_corridor", map_fire_corridor, 1.0),
        ("long_range", map_long_range, 1.0),
        ("reversed", map_reversed, 1.0),
    ]

    json_paths = sorted(glob.glob(hard_example_glob))
    for p in json_paths:
        try:
            wf = world_fn_from_json(p)
        except Exception:
            continue
        # Keep hard JSON examples but not at dominant weight.
        pool.append((f"json:{p}", wf, 1.5))

    names = [x[0] for x in pool]
    fns = [x[1] for x in pool]
    weights = [x[2] for x in pool]

    def _mixed_world(env: Any) -> None:
        idx = random.choices(range(len(fns)), weights=weights, k=1)[0]
        setattr(env, "_sampled_world_name", names[idx])
        fns[idx](env)

    return _mixed_world

