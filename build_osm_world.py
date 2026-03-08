#!/usr/bin/env python3
"""Build a Varaha world JSON from OpenStreetMap building footprints.

This script fetches OSM building ways from Overpass, converts them into
axis-aligned obstacle boxes in local metres, and emits a world JSON that can
be used directly with:

    python train_ppo.py --world-json <output.json>
    python train_unsloth.py --world-json <output.json>

Default area is downtown San Francisco.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BBox:
    south: float
    west: float
    north: float
    east: float

    def width_m(self, origin_lat: float) -> float:
        meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
        return max(1.0, (self.east - self.west) * meters_per_deg_lon)

    def height_m(self) -> float:
        return max(1.0, (self.north - self.south) * 111_320.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert OSM buildings to Varaha world JSON")
    p.add_argument(
        "--output",
        default="./results/world_sf_osm.json",
        help="Output world JSON path",
    )
    p.add_argument("--south", type=float, default=37.7680)
    p.add_argument("--west", type=float, default=-122.4320)
    p.add_argument("--north", type=float, default=37.8045)
    p.add_argument("--east", type=float, default=-122.3850)
    p.add_argument("--max-buildings", type=int, default=280, help="Max obstacles to keep")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=2800)
    p.add_argument("--world-z", type=float, default=220.0)
    p.add_argument("--default-height", type=float, default=22.0)
    p.add_argument("--timeout-s", type=float, default=90.0)
    return p.parse_args()


def _parse_height(tags: dict[str, str], default_height: float) -> float:
    h = tags.get("height", "").strip()
    if h:
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", h)
        if m:
            value = float(m.group(0))
            if "ft" in h.lower():
                value *= 0.3048
            return max(6.0, min(220.0, value))

    lv_raw = tags.get("building:levels", tags.get("levels", "")).strip()
    if lv_raw:
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", lv_raw)
        if m:
            return max(6.0, min(220.0, float(m.group(0)) * 3.2))
    return max(6.0, min(220.0, default_height))


def _to_local_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(origin_lat))
    x = (lon - origin_lon) * meters_per_deg_lon
    y = (lat - origin_lat) * meters_per_deg_lat
    return x, y


def _fetch_overpass_buildings(bbox: BBox, timeout_s: float) -> dict[str, Any]:
    # Building ways + referenced nodes.
    query = f"""
[out:json][timeout:60];
(
  way["building"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});
);
(._;>;);
out body;
"""
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ]
    last_error = ""
    for url in endpoints:
        req = urllib.request.Request(url, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status != 200:
                    last_error = f"{url} status={resp.status}"
                    continue
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{url} error={exc}"
            continue
    raise RuntimeError(f"Failed to fetch Overpass data: {last_error}")


def _footprints_to_obstacles(
    overpass: dict[str, Any],
    bbox: BBox,
    default_height: float,
    max_buildings: int,
) -> tuple[list[dict[str, Any]], int]:
    origin_lat = bbox.south
    origin_lon = bbox.west
    world_x = bbox.width_m(origin_lat)
    world_y = bbox.height_m()

    elements = overpass.get("elements", [])
    node_map: dict[int, tuple[float, float]] = {}
    ways: list[dict[str, Any]] = []
    for el in elements:
        et = el.get("type")
        if et == "node":
            node_map[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
        elif et == "way" and "tags" in el and "building" in el["tags"]:
            ways.append(el)

    raw_obstacles: list[dict[str, Any]] = []
    for way in ways:
        nodes = way.get("nodes", [])
        tags = way.get("tags", {})
        coords = [node_map[n] for n in nodes if n in node_map]
        if len(coords) < 3:
            continue

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        x1, y1 = _to_local_xy(min_lat, min_lon, origin_lat, origin_lon)
        x2, y2 = _to_local_xy(max_lat, max_lon, origin_lat, origin_lon)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        # Cull tiny slivers.
        if (x2 - x1) < 5.0 or (y2 - y1) < 5.0:
            continue

        # Keep only objects fully inside bbox -> world.
        if x2 <= 0 or y2 <= 0 or x1 >= world_x or y1 >= world_y:
            continue
        x1 = max(0.0, min(world_x, x1))
        y1 = max(0.0, min(world_y, y1))
        x2 = max(0.0, min(world_x, x2))
        y2 = max(0.0, min(world_y, y2))
        if (x2 - x1) < 5.0 or (y2 - y1) < 5.0:
            continue

        raw_obstacles.append(
            {
                "id": f"O{len(raw_obstacles) + 1}",
                "kind": "building",
                "height": _parse_height(tags, default_height),
                "min_corner": {"x": round(x1, 2), "y": round(y1, 2), "z": 0.0},
                "max_corner": {"x": round(x2, 2), "y": round(y2, 2), "z": 0.0},  # z assigned below
            }
        )

    # Keep largest footprints for runtime speed.
    raw_obstacles.sort(
        key=lambda o: (o["max_corner"]["x"] - o["min_corner"]["x"]) * (o["max_corner"]["y"] - o["min_corner"]["y"]),
        reverse=True,
    )
    kept = raw_obstacles[: max(1, max_buildings)]
    for obs in kept:
        obs["max_corner"]["z"] = round(obs.pop("height"), 2)
    # Renumber after truncation.
    for i, obs in enumerate(kept, start=1):
        obs["id"] = f"O{i}"
    return kept, len(raw_obstacles)


def _point_blocked(x: float, y: float, obstacles: list[dict[str, Any]], pad: float = 20.0) -> bool:
    for o in obstacles:
        mn = o["min_corner"]
        mx = o["max_corner"]
        if (mn["x"] - pad) <= x <= (mx["x"] + pad) and (mn["y"] - pad) <= y <= (mx["y"] + pad):
            return True
    return False


def _pick_point(
    candidates: list[tuple[float, float]],
    world_x: float,
    world_y: float,
    obstacles: list[dict[str, Any]],
    rng: random.Random,
    min_x: float = 80.0,
    min_y: float = 80.0,
    max_tries: int = 400,
) -> tuple[float, float]:
    for x, y in candidates:
        if min_x <= x <= world_x - min_x and min_y <= y <= world_y - min_y and not _point_blocked(x, y, obstacles):
            return x, y
    for _ in range(max_tries):
        x = rng.uniform(min_x, max(min_x + 1.0, world_x - min_x))
        y = rng.uniform(min_y, max(min_y + 1.0, world_y - min_y))
        if not _point_blocked(x, y, obstacles):
            return x, y
    return min_x, min_y


def _build_targets_and_hazards(
    world_x: float,
    world_y: float,
    obstacles: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_candidates = [
        (world_x * 0.08, world_y * 0.08),
        (world_x * 0.14, world_y * 0.10),
        (world_x * 0.10, world_y * 0.18),
    ]
    base_x, base_y = _pick_point(base_candidates, world_x, world_y, obstacles, rng, min_x=40.0, min_y=40.0)
    base = {
        "position": {"x": round(base_x, 2), "y": round(base_y, 2), "z": 0.0},
        "recharge_radius": 90.0,
    }

    target_specs = [
        ("T1", 1.0, 120.0, 35.0, (0.78, 0.20)),
        ("T2", 0.85, 110.0, 45.0, (0.62, 0.76)),
        ("T3", 0.70, 105.0, 30.0, (0.30, 0.62)),
    ]
    targets: list[dict[str, Any]] = []
    for tid, urg, radius, z, frac in target_specs:
        preferred = [(world_x * frac[0], world_y * frac[1])]
        x, y = _pick_point(preferred, world_x, world_y, obstacles, rng)
        targets.append(
            {
                "id": tid,
                "position": {"x": round(x, 2), "y": round(y, 2), "z": z},
                "urgency": urg,
                "delivered": False,
                "delivery_radius": radius,
            }
        )

    hazard_r = max(220.0, min(world_x, world_y) * 0.11)
    hazards = [
        {
            "id": "H1",
            "center": {"x": round(world_x * 0.52, 2), "y": round(world_y * 0.46, 2), "z": 0.0},
            "radius": round(hazard_r, 2),
            "severity": 0.85,
            "height": 90.0,
            "growth_rate": 0.006,
        },
        {
            "id": "H2",
            "center": {"x": round(world_x * 0.70, 2), "y": round(world_y * 0.72, 2), "z": 0.0},
            "radius": round(hazard_r * 0.88, 2),
            "severity": 0.72,
            "height": 70.0,
            "growth_rate": 0.004,
        },
    ]
    return targets, hazards, base


def main() -> None:
    args = _parse_args()
    bbox = BBox(args.south, args.west, args.north, args.east)
    rng = random.Random(args.seed)

    print("Fetching OSM buildings from Overpass...")
    overpass = _fetch_overpass_buildings(bbox, timeout_s=args.timeout_s)

    obstacles, raw_count = _footprints_to_obstacles(
        overpass=overpass,
        bbox=bbox,
        default_height=args.default_height,
        max_buildings=args.max_buildings,
    )
    if not obstacles:
        raise RuntimeError("No building obstacles found for bbox; try a larger area.")

    world_x = round(bbox.width_m(bbox.south), 2)
    world_y = round(bbox.height_m(), 2)
    world_z = float(args.world_z)
    targets, hazards, base_station = _build_targets_and_hazards(world_x, world_y, obstacles, rng)

    out = {
        "origin": {"lat": bbox.south, "lon": bbox.west},
        "bounds": {"x": world_x, "y": world_y, "z": world_z},
        "max_steps": int(args.max_steps),
        "base_station": base_station,
        "targets": targets,
        "hazards": hazards,
        "obstacles": obstacles,
        "cylinders": [],
        "responders": [],
        "metadata": {
            "source": "openstreetmap-overpass",
            "generated_unix_s": int(time.time()),
            "bbox": {
                "south": bbox.south,
                "west": bbox.west,
                "north": bbox.north,
                "east": bbox.east,
            },
            "buildings_raw": raw_count,
            "buildings_kept": len(obstacles),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote world JSON -> {out_path}")
    print(f"world bounds: {world_x:.1f}m x {world_y:.1f}m x {world_z:.1f}m")
    print(f"obstacles: {len(obstacles)} (raw={raw_count})")
    print("Use with:")
    print(f"  python train_ppo.py --world-json {out_path}")
    print(f"  python train_unsloth.py --world-json {out_path}")


if __name__ == "__main__":
    main()
