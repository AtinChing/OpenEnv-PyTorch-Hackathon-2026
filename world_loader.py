"""Helpers for loading custom Varaha worlds from JSON files.

Supports three JSON styles:
1. Trace files with a top-level ``world`` block (``env.get_trace()`` output)
2. ``render_state`` dumps (``env.render_state()`` output)
3. Lightweight scene object exports with a top-level ``objects`` list
"""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Callable

from sim_types import (
    BaseStation,
    CylindricalObstacle,
    DeliveryTarget,
    HazardRegion,
    ObstacleVolume,
    ResponderUnit,
    Vec3,
)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vec3_from_dict(raw: dict[str, Any] | None, default: Vec3 | None = None) -> Vec3:
    if not isinstance(raw, dict):
        return default or Vec3()
    return Vec3(
        _to_float(raw.get("x"), (default.x if default else 0.0)),
        _to_float(raw.get("y"), (default.y if default else 0.0)),
        _to_float(raw.get("z"), (default.z if default else 0.0)),
    )


def _extract_world_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "world" in data and isinstance(data["world"], dict):
        return data["world"]
    if "base_station" in data or "targets" in data:
        return data
    return data


def _parse_base_station(payload: dict[str, Any]) -> BaseStation:
    raw = payload.get("base_station", {})
    return BaseStation(
        position=_vec3_from_dict(raw.get("position")),
        recharge_radius=_to_float(raw.get("recharge_radius"), 80.0),
    )


def _parse_targets(payload: dict[str, Any]) -> list[DeliveryTarget]:
    out: list[DeliveryTarget] = []
    raw_targets = payload.get("targets", [])
    if not isinstance(raw_targets, list):
        return out
    for idx, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            continue
        out.append(
            DeliveryTarget(
                id=str(raw.get("id", f"T{idx + 1}")),
                position=_vec3_from_dict(raw.get("position")),
                urgency=max(0.1, min(1.0, _to_float(raw.get("urgency"), 0.6))),
                delivered=bool(raw.get("delivered", False)),
                delivery_radius=max(5.0, _to_float(raw.get("delivery_radius"), 90.0)),
            )
        )
    return out


def _parse_hazards(payload: dict[str, Any]) -> list[HazardRegion]:
    out: list[HazardRegion] = []
    raw_hazards = payload.get("hazards", [])
    if not isinstance(raw_hazards, list):
        return out
    for idx, raw in enumerate(raw_hazards):
        if not isinstance(raw, dict):
            continue
        out.append(
            HazardRegion(
                id=str(raw.get("id", f"H{idx + 1}")),
                center=_vec3_from_dict(raw.get("center")),
                radius=max(10.0, _to_float(raw.get("radius"), 300.0)),
                severity=max(0.1, min(1.0, _to_float(raw.get("severity"), 0.7))),
                height=max(5.0, _to_float(raw.get("height", raw.get("current_height", 70.0)), 70.0)),
                growth_rate=max(0.0, _to_float(raw.get("growth_rate"), 0.0)),
            )
        )
    return out


def _box_from_center_size(raw: dict[str, Any], idx: int) -> ObstacleVolume | None:
    center = raw.get("center") or raw.get("position")
    center_v = _vec3_from_dict(center)
    size = raw.get("size") or raw.get("dimensions") or {}
    sx = _to_float(size.get("x", size.get("width")), 0.0)
    sy = _to_float(size.get("y", size.get("depth")), 0.0)
    sz = _to_float(size.get("z", size.get("height")), 0.0)
    if sx <= 0 or sy <= 0 or sz <= 0:
        return None
    return ObstacleVolume(
        id=str(raw.get("id", f"O{idx + 1}")),
        kind=str(raw.get("kind", raw.get("type", "building"))),
        min_corner=Vec3(center_v.x - sx / 2, center_v.y - sy / 2, max(0.0, center_v.z - sz / 2)),
        max_corner=Vec3(center_v.x + sx / 2, center_v.y + sy / 2, max(sz, center_v.z + sz / 2)),
    )


def _parse_obstacles(payload: dict[str, Any]) -> list[ObstacleVolume]:
    out: list[ObstacleVolume] = []
    raw_obstacles = payload.get("obstacles", [])
    if isinstance(raw_obstacles, list):
        for idx, raw in enumerate(raw_obstacles):
            if not isinstance(raw, dict):
                continue
            if "min_corner" in raw and "max_corner" in raw:
                out.append(
                    ObstacleVolume(
                        id=str(raw.get("id", f"O{idx + 1}")),
                        kind=str(raw.get("kind", "building")),
                        min_corner=_vec3_from_dict(raw.get("min_corner")),
                        max_corner=_vec3_from_dict(raw.get("max_corner")),
                    )
                )
            else:
                box = _box_from_center_size(raw, idx)
                if box is not None:
                    out.append(box)
    # Optional generic scene objects block (e.g. exported render entities)
    raw_objects = payload.get("objects", [])
    if isinstance(raw_objects, list):
        start_idx = len(out)
        for idx, raw in enumerate(raw_objects):
            if not isinstance(raw, dict):
                continue
            otype = str(raw.get("type", raw.get("kind", ""))).lower()
            if otype in {"box", "building", "obstacle", "cuboid"}:
                box = _box_from_center_size(raw, start_idx + idx)
                if box is not None:
                    out.append(box)
    return out


def _parse_cylinders(payload: dict[str, Any]) -> list[CylindricalObstacle]:
    out: list[CylindricalObstacle] = []
    raw_cyls = payload.get("cylinders", [])
    if isinstance(raw_cyls, list):
        for idx, raw in enumerate(raw_cyls):
            if not isinstance(raw, dict):
                continue
            out.append(
                CylindricalObstacle(
                    id=str(raw.get("id", f"C{idx + 1}")),
                    kind=str(raw.get("kind", "tree")),
                    center=_vec3_from_dict(raw.get("center")),
                    radius=max(1.0, _to_float(raw.get("radius"), 10.0)),
                    height=max(1.0, _to_float(raw.get("height"), 50.0)),
                )
            )
    raw_objects = payload.get("objects", [])
    if isinstance(raw_objects, list):
        start_idx = len(out)
        for idx, raw in enumerate(raw_objects):
            if not isinstance(raw, dict):
                continue
            otype = str(raw.get("type", raw.get("kind", ""))).lower()
            if otype not in {"cylinder", "tree", "pole", "pillar"}:
                continue
            out.append(
                CylindricalObstacle(
                    id=str(raw.get("id", f"C{start_idx + idx + 1}")),
                    kind=str(raw.get("kind", otype or "tree")),
                    center=_vec3_from_dict(raw.get("center") or raw.get("position")),
                    radius=max(1.0, _to_float(raw.get("radius"), 8.0)),
                    height=max(1.0, _to_float(raw.get("height"), 40.0)),
                )
            )
    return out


def _parse_responders(payload: dict[str, Any]) -> list[ResponderUnit]:
    out: list[ResponderUnit] = []
    raw_responders = payload.get("responders", [])
    if not isinstance(raw_responders, list):
        return out
    for idx, raw in enumerate(raw_responders):
        if not isinstance(raw, dict):
            continue
        out.append(
            ResponderUnit(
                id=str(raw.get("id", f"R{idx + 1}")),
                position=_vec3_from_dict(raw.get("position")),
                linked_target_id=str(raw.get("linked_target_id", "")),
                status=str(raw.get("status", "stable")),
                current_need=str(raw.get("current_need", "supplies")),
                message=str(raw.get("message", "")),
                can_update_dropzone=bool(raw.get("can_update_dropzone", False)),
                active=bool(raw.get("active", True)),
                latest_intel=str(raw.get("latest_intel", "none")),
                intel_severity=max(0.0, min(1.0, _to_float(raw.get("intel_severity"), 0.0))),
            )
        )
    return out


def load_world_from_json(path: str | Path) -> dict[str, Any]:
    """Load a custom world JSON into canonical env objects.

    Returns a dict with keys: ``base``, ``targets``, ``hazards``, ``obstacles``,
    ``cylinders``, ``responders``, ``max_steps``.
    """
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    payload = _extract_world_payload(raw if isinstance(raw, dict) else {})

    return {
        "base": _parse_base_station(payload),
        "targets": _parse_targets(payload),
        "hazards": _parse_hazards(payload),
        "obstacles": _parse_obstacles(payload),
        "cylinders": _parse_cylinders(payload),
        "responders": _parse_responders(payload),
        "max_steps": int(_to_float(payload.get("max_steps"), 0.0)),
        "source_path": str(p),
    }


def world_fn_from_json(path: str | Path) -> Callable[[Any], None]:
    """Create a ``world_fn`` callback usable by ``VarahaEnv`` constructors."""
    world = load_world_from_json(path)

    def _builder(env: Any) -> None:
        env.base = copy.deepcopy(world["base"])
        env.targets = copy.deepcopy(world["targets"])
        env.hazards = copy.deepcopy(world["hazards"])
        env.obstacles = copy.deepcopy(world["obstacles"])
        env.cylinders = copy.deepcopy(world["cylinders"])
        env.responders = copy.deepcopy(world["responders"])
        if world["max_steps"] > 0:
            env.cfg.max_episode_steps = world["max_steps"]

    return _builder
