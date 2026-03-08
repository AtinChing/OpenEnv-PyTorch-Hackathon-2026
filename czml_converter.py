"""Convert a Varaha trace JSON file into CZML for CesiumJS playback."""

from __future__ import annotations

import json
import math
import sys
from typing import Any


# ── coordinate conversion (mirrors VarahaEnv.local_to_latlon) ──────────

ORIGIN_LAT = 38.55
ORIGIN_LON = -121.47
METERS_PER_DEG_LAT = 111_320.0
METERS_PER_DEG_LON = METERS_PER_DEG_LAT * math.cos(math.radians(ORIGIN_LAT))


def _to_latlon(pos: dict[str, float]) -> tuple[float, float, float]:
    lat = ORIGIN_LAT + pos["y"] / METERS_PER_DEG_LAT
    lon = ORIGIN_LON + pos["x"] / METERS_PER_DEG_LON
    alt = pos.get("z", 0.0)
    return (round(lon, 7), round(lat, 7), round(alt, 2))


def _iso_time(step: int, dt: float = 0.5) -> str:
    """Step index → ISO 8601 timestamp relative to an arbitrary epoch."""
    t = step * dt
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"2026-01-01T{h:02d}:{m:02d}:{s:05.2f}Z"


# ── CZML packet builders ──────────────────────────────────────────────

def _document_packet(total_steps: int, dt: float) -> dict[str, Any]:
    start = _iso_time(0, dt)
    end = _iso_time(total_steps, dt)
    return {
        "id": "document",
        "name": "Varaha Wildfire Logistics",
        "version": "1.0",
        "clock": {
            "interval": f"{start}/{end}",
            "currentTime": start,
            "multiplier": 20,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }


def _drone_packet(
    trace: list[dict[str, Any]], total_steps: int, dt: float
) -> dict[str, Any]:
    start = _iso_time(0, dt)
    end = _iso_time(total_steps, dt)

    cartographic: list[float] = []
    for pt in trace:
        seconds = pt["step"] * dt
        lon, lat, alt = _to_latlon(pt["position"])
        cartographic.extend([seconds, lon, lat, alt])

    return {
        "id": "drone",
        "name": "Delivery Drone",
        "availability": f"{start}/{end}",
        "position": {
            "epoch": start,
            "cartographicDegrees": cartographic,
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 1,
        },
        "point": {
            "color": {"rgba": [0, 180, 255, 255]},
            "pixelSize": 10,
            "heightReference": "NONE",
        },
        "path": {
            "material": {
                "solidColor": {"color": {"rgba": [0, 180, 255, 180]}}
            },
            "width": 2,
            "leadTime": 0,
            "trailTime": total_steps * dt,
        },
        "label": {
            "text": "Drone",
            "font": "12pt sans-serif",
            "fillColor": {"rgba": [255, 255, 255, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 200]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -14]},
        },
    }


def _base_packet(base: dict[str, Any]) -> dict[str, Any]:
    pos = base["position"]
    lon, lat, alt = _to_latlon(pos)
    radius = base.get("recharge_radius", 80.0)
    return {
        "id": "base_station",
        "name": "Base Station",
        "position": {"cartographicDegrees": [lon, lat, alt]},
        "point": {
            "color": {"rgba": [0, 200, 0, 255]},
            "pixelSize": 14,
            "outlineColor": {"rgba": [255, 255, 255, 255]},
            "outlineWidth": 2,
        },
        "ellipse": {
            "semiMajorAxis": radius,
            "semiMinorAxis": radius,
            "height": 0,
            "material": {"solidColor": {"color": {"rgba": [0, 200, 0, 40]}}},
            "outline": True,
            "outlineColor": {"rgba": [0, 200, 0, 120]},
        },
        "label": {
            "text": "Base",
            "font": "13pt sans-serif",
            "fillColor": {"rgba": [0, 230, 0, 255]},
            "style": "FILL",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -16]},
        },
    }


def _target_packet(target: dict[str, Any]) -> dict[str, Any]:
    tid = target["id"]
    pos = target["position"]
    lon, lat, alt = _to_latlon(pos)
    delivered = target.get("delivered", False)
    urgency = target.get("urgency", 0.5)
    radius = target.get("delivery_radius", 80.0)

    r = int(255 * urgency)
    g = int(100 * (1 - urgency))
    color = [r, g, 40, 255] if not delivered else [80, 200, 80, 200]
    zone_color = [r, g, 40, 30] if not delivered else [80, 200, 80, 25]

    return {
        "id": f"target_{tid}",
        "name": f"Target {tid} (urg={urgency:.1f})",
        "position": {"cartographicDegrees": [lon, lat, alt]},
        "point": {
            "color": {"rgba": color},
            "pixelSize": 12,
            "outlineColor": {"rgba": [255, 255, 255, 200]},
            "outlineWidth": 1,
        },
        "ellipse": {
            "semiMajorAxis": radius,
            "semiMinorAxis": radius,
            "height": 0,
            "extrudedHeight": alt,
            "material": {"solidColor": {"color": {"rgba": zone_color}}},
            "outline": True,
            "outlineColor": {"rgba": color[:3] + [100]},
        },
        "label": {
            "text": f"{tid}",
            "font": "12pt sans-serif",
            "fillColor": {"rgba": color},
            "style": "FILL",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -14]},
        },
    }


def _hazard_packet(hazard: dict[str, Any]) -> dict[str, Any]:
    hid = hazard["id"]
    center = hazard["center"]
    lon, lat, _ = _to_latlon(center)
    radius = hazard.get("radius", 200.0)
    height = hazard.get("current_height", hazard.get("height", 80.0))
    severity = hazard.get("severity", 0.5)

    r = int(200 + 55 * severity)
    alpha = int(30 + 40 * severity)

    return {
        "id": f"hazard_{hid}",
        "name": f"Hazard {hid} (sev={severity:.1f})",
        "position": {"cartographicDegrees": [lon, lat, 0]},
        "cylinder": {
            "length": height,
            "topRadius": radius * 0.4,
            "bottomRadius": radius,
            "material": {"solidColor": {"color": {"rgba": [r, 60, 0, alpha]}}},
            "outline": True,
            "outlineColor": {"rgba": [r, 60, 0, 100]},
            "numberOfVerticalLines": 0,
        },
        "label": {
            "text": f"{hid}",
            "font": "12pt sans-serif",
            "fillColor": {"rgba": [255, 100, 0, 255]},
            "style": "FILL",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -14]},
        },
    }


def _responder_packet(responder: dict[str, Any]) -> dict[str, Any]:
    rid = responder["id"]
    pos = responder["position"]
    lon, lat, alt = _to_latlon(pos)
    status = responder.get("status", "stable")
    linked = responder.get("linked_target_id", "")
    color = [255, 200, 100, 255] if status == "critical" else [255, 180, 80, 255] if status == "urgent" else [200, 200, 150, 255]
    return {
        "id": f"responder_{rid}",
        "name": f"Responder {rid} ({status}) → {linked}",
        "position": {"cartographicDegrees": [lon, lat, alt]},
        "point": {
            "color": {"rgba": color},
            "pixelSize": 10,
            "outlineColor": {"rgba": [255, 255, 255, 200]},
            "outlineWidth": 1,
        },
        "label": {
            "text": f"R{rid[-1]}",
            "font": "11pt sans-serif",
            "fillColor": {"rgba": color},
            "style": "FILL",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -12]},
        },
    }


def _cylinder_packet(cylinder: dict[str, Any]) -> dict[str, Any]:
    cid = cylinder["id"]
    center = cylinder["center"]
    lon, lat, _ = _to_latlon(center)
    radius = cylinder.get("radius", 15.0)
    height = cylinder.get("height", 50.0)
    kind = cylinder.get("kind", "tree")
    return {
        "id": f"cylinder_{cid}",
        "name": f"{kind} {cid}",
        "position": {"cartographicDegrees": [lon, lat, 0]},
        "cylinder": {
            "length": height,
            "topRadius": radius,
            "bottomRadius": radius,
            "material": {"solidColor": {"color": {"rgba": [80, 120, 60, 80]}}},
            "outline": True,
            "outlineColor": {"rgba": [60, 100, 40, 150]},
        },
    }


def _obstacle_packet(obstacle: dict[str, Any]) -> dict[str, Any]:
    oid = obstacle["id"]
    mn = obstacle["min_corner"]
    mx = obstacle["max_corner"]
    cx = (mn["x"] + mx["x"]) / 2
    cy = (mn["y"] + mx["y"]) / 2
    cz = 0.0
    height = mx["z"] - mn.get("z", 0.0)
    half_x = (mx["x"] - mn["x"]) / 2
    half_y = (mx["y"] - mn["y"]) / 2
    lon, lat, _ = _to_latlon({"x": cx, "y": cy, "z": cz})

    corners_local = [
        {"x": mn["x"], "y": mn["y"], "z": 0},
        {"x": mx["x"], "y": mn["y"], "z": 0},
        {"x": mx["x"], "y": mx["y"], "z": 0},
        {"x": mn["x"], "y": mx["y"], "z": 0},
    ]
    positions: list[float] = []
    for c in corners_local:
        clon, clat, _ = _to_latlon(c)
        positions.extend([clon, clat, 0.0])

    return {
        "id": f"obstacle_{oid}",
        "name": f"Obstacle {oid}",
        "polygon": {
            "positions": {"cartographicDegrees": positions},
            "height": 0,
            "extrudedHeight": height,
            "material": {"solidColor": {"color": {"rgba": [120, 120, 120, 60]}}},
            "outline": True,
            "outlineColor": {"rgba": [180, 180, 180, 150]},
        },
    }


def _delivery_event_packets(
    trace: list[dict[str, Any]], dt: float
) -> list[dict[str, Any]]:
    """Billboard markers at each delivery location/time."""
    packets: list[dict[str, Any]] = []
    for pt in trace:
        for evt in pt.get("events", []):
            if not evt.startswith("delivered_"):
                continue
            tid = evt.replace("delivered_", "")
            lon, lat, alt = _to_latlon(pt["position"])
            t = _iso_time(pt["step"], dt)
            packets.append({
                "id": f"delivery_event_{tid}_step{pt['step']}",
                "name": f"Delivered {tid}",
                "position": {"cartographicDegrees": [lon, lat, alt]},
                "point": {
                    "color": {"rgba": [255, 220, 0, 255]},
                    "pixelSize": 16,
                    "outlineColor": {"rgba": [255, 255, 255, 255]},
                    "outlineWidth": 2,
                },
                "label": {
                    "text": f"Delivered {tid}",
                    "font": "13pt sans-serif",
                    "fillColor": {"rgba": [255, 220, 0, 255]},
                    "outlineColor": {"rgba": [0, 0, 0, 200]},
                    "outlineWidth": 2,
                    "style": "FILL_AND_OUTLINE",
                    "verticalOrigin": "BOTTOM",
                    "pixelOffset": {"cartesian2": [0, -20]},
                    "showBackground": True,
                    "backgroundColor": {"rgba": [0, 0, 0, 150]},
                },
            })
    return packets


# ── main conversion ───────────────────────────────────────────────────

def trace_to_czml(trace_json: dict[str, Any], dt: float = 0.5) -> list[dict[str, Any]]:
    """Convert a full Varaha trace dict to a list of CZML packets."""
    world = trace_json["world"]
    trace = trace_json["trace"]
    summary = trace_json.get("summary", {})
    total_steps = summary.get("total_steps", len(trace))

    packets: list[dict[str, Any]] = [_document_packet(total_steps, dt)]
    packets.append(_drone_packet(trace, total_steps, dt))
    packets.append(_base_packet(world["base_station"]))

    for t in world.get("targets", []):
        packets.append(_target_packet(t))

    for h in world.get("hazards", []):
        packets.append(_hazard_packet(h))

    for o in world.get("obstacles", []):
        packets.append(_obstacle_packet(o))

    for c in world.get("cylinders", []):
        packets.append(_cylinder_packet(c))

    for r in world.get("responders", []):
        packets.append(_responder_packet(r))

    packets.extend(_delivery_event_packets(trace, dt))

    return packets


def convert_file(input_path: str, output_path: str | None = None) -> str:
    """Read a trace JSON, convert, and write CZML. Returns output path."""
    with open(input_path) as f:
        data = json.load(f)
    czml = trace_to_czml(data)
    if output_path is None:
        output_path = input_path.rsplit(".", 1)[0] + ".czml"
    with open(output_path, "w") as f:
        json.dump(czml, f)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python czml_converter.py <trace.json> [output.czml]")
        sys.exit(1)
    out = convert_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"Wrote {out}")
