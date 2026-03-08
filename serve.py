"""Varaha visualizer server.

Serves static files and provides an API to run simulations on demand.

    POST /api/run
    Body: {"method": "heuristic"|"random", "seed": int|null, "max_steps": int}
    Returns: trace JSON
"""

import json
import random
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any

from sim_types import Vec3
from varaha_env import VarahaEnv
from czml_converter import trace_to_czml

MODEL_ROOT = Path("assets/models")
MODEL_EXTS = {".glb", ".gltf"}


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    - Ignores blank lines and comments.
    - Does not override existing environment variables.
    """
    if not os.path.isfile(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                if not key:
                    continue
                # Strip optional single/double quotes around value.
                if len(value) >= 2 and (
                    (value[0] == '"' and value[-1] == '"')
                    or (value[0] == "'" and value[-1] == "'")
                ):
                    value = value[1:-1]
                existing = os.environ.get(key)
                if existing is None or str(existing).strip() == "":
                    os.environ[key] = value
    except OSError:
        # Non-fatal: server can continue without .env
        return


def _to_web_path(path: Path) -> str:
    return "/" + path.as_posix().lstrip("./")


def _find_model_candidate(subdir: str) -> str:
    """Find the first model in assets/models/<subdir> recursively."""
    root = MODEL_ROOT / subdir
    if not root.is_dir():
        return ""

    candidates = sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in MODEL_EXTS
        ),
        key=lambda p: p.as_posix().lower(),
    )
    if not candidates:
        return ""
    return _to_web_path(candidates[0])


def _model_uri(env_key: str, subdir: str, fallback: str = "") -> str:
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val

    discovered = _find_model_candidate(subdir)
    if discovered:
        return discovered

    return fallback.strip()


def _client_config() -> dict[str, str]:
    """Client-facing runtime config sourced from environment."""
    return {
        "googleMapsApiKey": os.environ.get("GOOGLE_MAPS_API_KEY", "").strip(),
        "ionToken": os.environ.get("CESIUM_ION_TOKEN", "").strip(),
        "originLat": os.environ.get("SIM_ORIGIN_LAT", "").strip(),
        "originLon": os.environ.get("SIM_ORIGIN_LON", "").strip(),
        "droneModelUri": _model_uri(
            "DRONE_MODEL_URI", "drone", "/assets/models/drone/default_drone.glb"
        ),
        "fireModelUri": _model_uri("FIRE_MODEL_URI", "fire"),
        "buildingModelUri": _model_uri("BUILDING_MODEL_URI", "building"),
        "parkModelUri": _model_uri("PARK_MODEL_URI", "park"),
        "responderModelUri": _model_uri("RESPONDER_MODEL_URI", "responder"),
    }


def _heuristic_action(obs: dict[str, Any], env: VarahaEnv) -> dict[str, Any]:
    drone_pos = Vec3(**obs["drone_position"])
    drone_vel = Vec3(**obs["drone_velocity"])
    undelivered = [t for t in obs["targets"] if not t["delivered"]]

    if undelivered:
        nearest = min(undelivered, key=lambda t: Vec3(**t["relative_position"]).norm())
        rel = Vec3(**nearest["relative_position"])
        in_range = rel.norm() <= 15.0
        goal_dir = rel
    else:
        goal_dir = env.base.position - drone_pos
        in_range = False

    kp, kd = 0.3, 0.8
    accel = goal_dir.scale(kp) - drone_vel.scale(kd)
    accel = accel.clamp_magnitude(env.cfg.max_acceleration)
    near_base = drone_pos.distance_to(env.base.position) <= env.base.recharge_radius

    return {
        "ax": accel.x,
        "ay": accel.y,
        "az": accel.z,
        "deliver": in_range,
        "recharge": near_base and obs["battery"] < 40,
    }


def run_simulation(method: str, seed: int | None = None, max_steps: int = 500) -> dict:
    env = VarahaEnv()
    obs = env.reset(seed=seed)

    for _ in range(max_steps):
        if method == "random":
            action = {
                "ax": random.uniform(-5, 5),
                "ay": random.uniform(-5, 5),
                "az": random.uniform(-2, 2),
                "deliver": random.random() < 0.1,
                "recharge": random.random() < 0.05,
            }
        else:
            action = _heuristic_action(obs, env)

        obs, reward, done, info = env.step(action)
        if done:
            break

    return env.get_trace()


def _find_trace_jsons():
    """Scan for JSON traces.

    - Legacy: top-level JSON files with "trace" in filename.
    - Recursive: results/, results_hardcore/, results_hardcore_run1/, results_hardcore_v2/
    """
    traces = []

    # Legacy top-level trace discovery.
    for f in sorted(os.listdir(".")):
        if os.path.isfile(f) and f.lower().endswith(".json") and "trace" in f.lower():
            traces.append(f)

    # Recursive discovery in results* directories.
    for name in sorted(os.listdir(".")):
        if not os.path.isdir(name) or not name.startswith("results"):
            continue
        for root, _, files in os.walk(name):
            for f in sorted(files):
                if not f.lower().endswith(".json"):
                    continue
                rel = os.path.join(root, f).lstrip("./")
                traces.append(rel)

    return sorted(set(traces))


class VarahaHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/traces":
            traces = _find_trace_jsons()
            payload = json.dumps(traces).encode()
            self._json_response(payload)
        elif self.path == "/api/client-config":
            # Re-read .env on each config request so browser refresh picks up changes
            # without requiring a server restart.
            _load_dotenv(".env")
            payload = json.dumps(_client_config()).encode()
            self._json_response(payload)
        elif self.path.startswith("/api/czml/"):
            trace_name = self.path[len("/api/czml/"):]
            self._serve_czml(trace_name)
        else:
            super().do_GET()

    def _json_response(self, payload: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_czml(self, trace_name: str):
        if not trace_name or ".." in trace_name:
            self.send_error(400, "Invalid trace name")
            return
        if not os.path.isfile(trace_name):
            self.send_error(404, f"Trace not found: {trace_name}")
            return
        try:
            with open(trace_name) as f:
                data = json.load(f)
            czml = trace_to_czml(data)
            payload = json.dumps(czml).encode()
            self._json_response(payload)
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_POST(self):
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            method = body.get("method", "heuristic")
            seed = body.get("seed")
            max_steps = body.get("max_steps", 500)

            trace = run_simulation(method, seed, max_steps)
            payload = json.dumps(trace).encode()
            self._json_response(payload)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if "/api/" in str(args[0]):
            super().log_message(format, *args)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _load_dotenv(".env")
    port = 9090
    server = ReusableHTTPServer(("", port), VarahaHandler)
    print(f"Varaha server running at http://localhost:{port}/visualizer.html")
    print(f"CesiumJS 3D viewer at  http://localhost:{port}/cesium_viewer.html")
    server.serve_forever()
