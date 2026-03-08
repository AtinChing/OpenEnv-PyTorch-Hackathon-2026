"""Varaha visualizer server.

Serves static files and provides an API to run simulations on demand.

    POST /api/run
    Body: {"method": "heuristic"|"random", "seed": int|null, "max_steps": int}
    Returns: trace JSON
"""

import json
import random
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any

from sim_types import Vec3
from varaha_env import VarahaEnv


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

    - Keep legacy behavior: include top-level project JSON files with "trace" in filename.
    - Include every JSON file under results/ recursively.
    """
    traces = []

    # Legacy top-level trace discovery.
    for f in sorted(os.listdir(".")):
        if os.path.isfile(f) and f.lower().endswith(".json") and "trace" in f.lower():
            traces.append(f)

    # Recursive results/** discovery.
    if os.path.isdir("results"):
        for root, _, files in os.walk("results"):
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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            method = body.get("method", "heuristic")
            seed = body.get("seed")
            max_steps = body.get("max_steps", 500)

            trace = run_simulation(method, seed, max_steps)
            payload = json.dumps(trace).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
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
    port = 9090
    server = ReusableHTTPServer(("", port), VarahaHandler)
    print(f"Varaha server running at http://localhost:{port}/visualizer.html")
    server.serve_forever()
