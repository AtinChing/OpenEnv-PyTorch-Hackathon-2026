#!/usr/bin/env python3
"""Visualize Varaha flight traces in 2D (matplotlib) or 3D (plotly).

Usage:
    python visualize_trace.py results/trace_trained_0.json              # 3D (default)
    python visualize_trace.py results/trace_trained_0.json --mode 2d    # 2D
    python visualize_trace.py results/trace_trained_0.json results/trace_untrained_0.json  # overlay
"""

import argparse
import json
import os
import sys

import numpy as np


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# 3D visualisation (Plotly)
# -----------------------------------------------------------------------

def _box_mesh(ob, color="rgba(120,125,140,0.35)"):
    """Create a Mesh3d trace for an axis-aligned box obstacle."""
    import plotly.graph_objects as go

    mc = ob["min_corner"]
    xc = ob["max_corner"]
    x0, y0, z0 = mc["x"], mc["y"], mc["z"]
    x1, y1, z1 = xc["x"], xc["y"], xc["z"]

    verts_x = [x0, x1, x1, x0, x0, x1, x1, x0]
    verts_y = [y0, y0, y1, y1, y0, y0, y1, y1]
    verts_z = [z0, z0, z0, z0, z1, z1, z1, z1]

    # 12 triangles covering 6 faces
    tri_i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    tri_j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    tri_k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]

    return go.Mesh3d(
        x=verts_x, y=verts_y, z=verts_z,
        i=tri_i, j=tri_j, k=tri_k,
        color=color, opacity=0.3,
        flatshading=True,
        name=ob["id"],
        hoverinfo="name",
    )


def _sphere_surface(hz, n=24):
    """Create a Surface trace for a hazard sphere."""
    import plotly.graph_objects as go

    cx, cy, cz = hz["center"]["x"], hz["center"]["y"], hz["center"]["z"]
    r = hz["radius"]
    sev = hz["severity"]

    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(theta), np.sin(phi))
    y = cy + r * np.outer(np.sin(theta), np.sin(phi))
    z = cz + r * np.outer(np.ones(n), np.cos(phi))

    red = int(200 + 55 * sev)
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, f"rgba({red},60,30,0.08)"], [1, f"rgba({red},60,30,0.12)"]],
        showscale=False,
        name=f'{hz["id"]} (sev={sev})',
        hoverinfo="name",
    )


def render_3d(traces_data: list[tuple[str, dict]], out_path: str = "trace_3d.html"):
    """Render one or more traces into an interactive 3D Plotly HTML file."""
    import plotly.graph_objects as go

    world = traces_data[0][1]["world"]
    bounds = world["bounds"]

    fig = go.Figure()

    # --- World elements (from first trace's world) ---

    # Obstacles
    for ob in world["obstacles"]:
        fig.add_trace(_box_mesh(ob))
        mc, xc = ob["min_corner"], ob["max_corner"]
        fig.add_trace(go.Scatter3d(
            x=[(mc["x"]+xc["x"])/2], y=[(mc["y"]+xc["y"])/2], z=[xc["z"]+10],
            mode="text", text=[ob["id"]], textfont=dict(size=10, color="rgba(180,185,200,0.7)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Hazards
    for hz in world["hazards"]:
        fig.add_trace(_sphere_surface(hz))

    # Base station
    bs = world["base_station"]
    fig.add_trace(go.Scatter3d(
        x=[bs["position"]["x"]], y=[bs["position"]["y"]], z=[bs["position"]["z"]],
        mode="markers+text",
        marker=dict(size=8, color="dodgerblue", symbol="diamond"),
        text=["BASE"], textposition="top center",
        textfont=dict(size=10, color="dodgerblue"),
        name="Base", hoverinfo="name",
    ))
    # Base radius ring
    theta = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(go.Scatter3d(
        x=bs["position"]["x"] + bs["recharge_radius"] * np.cos(theta),
        y=bs["position"]["y"] + bs["recharge_radius"] * np.sin(theta),
        z=np.full(60, bs["position"]["z"]),
        mode="lines", line=dict(color="dodgerblue", width=2, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    # Targets
    for tgt in world["targets"]:
        color = "orange"
        fig.add_trace(go.Scatter3d(
            x=[tgt["position"]["x"]], y=[tgt["position"]["y"]], z=[tgt["position"]["z"]],
            mode="markers+text",
            marker=dict(size=7, color=color, symbol="cross"),
            text=[f'{tgt["id"]} (u={tgt["urgency"]})'], textposition="top center",
            textfont=dict(size=9, color=color),
            name=tgt["id"], hoverinfo="name+text",
        ))
        # Delivery radius ring
        fig.add_trace(go.Scatter3d(
            x=tgt["position"]["x"] + tgt["delivery_radius"] * np.cos(theta),
            y=tgt["position"]["y"] + tgt["delivery_radius"] * np.sin(theta),
            z=np.full(60, tgt["position"]["z"]),
            mode="lines", line=dict(color=color, width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

    # Ground grid
    for g in range(0, int(bounds["x"]) + 1, 500):
        fig.add_trace(go.Scatter3d(
            x=[g, g], y=[0, bounds["y"]], z=[0, 0],
            mode="lines", line=dict(color="rgba(60,65,80,0.3)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    for g in range(0, int(bounds["y"]) + 1, 500):
        fig.add_trace(go.Scatter3d(
            x=[0, bounds["x"]], y=[g, g], z=[0, 0],
            mode="lines", line=dict(color="rgba(60,65,80,0.3)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # --- Flight paths ---
    path_colors = [
        "rgba(74,230,138,{})",   # green
        "rgba(88,166,244,{})",   # blue
        "rgba(244,163,88,{})",   # orange
        "rgba(200,130,240,{})",  # purple
    ]

    for idx, (label, data) in enumerate(traces_data):
        trace = data["trace"]
        summary = data["summary"]
        n = len(trace)
        color_tpl = path_colors[idx % len(path_colors)]

        xs = [p["position"]["x"] for p in trace]
        ys = [p["position"]["y"] for p in trace]
        zs = [p["position"]["z"] for p in trace]
        batteries = [p["battery"] for p in trace]

        # Path line (with fading alpha)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color_tpl.format("0.6"), width=3),
            name=f"{label}  (r={summary['cumulative_reward']:.0f})",
            hovertemplate=(
                "step %{customdata[0]}<br>"
                "pos (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>"
                "bat %{customdata[1]:.1f}<extra></extra>"
            ),
            customdata=list(zip(
                [p["step"] for p in trace],
                batteries,
            )),
        ))

        # Battery-colored markers (sampled to avoid clutter)
        sample = max(1, n // 80)
        sampled = list(range(0, n, sample))
        if (n - 1) not in sampled:
            sampled.append(n - 1)

        fig.add_trace(go.Scatter3d(
            x=[xs[i] for i in sampled],
            y=[ys[i] for i in sampled],
            z=[zs[i] for i in sampled],
            mode="markers",
            marker=dict(
                size=3,
                color=[batteries[i] for i in sampled],
                colorscale=[[0, "red"], [0.3, "orange"], [1, "limegreen"]],
                cmin=0, cmax=data["world"]["bounds"].get("battery_capacity", 300),
                showscale=(idx == 0),
                colorbar=dict(title="Battery", len=0.4, y=0.8) if idx == 0 else None,
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Event markers
        for p in trace:
            px, py, pz = p["position"]["x"], p["position"]["y"], p["position"]["z"]
            for ev in p.get("events", []):
                if ev == "reset":
                    continue
                if ev.startswith("delivered"):
                    fig.add_trace(go.Scatter3d(
                        x=[px], y=[py], z=[pz],
                        mode="markers",
                        marker=dict(size=8, color="limegreen", symbol="circle",
                                    line=dict(width=2, color="white")),
                        name=ev, showlegend=False,
                        hovertext=f"[step {p['step']}] {ev}",
                        hoverinfo="text",
                    ))
                elif ev == "collision":
                    fig.add_trace(go.Scatter3d(
                        x=[px], y=[py], z=[pz],
                        mode="markers",
                        marker=dict(size=10, color="red", symbol="x",
                                    line=dict(width=3, color="red")),
                        name="collision", showlegend=False,
                        hovertext=f"[step {p['step']}] COLLISION",
                        hoverinfo="text",
                    ))
                elif ev == "success":
                    fig.add_trace(go.Scatter3d(
                        x=[px], y=[py], z=[pz],
                        mode="markers",
                        marker=dict(size=10, color="dodgerblue", symbol="circle",
                                    line=dict(width=3, color="white")),
                        name="success", showlegend=False,
                        hovertext=f"[step {p['step']}] SUCCESS",
                        hoverinfo="text",
                    ))

        # Start marker
        fig.add_trace(go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[zs[0]],
            mode="markers",
            marker=dict(size=6, color="white", symbol="diamond",
                        line=dict(width=1, color="black")),
            showlegend=False, hovertext="START", hoverinfo="text",
        ))

    # --- Layout ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", range=[-100, bounds["x"]+100],
                       backgroundcolor="rgb(12,14,19)", gridcolor="rgba(60,65,80,0.3)"),
            yaxis=dict(title="y (m)", range=[-100, bounds["y"]+100],
                       backgroundcolor="rgb(12,14,19)", gridcolor="rgba(60,65,80,0.3)"),
            zaxis=dict(title="z (m)", range=[0, bounds["z"]+20],
                       backgroundcolor="rgb(12,14,19)", gridcolor="rgba(60,65,80,0.3)"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=bounds["z"]/bounds["x"]*3),
            bgcolor="rgb(12,14,19)",
        ),
        paper_bgcolor="rgb(12,14,19)",
        plot_bgcolor="rgb(12,14,19)",
        font=dict(color="rgb(200,205,216)"),
        title=dict(
            text="Varaha — 3D Flight Trace",
            font=dict(size=16, color="rgb(74,230,138)"),
        ),
        legend=dict(
            bgcolor="rgba(20,23,30,0.8)",
            bordercolor="rgba(30,35,48,0.8)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  3D visualisation saved → {out_path}")
    return fig


# -----------------------------------------------------------------------
# 2D visualisation (Matplotlib) — lightweight fallback
# -----------------------------------------------------------------------

def render_2d(traces_data: list[tuple[str, dict]], out_path: str = "trace_2d.png"):
    """Render one or more traces on a 2D top-down matplotlib plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    world = traces_data[0][1]["world"]
    bounds = world["bounds"]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Hazards
    for hz in world["hazards"]:
        circle = plt.Circle(
            (hz["center"]["x"], hz["center"]["y"]), hz["radius"],
            facecolor=(1, 0.3, 0.15, 0.12), edgecolor=(1, 0.3, 0.15, 0.4),
            linewidth=1.5, linestyle="--",
        )
        ax.add_patch(circle)
        ax.text(hz["center"]["x"], hz["center"]["y"],
                f'{hz["id"]}\nsev={hz["severity"]}',
                ha="center", va="center", fontsize=7, color=(1, 0.3, 0.15, 0.7))

    # Obstacles
    for ob in world["obstacles"]:
        w = ob["max_corner"]["x"] - ob["min_corner"]["x"]
        h = ob["max_corner"]["y"] - ob["min_corner"]["y"]
        rect = mpatches.FancyBboxPatch(
            (ob["min_corner"]["x"], ob["min_corner"]["y"]), w, h,
            boxstyle="round,pad=0", facecolor=(0.5, 0.5, 0.55, 0.25),
            edgecolor=(0.5, 0.5, 0.55, 0.6), linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(ob["min_corner"]["x"] + w/2, ob["min_corner"]["y"] + h/2,
                ob["id"], ha="center", va="center", fontsize=8, color="0.5")

    # Base
    bs = world["base_station"]
    ax.plot(bs["position"]["x"], bs["position"]["y"], "s",
            color="dodgerblue", markersize=9, zorder=5)
    ax.annotate("BASE", (bs["position"]["x"], bs["position"]["y"]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=8, fontweight="bold", color="dodgerblue")

    # Targets
    for tgt in world["targets"]:
        ax.plot(tgt["position"]["x"], tgt["position"]["y"], "^",
                color="orange", markersize=9, zorder=5)
        ax.annotate(f'{tgt["id"]} (u={tgt["urgency"]})',
                    (tgt["position"]["x"], tgt["position"]["y"]),
                    textcoords="offset points", xytext=(12, -10),
                    fontsize=7, color="orange")

    # Paths
    colors_2d = ["limegreen", "dodgerblue", "orange", "mediumpurple"]
    for idx, (label, data) in enumerate(traces_data):
        trace = data["trace"]
        summary = data["summary"]
        xs = [p["position"]["x"] for p in trace]
        ys = [p["position"]["y"] for p in trace]
        c = colors_2d[idx % len(colors_2d)]
        ax.plot(xs, ys, color=c, linewidth=1.2, alpha=0.7,
                label=f'{label}  (r={summary["cumulative_reward"]:.0f})')
        # Start
        ax.plot(xs[0], ys[0], "D", color="white", markersize=5,
                markeredgecolor="black", markeredgewidth=1, zorder=7)
        # Events
        for p in trace:
            px, py = p["position"]["x"], p["position"]["y"]
            for ev in p.get("events", []):
                if ev.startswith("delivered"):
                    ax.plot(px, py, "o", color="limegreen", markersize=6, zorder=6)
                elif ev == "collision":
                    ax.plot(px, py, "x", color="red", markersize=10,
                            markeredgewidth=2.5, zorder=6)

    ax.set_xlim(-150, bounds["x"] + 150)
    ax.set_ylim(-150, bounds["y"] + 150)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Varaha — Flight Trace (top-down)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  2D visualisation saved → {out_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Varaha traces in 2D or 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python visualize_trace.py results/trace_trained_0.json\n"
               "  python visualize_trace.py trace_a.json trace_b.json --mode 2d\n"
               "  python visualize_trace.py results/trace_trained_0.json -o my_vis.html\n",
    )
    parser.add_argument("traces", nargs="+", help="Trace JSON file(s)")
    parser.add_argument("--mode", choices=["2d", "3d"], default="3d",
                        help="Visualisation mode (default: 3d)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: auto)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each trace (default: filenames)")
    args = parser.parse_args()

    # Load traces
    traces_data = []
    for i, path in enumerate(args.traces):
        data = load_trace(path)
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            label = os.path.splitext(os.path.basename(path))[0]
        traces_data.append((label, data))

    # Default output name
    if args.output is None:
        ext = ".html" if args.mode == "3d" else ".png"
        args.output = "trace_vis" + ext

    if args.mode == "3d":
        render_3d(traces_data, args.output)
    else:
        render_2d(traces_data, args.output)
