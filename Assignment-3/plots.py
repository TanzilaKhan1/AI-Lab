"""
Plots and animation for the wildfire evacuation results: the static map,
the learned pheromone corridors, the ACO-vs-baselines convergence figure,
and the fire/road/car animation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter


GROUP_COLORS = ["#0d47a1", "#6a1b9a", "#e65100", "#00695c",
                "#ad1457", "#4e342e", "#00838f", "#f9a825"]


# ======================================================================
# Shared drawing helpers
# ======================================================================
def _draw_base_map(ax, inst, cfg, fire_time=None, houses=False):
    """Terrain, roads, fire-arrival shading, ignition, origins, shelters."""
    coords = inst.coords
    ax.set_facecolor("#f0f4ec")

    # Fire arrival field as soft filled contours (when will fire get here).
    xs = coords[:, 0]
    ys = coords[:, 1]
    tri_t = inst.t_fire
    levels = np.linspace(tri_t.min(), np.percentile(tri_t, 92), 12)
    ax.tricontourf(xs, ys, tri_t, levels=levels, cmap="YlOrRd_r", alpha=0.45)

    # Roads: grey if safe now, dark if already burned (for animation).
    for (u, v) in inst.edges:
        burned = (fire_time is not None and
                  min(inst.t_fire[u], inst.t_fire[v]) <= fire_time)
        ax.plot(coords[[u, v], 0], coords[[u, v], 1],
                color="#3d3d3d" if burned else "#b9c0b3",
                lw=2.4 if burned else 1.3, alpha=0.9 if burned else 0.8,
                zorder=2)

    # Ignition point.
    ax.scatter(*inst.ignition, marker="X", s=260, c="#d62828",
               edgecolors="black", linewidths=1.2, zorder=8)
    # Wind arrow.
    c0 = inst.ignition
    ax.annotate("wind", xy=c0 + inst.wind * 6.5, xytext=c0 + inst.wind * 1.5,
                arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=2.2),
                color="#d62828", fontweight="bold", fontsize=9, zorder=8)

    # Origins (homes) and shelters.
    if houses:
        for o in inst.origins:
            _draw_house(ax, coords[o], "#1d6fb8", size=0.95)
        for s in inst.shelters:
            _draw_house(ax, coords[s], "#1e8449", size=1.25, label="SHELTER")
    else:
        for o in inst.origins:
            ax.scatter(*coords[o], marker="^", s=200, c="#1d6fb8",
                       edgecolors="white", linewidths=1.4, zorder=9)
        for s in inst.shelters:
            ax.scatter(*coords[s], marker="s", s=230, c="#1e8449",
                       edgecolors="white", linewidths=1.4, zorder=9)


def _draw_car(ax, center, heading, color, size=0.75):
    """Draw a small top-down car: rotated body, darker cabin, headlight dot."""
    cx, cy = center
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    L, W = size, size * 0.52

    def poly(pts_local):
        return (np.array(pts_local) @ R.T) + center

    body = poly([[-L, -W], [L, -W], [L * 1.15, 0], [L, W], [-L, W]])
    cabin = poly([[-L * 0.45, -W * 0.7], [L * 0.35, -W * 0.7],
                  [L * 0.35, W * 0.7], [-L * 0.45, W * 0.7]])
    ax.fill(body[:, 0], body[:, 1], color=color, ec="white", lw=1.0,
            zorder=11)
    ax.fill(cabin[:, 0], cabin[:, 1], color="#0b0b0b", alpha=0.55, zorder=12)
    nose = np.array(center) + R @ np.array([L * 1.05, 0.0])
    ax.scatter([nose[0]], [nose[1]], s=14, c="#fff59d", zorder=13,
               edgecolors="none")


def _draw_house(ax, center, color, size=0.9, label=None):
    """Small house glyph: square walls plus a triangular roof."""
    cx, cy = center
    h = size
    walls = np.array([[cx - h/2, cy - h/2], [cx + h/2, cy - h/2],
                      [cx + h/2, cy + h/4], [cx - h/2, cy + h/4]])
    roof = np.array([[cx - h/1.55, cy + h/4], [cx + h/1.55, cy + h/4],
                     [cx, cy + h * 0.95]])
    ax.fill(walls[:, 0], walls[:, 1], color=color, ec="white", lw=1.1,
            zorder=9)
    ax.fill(roof[:, 0], roof[:, 1], color="#5d4037", ec="white", lw=1.1,
            zorder=9)
    if label:
        ax.text(cx, cy - h * 0.95, label, ha="center", va="top",
                fontsize=6.5, fontweight="bold", color=color, zorder=9)


def _route_progress(inst, route, t):
    """Where along its route a group is at time t (drives at road speed)."""
    if route is None:
        return inst.coords[route[0]] if route else None
    pos = inst.coords[route[0]].copy()
    t_used = 0.0
    for u, v in zip(route[:-1], route[1:]):
        te = next(tt for (m, tt) in inst.adj[u] if m == v)
        if t_used + te >= t:
            frac = (t - t_used) / te
            return inst.coords[u] + frac * (inst.coords[v] - inst.coords[u])
        t_used += te
    return inst.coords[route[-1]].copy()      # arrived


# ======================================================================
# PLOTS
# ======================================================================
def plot_evacuation_map(inst, cfg, routes, cost):
    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    _draw_base_map(ax, inst, cfg)
    for gi, route in enumerate(routes):
        if route is None:
            continue
        pts = inst.coords[route]
        ax.plot(pts[:, 0], pts[:, 1],
                color=GROUP_COLORS[gi % len(GROUP_COLORS)], lw=3.2,
                alpha=0.95, zorder=6, solid_capstyle="round",
                label=f"Group {gi + 1} route")
    ax.set_title(f"Wildfire evacuation plan (ACO): total {cost:.2f} h",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT_DIR}/evacuation_map.png", dpi=140)
    plt.close(fig)


def plot_pheromone(inst, cfg, tau):
    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")
    tmax = max(tau.values())
    for (u, v) in inst.edges:
        s = max(tau[(u, v)], tau[(v, u)]) / tmax
        if s < 0.05:
            continue
        ax.plot(inst.coords[[u, v], 0], inst.coords[[u, v], 1],
                color=cm.plasma(s), lw=0.6 + 5.0 * s,
                alpha=0.25 + 0.65 * s, zorder=2)
    ax.scatter(*inst.ignition, marker="X", s=240, c="#ff5252",
               edgecolors="white", linewidths=1.2, zorder=8)
    for o in inst.origins:
        ax.scatter(*inst.coords[o], marker="^", s=170, c="#64b5f6",
                   edgecolors="white", linewidths=1.0, zorder=9)
    for s_ in inst.shelters:
        ax.scatter(*inst.coords[s_], marker="s", s=190, c="#66bb6a",
                   edgecolors="white", linewidths=1.0, zorder=9)
    ax.set_title("Learned pheromone corridors (bright = trusted escape roads)",
                 color="white", fontweight="bold", fontsize=13)
    ax.set_xlabel("x (km)", color="white")
    ax.set_ylabel("y (km)", color="white")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{cfg.OUT_DIR}/pheromone.png", dpi=140,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_convergence(aco_hist, aco_iter_best, greedy_cost, static_cost, cfg):
    """
    Two-panel convergence figure. Left: the full picture against both
    baselines (their trapped-group penalties can dwarf the ACO curve, which
    is the point). Right: zoom on the ACO search itself, iteration-best
    exploration trace over the descending best-so-far staircase, so the
    optimization dynamics are visible instead of a flat line.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # Left panel: context against baselines.
    axL.plot(aco_hist, color="#b23a48", lw=2.6, label="ACO best so far")
    axL.axhline(greedy_cost, color="#e67e22", linestyle="--", lw=2,
                label=f"Fire-aware greedy ({greedy_cost:.2f} h)")
    axL.axhline(static_cost, color="#7f8c8d", linestyle=":", lw=2,
                label=f"Fire-blind shortest path ({static_cost:.2f} h)")
    axL.set_xlabel("Iteration")
    axL.set_ylabel("Total evacuation time (h, penalized)")
    axL.set_title("ACO vs baselines", fontweight="bold")
    axL.legend(fontsize=9)
    axL.grid(True, alpha=0.3)

    # Right panel: zoom on the search dynamics.
    axR.plot(aco_iter_best, color="#8ecae6", lw=1.1, alpha=0.9,
             label="Best ant each iteration (exploration)")
    axR.plot(aco_hist, color="#b23a48", lw=2.6, label="Best so far")
    final = aco_hist[-1]
    conv = next((i for i, h in enumerate(aco_hist) if h - final <= 1e-9), 0)
    axR.axvline(conv, color="#2ecc71", lw=1.2, alpha=0.7)
    # Clip y to the interesting band so the descent is visible.
    lo = final - 0.05
    hi = float(np.percentile(aco_iter_best, 96))
    axR.set_ylim(lo, hi)
    axR.set_xlim(0, min(max(conv * 4, 60), len(aco_hist)))
    axR.annotate(f"converged ~iter {conv}", xy=(conv, final),
                 xytext=(conv + 6, lo + 0.35 * (hi - lo)),
                 color="#207a3a", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#207a3a"))
    axR.set_xlabel("Iteration")
    axR.set_title("Zoom: colony search dynamics", fontweight="bold")
    axR.legend(fontsize=9)
    axR.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{cfg.OUT_DIR}/convergence.png", dpi=140)
    plt.close(fig)


# ======================================================================
# ANIMATION: fire spreads, roads burn, groups escape
# ======================================================================
def animate_evacuation(inst, cfg, routes, cost):
    """
    Realistic emergency-operations animation.

    The fire is drawn in three layers: a charred black core where it has been
    burning longest, a bright burning rim just behind the front, and glowing
    embers scattered through the rim. Road segments blacken as they burn.
    Each neighbourhood group drives its ACO route as a small labelled car that
    turns green on reaching a shelter. Houses mark origins, labelled houses
    mark shelters, and a HUD counts minutes, groups safe, and roads lost.
    """
    coords = inst.coords
    arrive = []
    for route in routes:
        if route is None:
            arrive.append(np.inf)
            continue
        t = 0.0
        for u, v in zip(route[:-1], route[1:]):
            t += next(tt for (m, tt) in inst.adj[u] if m == v)
        arrive.append(t)
    t_end = max([a for a in arrive if np.isfinite(a)] + [0.5]) * 1.30

    fig, ax = plt.subplots(figsize=(9.5, 8.8))
    trails = [[] for _ in routes]

    # Fine grid of the fire arrival field for the front polygons.
    gx, gy = np.meshgrid(
        np.linspace(coords[:, 0].min() - 2, coords[:, 0].max() + 2, 170),
        np.linspace(coords[:, 1].min() - 2, coords[:, 1].max() + 2, 170))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    vec = pts - inst.ignition
    dist = np.sqrt((vec ** 2).sum(axis=1)) + 1e-9
    cosang = (vec @ inst.wind) / dist
    rate = cfg.BASE_SPREAD * (1.0 + cfg.WIND_STRENGTH * cosang)
    rate = np.maximum(rate, cfg.BASE_SPREAD * (1.0 - cfg.WIND_STRENGTH) * 0.5)
    t_field = (dist / rate).reshape(gx.shape)
    # Deterministic ember points: subsample the grid once.
    ember_idx = np.arange(0, pts.shape[0], 37)
    ember_pts = pts[ember_idx]
    ember_t = (dist / rate)[ember_idx]

    def draw(idx):
        t = t_end * idx / max(1, cfg.ANIM_FRAMES - 1)
        ax.clear()
        _draw_base_map(ax, inst, cfg, fire_time=t, houses=True)

        # Three-layer fire: charred core, burning rim, bright front line.
        t_lvl = max(t, 1e-3)
        core = max(t_lvl - 0.35, 0.0)
        if core > 0:
            ax.contourf(gx, gy, t_field, levels=[0.0, core],
                        colors=["#1a0d08"], alpha=0.78, zorder=3)
        ax.contourf(gx, gy, t_field, levels=[core, t_lvl],
                    colors=["#e25822"], alpha=0.65, zorder=3)
        ax.contour(gx, gy, t_field, levels=[t_lvl], colors=["#ffb300"],
                   linewidths=2.6, zorder=4)
        # Embers: glowing dots inside the burning rim, brightness by age.
        in_rim = (ember_t <= t_lvl) & (ember_t > core)
        if in_rim.any():
            age = (t_lvl - ember_t[in_rim]) / max(t_lvl - core, 1e-6)
            ax.scatter(ember_pts[in_rim, 0], ember_pts[in_rim, 1],
                       s=6 + 14 * (1 - age), c="#ffd54f",
                       alpha=0.35 + 0.45 * (1 - age), zorder=5,
                       edgecolors="none")

        n_safe, n_burned = 0, 0
        for (u, v) in inst.edges:
            if min(inst.t_fire[u], inst.t_fire[v]) <= t:
                n_burned += 1

        for gi, route in enumerate(routes):
            col = GROUP_COLORS[gi % len(GROUP_COLORS)]
            if route is None:
                continue
            pos = _route_progress(inst, route, t)
            ahead = _route_progress(inst, route, min(t + 0.02, arrive[gi]))
            heading = np.arctan2(ahead[1] - pos[1], ahead[0] - pos[0]) \
                if not np.allclose(ahead, pos) else 0.0
            done = t >= arrive[gi]
            if done:
                n_safe += 1
            trails[gi].append(pos.copy())
            tr = np.array(trails[gi][-20:])
            if len(tr) > 1:
                ax.plot(tr[:, 0], tr[:, 1], color=col, lw=2.4, alpha=0.45,
                        zorder=6)
            _draw_car(ax, pos, heading, "#2ecc71" if done else col)
            ax.annotate(f"G{gi + 1}", pos, textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7,
                        fontweight="bold", color=col, zorder=14)

        hud = (f"t = {t * 60:5.0f} min    safe {n_safe}/{len(routes)}    "
               f"roads burned {n_burned}/{len(inst.edges)}")
        ax.text(0.02, 0.975, hud, transform=ax.transAxes, va="top",
                fontsize=10, fontweight="bold", color="#7f1d1d",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#bbb", alpha=0.92), zorder=15)
        ax.set_title(f"Wildfire evacuation ({len(routes)} groups, "
                     f"ACO plan {cost:.2f} h)", fontweight="bold", fontsize=13)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal")

    anim = FuncAnimation(fig, draw, frames=cfg.ANIM_FRAMES,
                         interval=1000 // cfg.ANIM_FPS)
    anim.save(f"{cfg.OUT_DIR}/evacuation_animation.gif",
              writer=PillowWriter(fps=cfg.ANIM_FPS), dpi=95)
    plt.close(fig)
