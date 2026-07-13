"""
All figures for the lab report. Run via run_lab.py, which supplies the
trained models. Every figure shares the palette and semantics defined in
style.py: Value Iteration = blue, Q-Learning = aqua, kept identical to the
Blackjack/DoorKey/Pig labs for consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, FancyArrowPatch, FancyBboxPatch, Rectangle

from style import (
    setup_style, style_axes,
    CATEGORICAL, COLOR_VI, COLOR_QL,
    STATUS_GOOD, STATUS_CRITICAL, SEQUENTIAL_BLUE,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
)
from windy_mdp import ROWS, COLS, START, GOAL, WIND, ACTION_ARROW

setup_style()

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
WIND_CMAP = LinearSegmentedColormap.from_list("wind", ["#eef3fb", "#9ec5f4", "#184f95"])

ARROW_DXY = {"^": (0, -0.32), "v": (0, 0.32), "<": (-0.32, 0), ">": (0.32, 0)}


def _base_grid(ax):
    """Static background: one column band per wind strength, with gridlines."""
    img = np.zeros((ROWS, COLS, 3))
    wmax = max(WIND) or 1
    for c in range(COLS):
        color = WIND_CMAP(WIND[c] / wmax if wmax else 0.0)[:3]
        for r in range(ROWS):
            img[r, c] = color
    ax.imshow(img, origin="upper", extent=[0, COLS, ROWS, 0], interpolation="nearest")
    for gx in range(COLS + 1):
        ax.plot([gx, gx], [0, ROWS], color=SURFACE, lw=1.2, zorder=2)
    for gy in range(ROWS + 1):
        ax.plot([0, COLS], [gy, gy], color=SURFACE, lw=1.2, zorder=2)
    ax.set_xlim(0, COLS); ax.set_ylim(ROWS, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _center(pos):
    r, c = pos
    return c + 0.5, r + 0.5


def _mark(ax, pos, color, label, textcolor="white"):
    x, y = _center(pos)
    ax.add_patch(plt.Circle((x, y), 0.30, color=color, zorder=3))
    ax.text(x, y, label, ha="center", va="center", color=textcolor,
            fontsize=11, fontweight="bold", zorder=4)


# ------------------------------------------------------------- environment ---

def plot_environment(filename):
    fig, ax = plt.subplots(figsize=(9, 7.3))
    _base_grid(ax)
    _mark(ax, START, COLOR_VI, "S")
    _mark(ax, GOAL, STATUS_GOOD, "G")
    for c in range(COLS):
        ax.text((c + 0.5) / COLS, 1.045, str(WIND[c]), ha="center", va="center",
                fontsize=10, color=INK_SECONDARY, fontweight="bold", transform=ax.transAxes)
    ax.text(-0.06, 1.045, "wind:", ha="right", va="center",
            fontsize=10, color=INK_SECONDARY, fontweight="bold", transform=ax.transAxes)
    fig.suptitle("Stochastic Windy Gridworld (7$\\times$10):\n"
                  "wind pushes the agent up by mean strength $\\pm1$", fontsize=12, y=1.0)
    legend = [
        Patch(facecolor=COLOR_VI, label="Start"),
        Patch(facecolor=STATUS_GOOD, label="Goal"),
        Patch(facecolor=WIND_CMAP(1.0)[:3], label="Strong wind column (mean 2)"),
        Patch(facecolor=WIND_CMAP(0.0)[:3], label="Calm column (no wind)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- convergence --

def plot_convergence(history, filename):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(history) + 1), history, color=COLOR_VI, marker="o", markersize=4)
    ax.set_yscale("log")
    style_axes(ax)
    ax.set_xlabel("Value Iteration sweep number")
    ax.set_ylabel(r"Max change in $V(s)$ across all states (log scale)")
    ax.set_title(f"Value Iteration converges in {len(history)} sweeps")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ------------------------------------------------------------- value heatmap -

def plot_value_heatmap(V, filename):
    grid = np.full((ROWS, COLS), np.nan)
    for (r, c), v in V.items():
        grid[r, c] = v
    grid[GOAL[0], GOAL[1]] = 0.0
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)

    fig, ax = plt.subplots(figsize=(9, 6.3))
    im = ax.imshow(grid, origin="upper", extent=[0, COLS, ROWS, 0],
                   cmap=SEQ_CMAP, norm=Normalize(vmin=vmin, vmax=vmax),
                   interpolation="nearest")
    for r in range(ROWS):
        for c in range(COLS):
            ax.text(c + 0.5, r + 0.5, f"{grid[r, c]:.0f}", ha="center", va="center",
                    color=INK_PRIMARY, fontsize=7.5)
    for gx in range(COLS + 1):
        ax.plot([gx, gx], [0, ROWS], color=SURFACE, lw=1.0, zorder=2)
    for gy in range(ROWS + 1):
        ax.plot([0, COLS], [gy, gy], color=SURFACE, lw=1.0, zorder=2)
    ax.set_xlim(0, COLS); ax.set_ylim(ROWS, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"Value Iteration $V^*(s)$ = $-$(expected discounted steps to goal)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"$V^*(s)$")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- gaps ----

def plot_gaps(gaps, filename):
    """Distribution of the best-vs-second-best action-value margin at every
    state -- shows the optimum is NOT flat here (contrast with Pig)."""
    values = np.array(list(gaps.values()))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=30, color=COLOR_VI, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(np.median(values), color=STATUS_CRITICAL, linestyle=(0, (5, 4)),
               linewidth=1.6, label=f"median = {np.median(values):.2f}")
    style_axes(ax)
    ax.set_xlabel(r"$Q(s, a^*) - Q(s, a_{\mathrm{2nd\ best}})$  (best-action margin)")
    ax.set_ylabel("Number of states")
    ax.set_title("The optimum is sharply peaked, not flat\n"
                  "(contrast with Pig's near-ties): every state has a clearly-best action",
                  fontsize=11)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# --------------------------------------------------------- trajectory plot ---

def _draw_trajectory(ax, states, color, title):
    _base_grid(ax)
    _mark(ax, START, INK_MUTED, "S")
    _mark(ax, GOAL, STATUS_GOOD, "G")
    pts = [_center(s) for s in states]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=13, color=color, lw=2.0, zorder=5,
                     shrinkA=4, shrinkB=4, alpha=0.85))
    ax.set_title(f"{title}\n({len(states) - 1} steps, one sampled rollout)", fontsize=10.5)


def plot_trajectories(vi_states, ql_states, filename):
    fig, axes = plt.subplots(2, 1, figsize=(9, 10.5))
    _draw_trajectory(axes[0], vi_states, COLOR_VI, "Value Iteration greedy policy")
    _draw_trajectory(axes[1], ql_states, COLOR_QL, "Q-Learning greedy policy")
    fig.suptitle("Sampled greedy trajectory from start to goal (stochastic wind)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- policy agreement -

def plot_policy_agreement(vi_policy, ql_policy, filename):
    grid = np.zeros((ROWS, COLS))
    for (r, c), a_vi in vi_policy.items():
        grid[r, c] = 1.0 if ql_policy.get((r, c)) == a_vi else 0.0
    n_agree = int(grid.sum())
    n_total = len(vi_policy)

    cmap = LinearSegmentedColormap.from_list("agree", [STATUS_CRITICAL, STATUS_GOOD])
    fig, ax = plt.subplots(figsize=(9, 6.3))
    ax.imshow(grid, origin="upper", extent=[0, COLS, ROWS, 0], cmap=cmap,
             vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == GOAL:
                continue
            a_vi = vi_policy[(r, c)]
            a_ql = ql_policy.get((r, c))
            sym = ACTION_ARROW[a_vi] if a_ql == a_vi else f"{ACTION_ARROW[a_vi]}/{ACTION_ARROW.get(a_ql, '?')}"
            ax.text(c + 0.5, r + 0.5, sym, ha="center", va="center",
                    color=INK_PRIMARY, fontsize=8.5, fontweight="bold")
    _mark(ax, GOAL, STATUS_GOOD, "G")
    for gx in range(COLS + 1):
        ax.plot([gx, gx], [0, ROWS], color=SURFACE, lw=1.0, zorder=2)
    for gy in range(ROWS + 1):
        ax.plot([0, COLS], [gy, gy], color=SURFACE, lw=1.0, zorder=2)
    ax.set_xlim(0, COLS); ax.set_ylim(ROWS, 0)
    ax.set_xticks([]); ax.set_yticks([])
    pct = 100 * n_agree / n_total
    ax.set_title(f"Greedy policy agreement per state (VI action shown; VI/QL where they differ)\n"
                 f"{n_agree}/{n_total} states agree ({pct:.1f}%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- gamma sweep -

def plot_gamma_sweep(gamma_data, filename):
    gammas = [d["gamma"] for d in gamma_data]
    labels = [str(g) for g in gammas]
    steps = [d["avg_steps"] for d in gamma_data]
    vstart = [d["v_start"] for d in gamma_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    bars = ax1.bar(labels, steps, color=COLOR_VI, width=0.6)
    for bar, s in zip(bars, steps):
        ax1.text(bar.get_x() + bar.get_width() / 2, s + 0.3, f"{s:.2f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK_PRIMARY)
    style_axes(ax1)
    ax1.set_xlabel(r"Discount factor $\gamma$")
    ax1.set_ylabel("Average steps to goal (greedy rollout)")
    ax1.set_title("Effect of $\\gamma$ on realised path length\n"
                  "(stochastic wind: smaller $\\gamma$ penalises variance, not just mean length)",
                  fontsize=10.5)

    bars = ax2.bar(labels, vstart, color=COLOR_QL, width=0.6)
    for bar, v in zip(bars, vstart):
        ax2.text(bar.get_x() + bar.get_width() / 2, v - 0.4, f"{v:.2f}",
                 ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK_PRIMARY)
    style_axes(ax2)
    ax2.set_xlabel(r"Discount factor $\gamma$")
    ax2.set_ylabel(r"$V^*(\mathrm{start})$")
    ax2.set_title(r"Effect of $\gamma$ on the start-state value", fontsize=11)

    fig.suptitle("Discount factor sweep (Value Iteration)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- theta sweep -

def plot_theta_sweep(theta_data, filename):
    """Effect of the VI stopping threshold theta: sweeps-to-converge (cost)
    vs. policy agreement with the tightest-theta run (quality)."""
    thetas = [d["theta"] for d in theta_data]
    labels = [f"{t:.0e}" for t in thetas]
    sweeps = [d["sweeps"] for d in theta_data]
    agree = [d["policy_agreement_pct"] for d in theta_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(labels, sweeps, color=COLOR_VI, marker="o", lw=2.4)
    style_axes(ax1)
    ax1.set_xlabel(r"Stopping threshold $\theta$")
    ax1.set_ylabel("Sweeps to converge")
    ax1.set_yscale("log")
    ax1.set_title(r"Looser $\theta$ converges in fewer sweeps", fontsize=11)

    ax2.plot(labels, agree, color=COLOR_QL, marker="o", lw=2.4)
    ax2.set_ylim(0, 105)
    style_axes(ax2)
    ax2.set_xlabel(r"Stopping threshold $\theta$")
    ax2.set_ylabel(r"Policy agreement vs. tightest $\theta$ (%)")
    ax2.set_title(r"...but the greedy policy stabilises long before $V$ does", fontsize=11)

    fig.suptitle(r"Value Iteration stopping threshold $\theta$: cost vs. policy quality",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- sweeps ----

def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_sweep(series, vi_reference, window, xlabel, ylabel, title, filename,
               ref_label="Value Iteration (true optimum)"):
    from scipy.ndimage import gaussian_filter1d
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (label, arr) in enumerate(series.items()):
        color = CATEGORICAL[i]
        ma = moving_average(arr, window)
        x = np.arange(len(ma))
        smooth = gaussian_filter1d(ma, sigma=window * 0.9)
        ax.plot(x, ma, color=color, alpha=0.18, lw=1.2, zorder=2)
        ax.plot(x, smooth, color=color, lw=2.8, label=label, zorder=4)
    if vi_reference is not None:
        ax.axhline(vi_reference, color=INK_SECONDARY, linestyle=(0, (5, 4)),
                   linewidth=1.6, zorder=3, label=ref_label)
    style_axes(ax)
    ax.margins(x=0.01)
    ax.set_xlabel(f"{xlabel} (moving average, window={window})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ------------------------------------------------------------ final compare --

def plot_evaluation_comparison(vi_eval, ql_eval, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
    labels = ["Value\nIteration", "Q-Learning"]
    colors = [COLOR_VI, COLOR_QL]

    steps = [vi_eval["avg_steps"], ql_eval["avg_steps"]]
    bars = ax1.bar(labels, steps, color=colors, width=0.55)
    for bar, s in zip(bars, steps):
        ax1.text(bar.get_x() + bar.get_width() / 2, s + 0.15, f"{s:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    style_axes(ax1)
    ax1.set_ylabel("Average steps to goal")
    ax1.set_title("Path length\n(5,000-episode rollout, lower is better)", fontsize=11)

    succ = [vi_eval["success_rate"], ql_eval["success_rate"]]
    bars = ax2.bar(labels, succ, color=colors, width=0.55)
    for bar, s in zip(bars, succ):
        ax2.text(bar.get_x() + bar.get_width() / 2, s + 0.008, f"{s:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK_PRIMARY)
    ax2.set_ylim(0, 1.05)
    style_axes(ax2)
    ax2.set_ylabel("Success rate")
    ax2.set_title("Goal-reached rate\n(5,000-episode rollout)", fontsize=11)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# --------------------------------------------------------------- animations --
#
# The animations use a Gymnasium-style scene (near-black canvas, flat
# geometric shapes, flag on the goal, purple agent -- the Lunar Lander look)
# rather than the report's light chart style: they show the TASK, not a chart.

GYM_BG = "#0b0d12"          # canvas
GYM_PANEL = "#11141b"       # calm cells / panel background
GYM_GRIDLINE = "#232833"
GYM_WIND = {0: GYM_PANEL, 1: "#16263f", 2: "#1e3c66"}
GYM_WIND_CHEVRON = {1: "#3d6db3", 2: "#6ea8ff"}
GYM_AGENT = "#8b7ff2"       # Lunar-Lander purple
GYM_TRAIL = "#8b7ff2"
GYM_FLAG = "#ffd23f"        # goal flag yellow
GYM_POLE = "#e8eaf0"
GYM_START = "#9aa3b2"
GYM_TEXT = "#e8eaf0"
GYM_MUTED = "#8b93a3"
GYM_GOOD = "#3ddc84"
GYM_BAD = "#ff5a5f"
GYM_CURVE = "#27c99a"
GYM_REF = "#6ea8ff"

VALUE_CMAP = LinearSegmentedColormap.from_list(
    "gym_value", ["#141b2e", "#1d3a63", "#2a78d6", "#27c99a"])
VALUE_CMAP.set_bad(GYM_PANEL)

GYM_FONT = {"family": "DejaVu Sans", "color": GYM_TEXT}


def _gym_axes(ax):
    ax.set_facecolor(GYM_BG)
    ax.tick_params(colors=GYM_MUTED, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_color(GYM_GRIDLINE)
    ax.grid(color=GYM_GRIDLINE, alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)


def _draw_scene(ax, states_so_far, step, wind_phase, hud_left, hud_right,
                banner=None, banner_color=GYM_GOOD):
    """Redraw the full windy-grid scene onto `ax` (cleared each frame).

    states_so_far : trajectory up to and including the current position
    wind_phase    : [0,1) scroll offset that makes the wind chevrons drift up
    """
    ax.clear()
    ax.set_facecolor(GYM_BG)
    ax.set_xlim(-0.25, COLS + 0.25)
    ax.set_ylim(ROWS + 0.55, -0.85)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # wind-shaded columns
    for c in range(COLS):
        ax.add_patch(Rectangle((c, 0), 1, ROWS, facecolor=GYM_WIND[WIND[c]],
                               edgecolor="none", zorder=1))
    for gx in range(COLS + 1):
        ax.plot([gx, gx], [0, ROWS], color=GYM_GRIDLINE, lw=0.9, zorder=2)
    for gy in range(ROWS + 1):
        ax.plot([0, COLS], [gy, gy], color=GYM_GRIDLINE, lw=0.9, zorder=2)

    # drifting chevrons show wind strength and direction (up)
    for c in range(COLS):
        w = WIND[c]
        label_color = GYM_WIND_CHEVRON.get(w, GYM_GRIDLINE)
        ax.text(c + 0.5, ROWS + 0.34, str(w), ha="center", va="center",
                fontsize=9, color=label_color, fontweight="bold")
        if w == 0:
            continue
        spacing = 1.5 if w == 1 else 0.75
        y = -(wind_phase * spacing)
        ys = []
        while y < ROWS:
            if y > 0.25:
                ys.append(y)
            y += spacing
        if ys:
            ax.plot([c + 0.5] * len(ys), ys, linestyle="none", marker="^",
                    markersize=5 + 2 * w, markerfacecolor="none",
                    markeredgecolor=GYM_WIND_CHEVRON[w], markeredgewidth=1.3,
                    alpha=0.65, zorder=3)

    # start pad
    sx, sy = _center(START)
    ax.add_patch(plt.Circle((sx, sy), 0.30, facecolor="none", edgecolor=GYM_START,
                            lw=1.6, zorder=4))
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=9.5,
            color=GYM_START, fontweight="bold", zorder=4)

    # goal: Lunar-Lander-style flag
    gx, gy = _center(GOAL)
    ax.plot([gx - 0.08, gx - 0.08], [gy + 0.38, gy - 0.42], color=GYM_POLE,
            lw=2.2, solid_capstyle="round", zorder=5)
    ax.add_patch(plt.Polygon([[gx - 0.08, gy - 0.42], [gx + 0.42, gy - 0.27],
                              [gx - 0.08, gy - 0.12]],
                             closed=True, facecolor=GYM_FLAG, edgecolor="none", zorder=5))

    # trajectory: faint full path + bright recent trail
    if len(states_so_far) > 1:
        pts = [_center(s) for s in states_so_far]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=GYM_TRAIL, lw=1.4, alpha=0.16, zorder=5)
        tail = pts[-10:]
        for i in range(len(tail) - 1):
            a = 0.12 + 0.55 * (i + 1) / (len(tail) - 1)
            ax.plot([tail[i][0], tail[i + 1][0]], [tail[i][1], tail[i + 1][1]],
                    color=GYM_TRAIL, lw=2.6, alpha=a,
                    solid_capstyle="round", zorder=6)

    # the agent: flat purple square, Gymnasium-style
    axp, ayp = _center(states_so_far[-1])
    ax.add_patch(FancyBboxPatch((axp - 0.26, ayp - 0.26), 0.52, 0.52,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=GYM_AGENT, edgecolor="#c7c0ff",
                                linewidth=1.2, zorder=7))

    # HUD
    ax.text(0.0, -0.42, hud_left, ha="left", va="center", fontsize=10.5,
            color=GYM_TEXT, fontweight="bold")
    ax.text(COLS, -0.42, hud_right, ha="right", va="center", fontsize=10.5,
            color=GYM_MUTED)

    if banner:
        ax.text(COLS / 2, ROWS / 2, banner, ha="center", va="center",
                fontsize=17, color=banner_color, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.55", facecolor=GYM_BG,
                          edgecolor=banner_color, linewidth=1.6, alpha=0.92))


def _draw_value_panel(ax, Q_snapshot):
    """Learned state-value heatmap V(s) = max_a Q(s, a); unvisited states dark."""
    ax.clear()
    grid = np.full((ROWS, COLS), np.nan)
    if Q_snapshot:
        for (r, c), q in Q_snapshot.items():
            grid[r, c] = float(np.max(q))
    ax.imshow(grid, origin="upper", extent=[0, COLS, ROWS, 0], cmap=VALUE_CMAP,
              interpolation="nearest")
    for gx2 in range(COLS + 1):
        ax.plot([gx2, gx2], [0, ROWS], color=GYM_GRIDLINE, lw=0.5)
    for gy2 in range(ROWS + 1):
        ax.plot([0, COLS], [gy2, gy2], color=GYM_GRIDLINE, lw=0.5)
    gx, gy = _center(GOAL)
    ax.plot([gx], [gy], marker="*", markersize=9, color=GYM_FLAG, linestyle="none")
    ax.set_xlim(0, COLS); ax.set_ylim(ROWS, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(GYM_PANEL)
    for spine in ax.spines.values():
        spine.set_color(GYM_GRIDLINE)
    n_seen = 0 if not Q_snapshot else len(Q_snapshot)
    ax.set_title(f"learned value  V(s) = max Q(s,a)   ({n_seen}/{ROWS * COLS - 1} states seen)",
                 fontsize=9, color=GYM_MUTED, pad=5)


def _gif_frames(snapshots, hold_start=6, hold_end=10, speedup_after=50):
    """(snapshot_idx, step_idx, display_speed) triples; long wanderers play at 2x."""
    frames = []
    for si, snap in enumerate(snapshots):
        n = len(snap["states"])
        if n == 0:
            continue
        stride = 2 if n > speedup_after else 1
        frames.extend((si, 0, stride) for _ in range(hold_start))
        frames.extend((si, i, stride) for i in range(0, n, stride))
        frames.extend((si, n - 1, stride) for _ in range(hold_end))
    return frames


def animate_training_rollouts(snapshots, episode_steps, vi_reference, filename):
    """The learning film: actual epsilon-greedy training episodes at
    checkpoints, in a Gymnasium-style scene, alongside the value function the
    agent has learned so far and a steps-to-goal learning curve with a cursor
    marking where in training each episode sits."""
    steps_arr = np.asarray(episode_steps, dtype=float)
    window = 200
    ma = moving_average(steps_arr, window)
    ma_x = np.arange(window - 1, window - 1 + len(ma))

    frames = _gif_frames(snapshots)

    fig = plt.figure(figsize=(12.6, 6.1))
    fig.patch.set_facecolor(GYM_BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1], height_ratios=[1.05, 1],
                          left=0.035, right=0.975, top=0.90, bottom=0.075,
                          wspace=0.14, hspace=0.30)
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_value = fig.add_subplot(gs[0, 1])
    ax_curve = fig.add_subplot(gs[1, 1])

    fig.suptitle("Q-LEARNING IN THE STOCHASTIC WINDY GRIDWORLD — actual training episodes",
                 fontsize=12.5, color=GYM_TEXT, fontweight="bold", x=0.5, y=0.965)

    # static learning-curve panel, cursor moved per episode
    _gym_axes(ax_curve)
    ax_curve.plot(ma_x, ma, color=GYM_CURVE, lw=1.8)
    ax_curve.axhline(vi_reference, color=GYM_REF, linestyle=(0, (5, 4)), lw=1.3)
    ax_curve.text(0.02, vi_reference * 1.12, f"VI optimum {vi_reference:.1f}",
                  ha="left", va="bottom", fontsize=8, color=GYM_REF,
                  transform=ax_curve.get_yaxis_transform())
    ax_curve.set_yscale("log")
    ax_curve.set_xlim(0, len(steps_arr))
    ax_curve.set_title("steps to goal per training episode (moving avg)",
                       fontsize=9, color=GYM_MUTED, pad=5)
    cursor = ax_curve.axvline(0, color=GYM_TEXT, lw=1.1, alpha=0.9)
    cursor_dot, = ax_curve.plot([], [], marker="o", markersize=5, color=GYM_TEXT,
                                linestyle="none")

    last_si = [-1]

    def update(f):
        si, step, stride = frames[f]
        snap = snapshots[si]
        states = snap["states"]

        if snap["success"] and step == len(states) - 1 and not snap["truncated_for_video"]:
            banner, bcolor = f"GOAL  in {snap['steps']} steps", GYM_GOOD
        elif step == len(states) - 1 and snap["truncated_for_video"] and snap["success"]:
            banner, bcolor = f"still lost — goal only after {snap['steps']} steps", GYM_BAD
        elif step == len(states) - 1 and not snap["success"]:
            banner, bcolor = f"FAILED — no goal in {snap['steps']} steps", GYM_BAD
        else:
            banner, bcolor = None, GYM_GOOD

        speed = "  |  2x speed" if stride == 2 else ""
        _draw_scene(
            ax_scene, states[:step + 1], step,
            wind_phase=(f * 0.18) % 1.0,
            hud_left=f"EPISODE {snap['episode']:,}",
            hud_right=f"eps = {snap['epsilon']:.3f}   step {step}   return {-(step):,.0f}{speed}",
            banner=banner, banner_color=bcolor,
        )

        if si != last_si[0]:
            _draw_value_panel(ax_value, snap.get("Q"))
            ep = snap["episode"]
            cursor.set_xdata([ep, ep])
            if len(ma):
                j = np.clip(ep - (window - 1), 0, len(ma) - 1)
                cursor_dot.set_data([ep], [ma[int(j)]])
            last_si[0] = si
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=110, blit=False)
    anim.save(filename, writer=PillowWriter(fps=10), dpi=100,
              savefig_kwargs={"facecolor": GYM_BG})
    plt.close(fig)


def animate_greedy_trajectory(states, filename):
    """One learned greedy rollout through the stochastic wind, Gymnasium-style."""
    n = len(states)
    frames = [0] * 5 + list(range(n)) + [n - 1] * 10

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    fig.patch.set_facecolor(GYM_BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
    fig.suptitle("LEARNED GREEDY POLICY — one rollout under stochastic wind",
                 fontsize=11.5, color=GYM_TEXT, fontweight="bold", y=0.965)

    def update(f):
        step = frames[f]
        banner = f"GOAL  in {n - 1} steps" if step == n - 1 else None
        _draw_scene(ax, states[:step + 1], step, wind_phase=(f * 0.18) % 1.0,
                    hud_left="Q-LEARNING  (greedy)",
                    hud_right=f"step {step}   return {-step:,.0f}",
                    banner=banner)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=150, blit=False)
    anim.save(filename, writer=PillowWriter(fps=7), dpi=100,
              savefig_kwargs={"facecolor": GYM_BG})
    plt.close(fig)


def animate_learning_progress(episode_steps, vi_reference, filename, window=300, frames=90):
    """Animated learning curve, restyled to match the Gymnasium-dark scenes."""
    steps = np.asarray(episode_steps, dtype=float)
    if len(steps) < window:
        window = max(5, len(steps) // 10)
    ma = moving_average(steps, window)
    if len(ma) == 0:
        return
    x = np.arange(window - 1, window - 1 + len(ma))
    frame_idx = np.unique(np.linspace(1, len(ma), min(frames, len(ma)), dtype=int))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor(GYM_BG)
    _gym_axes(ax)
    line, = ax.plot([], [], color=GYM_CURVE, lw=2.6)
    dot, = ax.plot([], [], marker="o", color=GYM_TEXT, markersize=6)
    ax.axhline(vi_reference, color=GYM_REF, linestyle=(0, (5, 4)), lw=1.4,
               label=f"Value Iteration optimum ({vi_reference:.1f})")
    ax.set_yscale("log")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(max(vi_reference * 0.7, 5), np.nanmax(ma) * 1.4)
    ax.set_xlabel("training episode", color=GYM_MUTED)
    ax.set_ylabel(f"steps to goal (moving avg, w={window})", color=GYM_MUTED)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(GYM_MUTED)

    def update(nf):
        end = frame_idx[nf]
        line.set_data(x[:end], ma[:end])
        dot.set_data([x[end - 1]], [ma[end - 1]])
        ax.set_title(f"Q-LEARNING PROGRESS — episode {int(x[end - 1]):,}:  "
                     f"{ma[end - 1]:.1f} avg steps to goal",
                     fontsize=11, color=GYM_TEXT, fontweight="bold", pad=10)
        return line, dot

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=70, blit=False)
    anim.save(filename, writer=PillowWriter(fps=12), dpi=110,
              savefig_kwargs={"facecolor": GYM_BG})
    plt.close(fig)
