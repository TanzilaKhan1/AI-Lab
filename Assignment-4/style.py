"""
Shared plot styling. Colors follow a validated categorical/sequential/diverging/
status palette (fixed hue order, checked for colorblind-safe adjacent contrast)
rather than matplotlib defaults, and every figure uses the same semantic mapping:
Value Iteration = blue, Q-Learning = aqua, "Stand" = blue, "Hit" = orange.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Categorical palette, fixed order -- never reassigned per-chart.
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

CATEGORICAL = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE]

# Fixed semantic roles used consistently across every figure in this report.
COLOR_VI = BLUE
COLOR_QL = AQUA
COLOR_STAND = BLUE
COLOR_HIT = ORANGE

# Status palette (reserved for state, never reused as a categorical color).
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# Sequential ramp (single hue, light -> dark), for magnitude.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Chart chrome.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "red_gray_blue", [RED, "#ef9a92", "#f0efec", "#6da7ec", "#184f95"]
)


def setup_style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.labelpad": 7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.9,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.4,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.antialiased": True,
        "patch.antialiased": True,
        "figure.dpi": 120,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def style_axes(ax, y_zero_line=False):
    """Recessive grid/axes: keep only what's needed to read values."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    if y_zero_line:
        ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=0)


def diverging_norm(vmin, vmax):
    return TwoSlopeNorm(vmin=min(vmin, -1e-9), vcenter=0.0, vmax=max(vmax, 1e-9))
