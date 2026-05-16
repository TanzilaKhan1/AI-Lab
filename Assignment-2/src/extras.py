"""Extra visualisations from one solved instance (N=80, MRV)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from . import algorithms as alg
from .problem import SLOTS_PER_DAY, load_stations_from_osm, make_instance

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
WEB = ROOT / "web"
WEB.mkdir(exist_ok=True)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    stations = load_stations_from_osm(seed=0)
    inst = make_instance(80, seed=42, stations=stations)
    res = alg.backtracking_mrv(inst, time_limit=20)

    sids = [s.sid for s in inst.stations]
    snames = {s.sid: (s.name[:18] + ('…' if len(s.name) > 18 else '')) for s in inst.stations}
    sid_idx = {sid: i for i, sid in enumerate(sids)}
    occ = np.zeros((len(sids), SLOTS_PER_DAY), dtype=int)
    cap = np.zeros_like(occ)
    for vid, (sid, t) in res.assignment.items():
        occ[sid_idx[sid], t] += 1
    for s in inst.stations:
        for t in range(SLOTS_PER_DAY):
            cap[sid_idx[s.sid], t] = s.cap(t)
    util = np.where(cap > 0, occ / np.maximum(cap, 1), 0.0)

    # static heatmap
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(util, ax=ax, cmap="rocket_r", vmin=0, vmax=1.5,
                annot=occ, fmt="d", cbar_kws={"label": "utilisation"},
                yticklabels=[snames[s] for s in sids],
                xticklabels=[f"slot {i}" for i in range(SLOTS_PER_DAY)])
    ax.set_title("Station × slot utilisation (MRV, N=80)\n— number in cell = vehicles assigned")
    fig.tight_layout()
    fig.savefig(FIGS / "slot_heatmap.png", dpi=160)
    plt.close(fig)
    print(f"[save] {FIGS/'slot_heatmap.png'}")

    pfig = px.imshow(
        util,
        x=[f"slot {i}" for i in range(SLOTS_PER_DAY)],
        y=[snames[s] for s in sids],
        color_continuous_scale="Inferno_r", aspect="auto",
        labels=dict(color="utilisation", x="time slot", y="station"),
        title="Dhaka stations × time slots — utilisation under MRV (N=80)",
        text_auto=False,
    )
    pfig.update_layout(template="plotly_white", height=700,
                       font=dict(family="Inter,system-ui"))
    pfig.write_html(WEB / "heatmap.html", include_plotlyjs="cdn")
    print(f"[save] {WEB/'heatmap.html'}")

    # class breakdown
    rows = [{"class": v.vclass, "assigned": res.assignment.get(v.vid) is not None}
            for v in inst.vehicles]
    cdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cdf["class"].value_counts().plot.pie(
        ax=axes[0], autopct="%1.0f%%",
        colors=["#3b82f6", "#a855f7", "#10b981", "#ef4444"][:cdf["class"].nunique()])
    axes[0].set_title("Vehicle class mix (N=80)")
    axes[0].set_ylabel("")
    g = cdf.groupby("class")["assigned"].mean().sort_values()
    g.plot.barh(ax=axes[1], color="#3b82f6")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_title("Per-class assignment success rate")
    axes[1].set_xlabel("fraction assigned")
    fig.tight_layout()
    fig.savefig(FIGS / "class_breakdown.png", dpi=160)
    plt.close(fig)
    print(f"[save] {FIGS/'class_breakdown.png'}")

    # per-station load distribution
    load = pd.Series(occ.sum(axis=1), index=[snames[s] for s in sids], name="daily_load")
    fig, ax = plt.subplots(figsize=(13, 6))
    load.sort_values(ascending=False).plot.bar(ax=ax, color="#ef4444")
    ax.set_title("Per-station daily load under MRV solution (N=80)")
    ax.set_xlabel("station")
    ax.set_ylabel("vehicles served")
    fig.tight_layout()
    fig.savefig(FIGS / "load_distribution.png", dpi=160)
    plt.close(fig)
    print(f"[save] {FIGS/'load_distribution.png'}")


if __name__ == "__main__":
    main()
