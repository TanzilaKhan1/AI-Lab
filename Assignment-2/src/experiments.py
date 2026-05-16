"""Benchmark the five solvers across vehicle sizes."""
from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import algorithms as alg
from .problem import load_stations_from_osm, make_instance

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

SIZES = [10, 20, 30, 50, 75, 100]
REPEATS = 5
TIME_LIMIT = 10.0


def _row(size: int, seed: int, res: alg.SolveResult, inst) -> dict:
    return {
        "n_vehicles": size,
        "seed": seed,
        "algorithm": res.name,
        "solved": res.solved,
        "nodes": res.nodes,
        "time_s": res.time_s,
        "conflicts": res.conflicts,
        "travel_km": res.travel_km if math.isfinite(res.travel_km) else -1.0,
        "wait_min": res.wait_min if math.isfinite(res.wait_min) else -1.0,
        "cost": res.cost if math.isfinite(res.cost) else -1.0,
        "n_stations": len(inst.stations),
        "avg_domain_size": sum(len(d) for d in inst.domain.values()) / inst.n_vars,
    }


def run_one(size: int, seed: int, stations) -> list[dict]:
    inst = make_instance(size, seed=seed, stations=stations)
    rows: list[dict] = []
    for name, solver in alg.ALGORITHMS.items():
        print(f"        - {name:<25s}", end="", flush=True)
        if name == "Min-Conflicts":
            res = solver(inst, time_limit=TIME_LIMIT, seed=seed)
        else:
            res = solver(inst, time_limit=TIME_LIMIT)
        print(f"solved={res.solved!s:<5s}  nodes={res.nodes:>7d}  t={res.time_s:5.2f}s  "
              f"travel={res.travel_km if math.isfinite(res.travel_km) else -1:6.1f}km  "
              f"wait={res.wait_min if math.isfinite(res.wait_min) else -1:6.1f}m  "
              f"J={res.cost if math.isfinite(res.cost) else -1:7.1f}",
              flush=True)
        rows.append(_row(size, seed, res, inst))
    return rows


def main() -> None:
    print("[exp] loading stations from OSM cache")
    stations = load_stations_from_osm(seed=0)
    print(f"[exp] {len(stations)} stations loaded")
    all_rows: list[dict] = []
    for size in SIZES:
        for r in range(REPEATS):
            seed = 1000 + size * 10 + r
            print(f"[exp] size={size} seed={seed}")
            t0 = time.perf_counter()
            rows = run_one(size, seed, stations)
            all_rows.extend(rows)
            print(f"        done in {time.perf_counter() - t0:.2f}s — "
                  f"{sum(1 for r in rows if r['solved'])}/{len(rows)} solved",
                  flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "experiments.csv", index=False)
    print(f"[save] {RESULTS / 'experiments.csv'}")
    summary = (df.groupby(["algorithm", "n_vehicles"])
                 .agg(time_s=("time_s", "mean"),
                      nodes=("nodes", "mean"),
                      conflicts=("conflicts", "mean"),
                      travel_km=("travel_km", "mean"),
                      wait_min=("wait_min", "mean"),
                      cost=("cost", "mean"),
                      solved_rate=("solved", "mean"))
                 .reset_index())
    summary.to_csv(RESULTS / "summary.csv", index=False)
    print(f"[save] {RESULTS / 'summary.csv'}")
    _make_plots(df)


def _make_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "Backtracking": "#ef4444",
        "BT + MRV": "#22c55e",
        "BT + LCV": "#f59e0b",
        "BT + Forward Checking": "#3b82f6",
        "Min-Conflicts": "#a855f7",
    }

    def _line(metric, ylabel, log, fname):
        fig, ax = plt.subplots(figsize=(11, 6))
        d = df.copy()
        if metric in {"cost", "travel_km", "wait_min"}:
            d = d[d[metric] > 0]
        sns.lineplot(data=d, x="n_vehicles", y=metric, hue="algorithm",
                     palette=palette, marker="o", errorbar="sd", ax=ax)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Number of vehicles N")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs problem size")
        ax.legend(title="", loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(FIGS / fname, dpi=160)
        plt.close(fig)

    _line("time_s", "Wall-clock time (s)", True, "runtime_vs_size.png")
    _line("nodes", "Nodes expanded", True, "nodes_vs_size.png")
    _line("travel_km", "Total travel distance (km)", False, "travel_vs_size.png")
    _line("wait_min", "Total wait time (min)", False, "wait_vs_size.png")
    _line("cost", "COP objective J", False, "cost_vs_size.png")

    # solved rate as bar grid
    fig, ax = plt.subplots(figsize=(11, 6))
    sr = df.groupby(["algorithm", "n_vehicles"]).solved.mean().reset_index()
    sns.lineplot(data=sr, x="n_vehicles", y="solved", hue="algorithm",
                 palette=palette, marker="o", ax=ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Number of vehicles N")
    ax.set_ylabel("Solved fraction")
    ax.set_title("Success rate vs problem size")
    fig.tight_layout()
    fig.savefig(FIGS / "solved_rate.png", dpi=160)
    plt.close(fig)
    print(f"[save] 6 figures in {FIGS}")


if __name__ == "__main__":
    main()
