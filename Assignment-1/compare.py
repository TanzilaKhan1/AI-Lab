"""Run all algorithms on the same Dhaka problem, save maps, print a table.

Uninformed (no heuristic):  BFS, DFS, UCS, IDS
Informed (heuristic-driven): Greedy, A*, Weighted A*

The three heuristics -- Euclidean, Haversine, Risk-weighted -- only apply to
informed searches, so each informed algorithm is run three times (once per
heuristic). Total runs = 4 + 3*3 = 13.

Risk-model parameters are pinned to the same values as main.py so the two
drivers are reproducible against each other.
"""
import os

from graph_loader import (
    load_graph, poi_points, FACTOR_TAGS, SAFETY_NAMES, DHAKA_PLACE,
)
from risk_model import RiskModel, DEFAULT_FACTORS
from heuristics import h_euclidean, h_haversine, h_risk_weighted
from astar import astar
from weighted_astar import weighted_astar
from bfs import bfs
from dfs import dfs
from ucs import ucs
from ids import ids
from greedy import greedy
from runner import run, format_report
from visualize import plot_search
from main import pick_endpoints

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def _safe(label):
    return label.replace("*", "star").replace("/", "_").lower()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading Dhaka graph + POIs ({DHAKA_PLACE}) ...")
    G = load_graph(DHAKA_PLACE)

    factors = []
    safety_pts, hazard_pts = [], []
    for spec in DEFAULT_FACTORS:
        tkey, tval = FACTOR_TAGS[spec.name]
        spec.points = poi_points(DHAKA_PLACE, tkey, tval)
        factors.append(spec)
        (safety_pts if spec.name in SAFETY_NAMES else hazard_pts).extend(spec.points)

    rm = RiskModel(G, factors, seed=42, r_base=0.20,
                   hazard_fraction=0.03, hazard_risk=0.50, tau=1.0)
    start, goal = pick_endpoints(G, min_hops=250, seed=7)

    heuristics = {
        "euclidean":     h_euclidean(G, rm),
        "haversine":     h_haversine(G, rm),
        "risk_weighted": h_risk_weighted(G, rm, risk_weight=2.0),
    }

    uninformed = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("IDS", ids)]
    informed   = [("Greedy", greedy), ("A*", astar), ("WA*", weighted_astar)]

    print(f"Problem: start={start}  goal={goal}")
    print("=" * 72)

    rows = []

    for name, fn in uninformed:
        res = run(fn, G, rm, start, goal,
                  alpha=1.5, time_multiplier=1.0, sim_seed=0)
        rows.append((name, res))
        print(format_report(name, res))
        if res.found:
            png = os.path.join(OUT_DIR, f"{_safe(name)}.png")
            plot_search(G, res, pois_safety=safety_pts, pois_hazard=hazard_pts,
                        save_path=png,
                        title=f"{name}  (expanded={res.expanded:,}, "
                              f"depth={res.depth}, g={res.cost:,.0f} m)")

    for name, fn in informed:
        for hname, h in heuristics.items():
            label = f"{name}/{hname}"
            res = run(fn, G, rm, start, goal, heuristic=h,
                      alpha=1.5, time_multiplier=1.0, sim_seed=0)
            rows.append((label, res))
            print(format_report(label, res))
            if res.found:
                png = os.path.join(OUT_DIR, f"{_safe(label)}.png")
                plot_search(G, res, pois_safety=safety_pts, pois_hazard=hazard_pts,
                            save_path=png,
                            title=f"{label}  (expanded={res.expanded:,}, "
                                  f"depth={res.depth}, g={res.cost:,.0f} m)")

    print("\nSummary:")
    hdr = (f"{'Algo / heuristic':<22} {'Depth':>6} {'g-cost':>12} {'actual':>12} "
           f"{'Expanded':>10} {'MaxFront':>9} {'Time':>8} {'EBF':>6}")
    print(hdr); print("-" * len(hdr))
    for label, r in rows:
        if not r.found:
            print(f"{label:<22} NOT FOUND (expanded={r.expanded:,}, "
                  f"t={r.runtime_sec:.2f}s)")
            continue
        ebf = r.extra.get("ebf", 0.0)
        actual = r.extra.get("actual_cost", 0.0)
        print(f"{label:<22} {r.depth:>6} {r.cost:>12,.0f} {actual:>12,.0f} "
              f"{r.expanded:>10,} {r.max_frontier:>9,} "
              f"{r.runtime_sec:>7.2f}s {ebf:>6.3f}")


if __name__ == "__main__":
    main()
