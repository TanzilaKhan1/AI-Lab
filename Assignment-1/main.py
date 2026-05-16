"""End-to-end demo on the WHOLE DHAKA CITY graph.

Loads drivable streets for 'Dhaka, Bangladesh', fetches 6 POI categories, builds
a factor-based RiskModel, runs A* with 3 heuristics, and saves maps with
exploration heatmap + route + POI markers.
"""
import os
import random
import time

import networkx as nx

from graph_loader import (
    load_graph, poi_points, FACTOR_TAGS, SAFETY_NAMES, DHAKA_PLACE,
)
from risk_model import RiskModel, DEFAULT_FACTORS
from heuristics import h_euclidean, h_haversine, h_risk_weighted
from astar import astar
from runner import run, format_report
from visualize import plot_search

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def pick_endpoints(G, min_hops=200, seed=7, attempts=800):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    for _ in range(attempts):
        s, g = rng.choice(nodes), rng.choice(nodes)
        if s == g:
            continue
        try:
            if nx.shortest_path_length(G, s, g) >= min_hops:
                return s, g
        except nx.NetworkXNoPath:
            continue
    return rng.choice(nodes), rng.choice(nodes)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()

    print(f"Loading Dhaka graph from OSM: {DHAKA_PLACE!r} ...")
    G = load_graph(DHAKA_PLACE)
    print(f"Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    print("Fetching POI categories:")
    factors = []
    safety_pts, hazard_pts = [], []
    for spec in DEFAULT_FACTORS:
        tkey, tval = FACTOR_TAGS[spec.name]
        pts = poi_points(DHAKA_PLACE, tkey, tval)
        spec.points = pts
        factors.append(spec)
        print(f"  {spec.name:15s} sigma={spec.polarity:+d}  w={spec.weight:.2f}  "
              f"R={spec.radius_m:.0f}m  count={len(pts):>5d}")
        (safety_pts if spec.name in SAFETY_NAMES else hazard_pts).extend(pts)

    print("Scoring risk for every node ...")
    rm = RiskModel(G, factors, seed=42, r_base=0.20,
                   hazard_fraction=0.03, hazard_risk=0.50, tau=1.0)
    avg_risk = sum(rm.risk.values()) / len(rm.risk)
    print(f"Mean node risk: {avg_risk:.3f}  (over {len(rm.risk):,} nodes)")

    start, goal = pick_endpoints(G, min_hops=250, seed=7)
    print(f"Endpoints: start={start}  goal={goal}")

    heuristics = {
        "euclidean":     h_euclidean(G, rm),
        "haversine":     h_haversine(G, rm),
        "risk_weighted": h_risk_weighted(G, rm, risk_weight=2.0),
    }

    print("=" * 70)
    print(f"A* on full Dhaka  --  same start/goal, 3 heuristics")
    print("=" * 70)
    results = {}
    for name, h in heuristics.items():
        res = run(astar, G, rm, start, goal, heuristic=h,
                  alpha=1.5, time_multiplier=1.0, sim_seed=0)
        results[name] = res
        print(format_report(f"A* / {name}", res))
        png = os.path.join(OUT_DIR, f"astar_{name}.png")
        plot_search(G, res,
                    pois_safety=safety_pts, pois_hazard=hazard_pts,
                    save_path=png,
                    title=f"A*  -  h = {name}   (expanded={res.expanded}, "
                          f"depth={res.depth}, cost={res.cost:,.0f} m)")
        print(f"  -> map: {png}\n")

    print("=" * 70)
    print("Simulator sweep (same Euclidean path, vary alpha):")
    print("=" * 70)
    for alpha in (0.5, 1.5, 3.0):
        r = run(astar, G, rm, start, goal,
                heuristic=heuristics["euclidean"],
                alpha=alpha, time_multiplier=1.0, sim_seed=0)
        print(format_report(f"A* / euclidean / alpha={alpha}", r))

    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
