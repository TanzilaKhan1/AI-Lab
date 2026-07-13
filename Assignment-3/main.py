"""
Problem: Wildfire evacuation routing over a fire-consuming road network.
Method:  Ant Colony Optimization (ACO), implemented from scratch.

A wildfire ignites near a wildland-urban-interface town and spreads outward,
faster downwind, slower upwind (an elliptical, wind-driven growth pattern in
the spirit of Huygens-type fire-spread models). Neighbourhood groups must
drive from their homes to designated shelters over a road network whose
segments become impassable as the fire front reaches them. A route is only
valid if every road segment on it is crossed BEFORE the fire arrives there,
with a safety margin. The goal is to get every group to a shelter and to
minimize the total evacuation time.

This mirrors real wildfire evacuations at the wildland-urban interface
(California, Australia, Canada), where route choice under a moving fire front
decides outcomes. The exact combination (ACO over an outdoor road network
consumed by a wind-driven fire-arrival field) was flagged as a literature gap
in the accompanying literature review.

This module wires the pieces together: problem.py (Config + world model),
aco.py (the optimizer), baselines.py (comparison methods), and plots.py
(figures + animation). Run:  python main.py

Only numpy, matplotlib, and scipy are used.
"""

import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

from problem import Config, make_instance
from aco import ACO
from baselines import greedy_plan, static_plan
from plots import (plot_evacuation_map, plot_pheromone, plot_convergence,
                    animate_evacuation)


def iterations_to_converge(history, tol):
    final = history[-1]
    for i, h in enumerate(history):
        if h - final <= tol:
            return i + 1
    return len(history)


def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.SEED)
    inst = make_instance(cfg, rng)

    # ---- Detailed run for plots and the summary ----
    aco = ACO(cfg, inst, np.random.default_rng(cfg.SEED),
              record_frames=cfg.ANIMATE)
    best_routes, best_cost = aco.optimize()

    g_routes, g_cost = greedy_plan(inst, cfg)
    s_routes, s_cost = static_plan(inst, cfg)

    conv_iter = iterations_to_converge(aco.history, cfg.CONVERGE_TOL)
    n_ok = sum(1 for r in best_routes
               if r is not None and r[-1] in inst.shelter_set)

    print("=" * 20)
    print("ACO WILDFIRE EVACUATION ROUTING")
    print("=" * 20)
    print(f"Road intersections / segments: {inst.n} / {len(inst.edges)}")
    print(f"Groups to evacuate:            {cfg.N_ORIGINS}   "
          f"Shelters: {cfg.N_SHELTERS}")
    print(f"Fire spread (still air):       {cfg.BASE_SPREAD:.0f} km/h, "
          f"wind factor {cfg.WIND_STRENGTH}")
    print(f"Ants x iterations:             {cfg.N_ANTS} x {cfg.ITERATIONS}")
    print("-" * 20)
    print(f"ACO best plan: {n_ok}/{cfg.N_ORIGINS} groups safe, "
          f"total {best_cost:.3f} h")
    for gi, route in enumerate(best_routes):
        if route is None:
            print(f"   Group {gi + 1}: TRAPPED")
        else:
            print(f"   Group {gi + 1}: {len(route) - 1} road segments to "
                  f"shelter node {route[-1]}")
    print(f"ACO iterations to converge:    {conv_iter} of {cfg.ITERATIONS}")
    print("-" * 20)
    print("SIDE-BY-SIDE (same scenario, same feasibility rule):")
    print(f"   {'Method':<28}{'Total time (h, penalized)':>26}")
    print(f"   {'ACO':<28}{best_cost:>26.3f}")
    print(f"   {'Fire-aware greedy':<28}{g_cost:>26.3f}")
    print(f"   {'Fire-blind shortest path':<28}{s_cost:>26.3f}")
    gain_g = 100.0 * (g_cost - best_cost) / g_cost
    gain_s = 100.0 * (s_cost - best_cost) / s_cost
    print(f"   ACO better than greedy by:      {gain_g:.1f}%")
    print(f"   ACO better than fire-blind by:  {gain_s:.1f}%")

    plot_evacuation_map(inst, cfg, best_routes, best_cost)
    plot_pheromone(inst, cfg, aco.tau)
    plot_convergence(aco.history, aco.iter_best, g_cost, s_cost, cfg)
    print("-" * 20)
    print("Plots written: evacuation_map.png, pheromone.png, convergence.png")
    if cfg.ANIMATE:
        animate_evacuation(inst, cfg, best_routes, best_cost)
        print("Animation written: evacuation_animation.gif")

    # ---- Multi-seed significance test ----
    print("-" * 20)
    print(f"SIGNIFICANCE TEST across {cfg.N_SEEDS} independent scenarios:")
    aco_scores, greedy_scores = [], []
    for s in range(cfg.N_SEEDS):
        rng_s = np.random.default_rng(3000 + s)
        inst_s = make_instance(cfg, rng_s)
        _, ca = ACO(cfg, inst_s, np.random.default_rng(3000 + s)).optimize()
        _, cg = greedy_plan(inst_s, cfg)
        aco_scores.append(ca)
        greedy_scores.append(cg)
    aco_scores = np.array(aco_scores)
    greedy_scores = np.array(greedy_scores)
    print(f"   ACO             : mean={aco_scores.mean():.3f}  "
          f"std={aco_scores.std():.3f}  best={aco_scores.min():.3f}")
    print(f"   Fire-aware greedy: mean={greedy_scores.mean():.3f}  "
          f"std={greedy_scores.std():.3f}  best={greedy_scores.min():.3f}")
    if HAVE_SCIPY:
        try:
            stat, p = wilcoxon(aco_scores, greedy_scores)
            verdict = ("significant (p < 0.05)" if p < 0.05
                       else "not significant")
            print(f"   Wilcoxon signed-rank: statistic={stat:.2f}, "
                  f"p={p:.5f}  -> {verdict}")
        except ValueError as e:
            print(f"   Wilcoxon could not run: {e}")
    else:
        print("   scipy not available; install it to run the Wilcoxon test.")
    print("=" * 20)


if __name__ == "__main__":
    main()
