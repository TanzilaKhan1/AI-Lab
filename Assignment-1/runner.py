"""Uniform runner + text report.

Works for both informed (needs `heuristic`) and uninformed (no heuristic) search
algorithms, so the driver in main.py doesn't care which is which.
"""
from cost_model import simulate_actual_cost
from metrics import effective_branching_factor


def run(algorithm_fn, G, risk_model, start, goal, *,
        heuristic=None, alpha=1.5, time_multiplier=1.0,
        sim_seed=0, **kwargs):
    if heuristic is not None:
        result = algorithm_fn(G, start, goal, heuristic, **kwargs)
    else:
        result = algorithm_fn(G, start, goal, **kwargs)

    if result.found:
        actual, _ = simulate_actual_cost(
            G, risk_model, result.path,
            alpha=alpha, time_multiplier=time_multiplier, seed=sim_seed,
        )
        result.extra["actual_cost"] = actual
        result.extra["ebf"] = effective_branching_factor(result.expanded, result.depth)
    return result


def format_report(name, result):
    if not result.found:
        return (f"[{name}] NO PATH FOUND   "
                f"(expanded={result.expanded}, runtime={result.runtime_sec:.3f}s)")
    ebf = result.extra.get("ebf", 0.0)
    actual = result.extra.get("actual_cost", 0.0)
    return (
        f"[{name}]\n"
        f"  depth (hops)            : {result.depth}\n"
        f"  path g-cost  (meters)   : {result.cost:10,.1f}\n"
        f"  actual cost  (sim, m)   : {actual:10,.1f}\n"
        f"  nodes expanded          : {result.expanded}\n"
        f"  nodes generated         : {result.generated}\n"
        f"  max frontier size       : {result.max_frontier}\n"
        f"  runtime (seconds)       : {result.runtime_sec:.4f}\n"
        f"  effective branching (b*): {ebf:.3f}\n"
    )
