"""Edge cost functions.

    distance_cost(G, u, v)        -> meters. The raw g-cost used by every search.
    simulate_actual_cost(...)     -> what the user would *really* experience walking
                                     the found path, given risk + traffic + noise.

The search algorithms only ever see `distance_cost`.  `simulate_actual_cost` is
applied post-hoc to the solution to judge path quality in the comparison report.
"""
import random


def distance_cost(G, u, v):
    """Meters along the shortest parallel edge from u to v."""
    data = G.get_edge_data(u, v)
    if data is None:
        return float("inf")
    # MultiDiGraph: keys -> attrs dicts
    return min(d.get("length", float("inf")) for d in data.values())


def simulate_actual_cost(G, risk_model, path, *, alpha=1.5,
                         time_multiplier=1.0, seed=None):
    """Synthesise the cost a user would actually experience on `path`.

    For each edge (u, v):
        actual(u,v) = distance(u,v) * (1 + alpha*risk(u,v)) * T * noise
        noise  ~ Uniform(0.9, 1.15)

    Returns (total_actual_cost, per_edge_details).
    """
    rng = random.Random(seed)
    total = 0.0
    details = []
    for u, v in zip(path[:-1], path[1:]):
        base = distance_cost(G, u, v)
        risk = risk_model.edge_risk(u, v)
        noise = rng.uniform(0.9, 1.15)
        actual = base * (1 + alpha * risk) * time_multiplier * noise
        total += actual
        details.append({"u": u, "v": v, "base_m": base, "risk": risk,
                        "noise": noise, "actual_m": actual})
    return total, details
