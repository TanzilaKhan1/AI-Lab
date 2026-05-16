"""Three heuristic functions h(n, goal) -> meters.

All three are factories: they capture (G, risk_model) and return a callable so
the search algorithms stay agnostic.

    h_euclidean     planar distance (lat/lon -> meters via flat-earth approx)     admissible
    h_haversine     great-circle distance                                         admissible
    h_risk_weighted Euclidean inflated by node risk                               NOT admissible
"""
import math
from risk_model import haversine


def h_euclidean(G, risk_model=None):
    def h(n, goal):
        y1, x1 = G.nodes[n]["y"], G.nodes[n]["x"]
        y2, x2 = G.nodes[goal]["y"], G.nodes[goal]["x"]
        mid_lat = math.radians((y1 + y2) / 2)
        dy = (y2 - y1) * 111_320.0
        dx = (x2 - x1) * 111_320.0 * math.cos(mid_lat)
        return math.hypot(dx, dy)
    return h


def h_haversine(G, risk_model=None):
    def h(n, goal):
        y1, x1 = G.nodes[n]["y"], G.nodes[n]["x"]
        y2, x2 = G.nodes[goal]["y"], G.nodes[goal]["x"]
        return haversine(y1, x1, y2, x2)
    return h


def h_risk_weighted(G, risk_model, risk_weight=0.5):
    """Inadmissible: inflates Euclidean by (1 + w * risk(n)) to steer around hazards."""
    base = h_euclidean(G)
    def h(n, goal):
        return base(n, goal) * (1.0 + risk_weight * risk_model.node_risk(n))
    return h
