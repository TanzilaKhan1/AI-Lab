"""Factor-based node risk for Dhaka.

Each Factor has:
    - polarity sigma (+1 hazard, -1 safety)
    - weight w
    - influence radius R (meters)
    - list of POI points (lat, lon)

Contribution of factor k to node n (linear-falloff kernel):

    c_k(n) = sigma_k * w_k * max(0, 1 - d_k(n) / R_k)

Final risk:

    r(n) = clip( (r_base + SUM_k c_k(n) + hazard_bump(n)) * tau , 0 , 1 )

Edge risk = average of endpoint node risks.

All node scoring is vectorized with numpy (flat-earth equirectangular distance),
so scoring a whole-city graph (~10^5 nodes x ~10^3 POIs) takes a couple of seconds.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters. Kept here so heuristics.py can import it."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class Factor:
    name: str
    polarity: int                  # +1 hazard, -1 safety
    weight: float                  # 0..1
    radius_m: float                # influence radius
    points: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon) POIs


# Dhaka-tuned defaults. Override per-call if you need a different mix.
DEFAULT_FACTORS = [
    Factor("police",         -1, 0.30, 500),
    Factor("hospital",       -1, 0.25, 400),
    Factor("fire_station",   -1, 0.15, 500),
    Factor("bus_station",    +1, 0.25, 400),
    Factor("marketplace",    +1, 0.20, 300),
    Factor("level_crossing", +1, 0.30, 150),
]


def _min_flat_distance(node_coords, factor_points):
    """N-node x M-POI distance matrix, take min per node. Returns (N,) array of meters."""
    if len(factor_points) == 0:
        return np.full(len(node_coords), np.inf)
    fc = np.asarray(factor_points)
    lat1 = node_coords[:, 0:1]
    lon1 = node_coords[:, 1:2]
    lat2 = fc[:, 0].reshape(1, -1)
    lon2 = fc[:, 1].reshape(1, -1)
    mid = np.radians((lat1 + lat2) * 0.5)
    dy = (lat1 - lat2) * 111_320.0
    dx = (lon1 - lon2) * 111_320.0 * np.cos(mid)
    return np.hypot(dx, dy).min(axis=1)


class RiskModel:
    def __init__(self, G, factors, *, seed=42, r_base=0.20, tau=1.0,
                 hazard_fraction=0.03, hazard_risk=0.50):
        self.G = G
        self.factors = factors
        self.r_base = r_base
        self.tau = tau
        self.hazard_risk = hazard_risk

        rng = random.Random(seed)
        node_ids = list(G.nodes())
        n_haz = max(1, int(hazard_fraction * len(node_ids)))
        self.hazards = set(rng.sample(node_ids, n_haz))

        coords = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in node_ids])
        r = np.full(len(node_ids), r_base, dtype=float)
        self._per_factor_contrib = {}           # diagnostic, kept for report

        for f in factors:
            d = _min_flat_distance(coords, f.points)
            contrib = f.polarity * f.weight * np.maximum(0.0, 1.0 - d / f.radius_m)
            r += contrib
            self._per_factor_contrib[f.name] = float(np.mean(contrib))

        is_hazard = np.array([n in self.hazards for n in node_ids])
        r += is_hazard * hazard_risk
        r *= tau
        r = np.clip(r, 0.0, 1.0)

        self.risk = dict(zip(node_ids, r.tolist()))

    def node_risk(self, n):
        return self.risk.get(n, self.r_base)

    def edge_risk(self, u, v):
        return 0.5 * (self.node_risk(u) + self.node_risk(v))
