"""
Problem definition for the wildfire evacuation routing task.

Holds everything that describes the world and the rules, independent of any
optimizer: the tunable Config, the road network + wind-driven fire model
(WildfireInstance), the shared feasibility/evaluation logic (plan_cost), and
the hop-by-hop route builder shared by ACO and the baselines (build_route).
"""

import heapq

import numpy as np


# ======================================================================
# CONFIG. Every tunable parameter lives here.
# ======================================================================
class Config:
    # Road network: an NX x NY grid of intersections with jittered positions.
    # A fraction of road segments is removed so the network has real dead
    # ends and detours, like an actual rural road map.
    NX = 11
    NY = 11
    SPACING = 3.0        # km between neighbouring intersections
    JITTER = 0.55        # positional jitter so the map looks organic
    ROAD_DROP = 0.22     # fraction of grid road segments that do not exist

    # Traffic
    SPEED = 40.0         # km/h evacuation driving speed

    # Fire model (wind-driven elliptical growth of the arrival-time field)
    BASE_SPREAD = 6.0    # km/h fire spread rate in still air
    WIND_STRength = 0.0  # placeholder (overwritten below; kept for clarity)
    WIND_STRENGTH = 0.65 # 0..1, how strongly wind elongates the fire ellipse
    SAFETY_MARGIN = 0.12 # hours; a segment must be cleared this long before fire

    # Scenario
    N_ORIGINS = 6        # neighbourhood groups to evacuate
    N_SHELTERS = 4       # designated safe zones
    FAIL_PENALTY = 8.0   # hours added per group that cannot reach a shelter

    # ACO parameters
    N_ANTS = 40
    ITERATIONS = 120
    ALPHA = 1.0          # pheromone influence
    BETA = 3.0           # heuristic influence (time-to-shelter estimate)
    RHO = 0.15           # evaporation rate
    Q_DEPOSIT = 10.0     # pheromone deposit constant
    TAU0 = 1.0

    # Experiment control
    SEED = 19
    N_SEEDS = 15
    CONVERGE_TOL = 1e-9
    OUT_DIR = "."

    # Animation
    ANIMATE = True
    ANIM_FRAMES = 70
    ANIM_FPS = 9


# ======================================================================
# INSTANCE: road network + wind-driven fire arrival-time field
# ======================================================================
class WildfireInstance:
    """
    Holds the road graph, the fire arrival time at every node, the origin
    groups, and the shelters. Node ids are 0..N-1 on an NX x NY grid.
    """

    def __init__(self, cfg, rng):
        self.cfg = cfg
        nx, ny = cfg.NX, cfg.NY

        # Jittered intersection coordinates.
        coords = np.zeros((nx * ny, 2))
        for iy in range(ny):
            for ix in range(nx):
                n = iy * nx + ix
                coords[n] = [ix * cfg.SPACING + rng.normal(0, cfg.JITTER),
                             iy * cfg.SPACING + rng.normal(0, cfg.JITTER)]
        self.coords = coords
        self.n = nx * ny

        # 4-neighbour road edges with travel time in hours. A random fraction
        # of segments is dropped so the map has dead ends and forced detours.
        self.adj = [[] for _ in range(self.n)]   # node -> [(nbr, time_h)]
        self.edges = []                          # [(u, v)] for drawing
        for iy in range(ny):
            for ix in range(nx):
                n = iy * nx + ix
                for dx, dy in ((1, 0), (0, 1)):
                    jx, jy = ix + dx, iy + dy
                    if jx < nx and jy < ny:
                        if rng.random() < cfg.ROAD_DROP:
                            continue             # this road does not exist
                        m = jy * nx + jx
                        d = float(np.linalg.norm(coords[n] - coords[m]))
                        t = d / cfg.SPEED
                        self.adj[n].append((m, t))
                        self.adj[m].append((n, t))
                        self.edges.append((n, m))

        # Wind direction (unit vector) and ignition on the upwind side, so
        # the fire blows across the town and threatens the road network.
        theta = rng.uniform(0, 2 * np.pi)
        self.wind = np.array([np.cos(theta), np.sin(theta)])
        center = coords.mean(axis=0)
        span = cfg.SPACING * max(nx, ny)
        self.ignition = center - self.wind * span * 0.32 \
            + rng.normal(0, cfg.SPACING * 0.4, size=2)

        # Fire arrival time at each node: distance / directional spread rate.
        # Downwind (cos=+1) the fire moves up to (1+W)x faster; upwind it is
        # slowed to (1-W)x. This gives the classic wind-stretched ellipse.
        self.t_fire = np.zeros(self.n)
        for i in range(self.n):
            vec = coords[i] - self.ignition
            dist = np.linalg.norm(vec) + 1e-9
            cosang = float(vec @ self.wind) / dist
            rate = cfg.BASE_SPREAD * (1.0 + cfg.WIND_STRENGTH * cosang)
            rate = max(rate, cfg.BASE_SPREAD * (1.0 - cfg.WIND_STRENGTH) * 0.5)
            self.t_fire[i] = dist / rate

        # Origins: nodes threatened soon (small fire-arrival time, but not
        # already burning); shelters: nodes the fire reaches last.
        order = np.argsort(self.t_fire)
        early = order[self.n // 20: self.n // 4]          # threatened band
        late = order[-max(cfg.N_SHELTERS * 4, 8):]        # safest band
        self.origins = list(rng.choice(early, size=cfg.N_ORIGINS, replace=False))
        self.shelters = list(rng.choice(late, size=cfg.N_SHELTERS, replace=False))
        self.shelter_set = set(int(s) for s in self.shelters)

        # Straight-line time-to-nearest-shelter estimate per node (heuristic).
        sh = coords[self.shelters]
        d2sh = np.sqrt(((coords[:, None, :] - sh[None, :, :]) ** 2).sum(axis=2))
        self.h_time = d2sh.min(axis=1) / cfg.SPEED

    # -- feasibility rule shared by every method (ants and baselines) -----
    def can_cross(self, u, v, t_now, t_edge):
        """A group may drive u -> v starting at t_now only if it leaves u
        before the fire arrives at u and reaches v before the fire arrives
        at v, both with the safety margin."""
        m = self.cfg.SAFETY_MARGIN
        return (t_now <= self.t_fire[u] - m and
                t_now + t_edge <= self.t_fire[v] - m)

    def fire_aware_dijkstra(self, src):
        """Earliest arrival time at every node respecting the fire rule.
        Used only to validate that the instance is solvable."""
        dist = np.full(self.n, np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            t, u = heapq.heappop(pq)
            if t > dist[u]:
                continue
            for v, te in self.adj[u]:
                if self.can_cross(u, v, t, te) and t + te < dist[v]:
                    dist[v] = t + te
                    heapq.heappush(pq, (t + te, v))
        return dist

    def solvable(self):
        """Every origin must be able to reach some shelter ahead of the fire."""
        for o in self.origins:
            d = self.fire_aware_dijkstra(o)
            if not any(np.isfinite(d[s]) for s in self.shelters):
                return False
        return True


def make_instance(cfg, rng):
    """Resample until the scenario is survivable (it almost always is)."""
    for _ in range(60):
        inst = WildfireInstance(cfg, rng)
        if inst.solvable():
            return inst
    raise RuntimeError("could not generate a solvable wildfire scenario")


# ======================================================================
# PLAN EVALUATION (separate from the optimizer)
# ======================================================================
def plan_cost(routes, inst, cfg):
    """
    Total evacuation time over all groups (hours). A route is a node list
    starting at its origin; it succeeds if it ends at a shelter. Failed or
    missing routes add FAIL_PENALTY each. Lower is better.
    """
    total = 0.0
    for route in routes:
        if route is None or route[-1] not in inst.shelter_set:
            total += cfg.FAIL_PENALTY
            continue
        t = 0.0
        ok = True
        for u, v in zip(route[:-1], route[1:]):
            te = next(tt for (m, tt) in inst.adj[u] if m == v)
            if not inst.can_cross(u, v, t, te):
                ok = False
                break
            t += te
        total += t if ok else cfg.FAIL_PENALTY
    return total


# ======================================================================
# ROUTE CONSTRUCTION shared by ACO and the greedy baseline
# ======================================================================
def build_route(inst, cfg, origin, pick_next, max_hops=400):
    """
    Build one group's route hop by hop from `origin`. At each intersection
    only fire-feasible, not-yet-visited roads are candidates; `pick_next`
    chooses among them. Returns the node list, or None on a dead end.
    """
    route = [origin]
    visited = {origin}
    t = 0.0
    cur = origin
    for _ in range(max_hops):
        if cur in inst.shelter_set:
            return route
        cands = [(v, te) for (v, te) in inst.adj[cur]
                 if v not in visited and inst.can_cross(cur, v, t, te)]
        if not cands:
            return None                       # trapped: fire cut the options
        v, te = pick_next(cur, cands, t)
        route.append(v)
        visited.add(v)
        t += te
        cur = v
    return None
