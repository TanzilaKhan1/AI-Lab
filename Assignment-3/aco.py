"""
The optimizer: Ant Colony Optimization (ACO), implemented from scratch.

Biological analogy (kept in the comments throughout):
  - Real ants find short safe paths by laying pheromone; short successful
    paths are walked more often, gather more pheromone, and recruit more
    ants. Evaporation forgets stale paths.
  - Here each "ant" proposes one full evacuation plan (a route per group).
    Pheromone accumulates on road segments used by fast, successful plans,
    so the colony learns which corridors stay ahead of the fire.
  - The heuristic term steers ants toward shelters (instinct); pheromone is
    the colony's learned memory of corridors that worked.
"""

import numpy as np

from problem import build_route, plan_cost


class ACO:
    def __init__(self, cfg, inst, rng, record_frames=False):
        self.cfg = cfg
        self.inst = inst
        self.rng = rng
        # Pheromone per directed edge, stored in a dict keyed by (u, v).
        self.tau = {}
        for u in range(inst.n):
            for v, _ in inst.adj[u]:
                self.tau[(u, v)] = cfg.TAU0
        self.history = []      # best-so-far cost per iteration
        self.iter_best = []    # best ant cost each iteration
        self.best_routes = None
        self.best_cost = np.inf
        self.record_frames = record_frames
        self.frames = []       # (iter, tau snapshot, best routes, best cost)

    def _pick(self, cur, cands, t_now):
        """Roulette choice: p ~ tau^alpha * eta^beta, where eta estimates the
        remaining time to safety through the candidate node."""
        cfg, inst = self.cfg, self.inst
        w = np.empty(len(cands))
        for k, (v, te) in enumerate(cands):
            eta = 1.0 / (te + inst.h_time[v] + 1e-6)   # instinct: head to safety
            w[k] = (self.tau[(cur, v)] ** cfg.ALPHA) * (eta ** cfg.BETA)
        s = w.sum()
        if s <= 0:
            k = int(self.rng.integers(len(cands)))
        else:
            k = int(self.rng.choice(len(cands), p=w / s))
        return cands[k]

    def optimize(self):
        cfg, inst = self.cfg, self.inst
        for it in range(cfg.ITERATIONS):
            it_best_routes, it_best_cost = None, np.inf
            for _ant in range(cfg.N_ANTS):
                routes = [build_route(inst, cfg, o, self._pick)
                          for o in inst.origins]
                cost = plan_cost(routes, inst, cfg)
                if cost < it_best_cost:
                    it_best_cost, it_best_routes = cost, routes
            self.iter_best.append(it_best_cost)
            if it_best_cost < self.best_cost:
                self.best_cost, self.best_routes = it_best_cost, it_best_routes

            # Evaporation: the colony slowly forgets stale corridors.
            for key in self.tau:
                self.tau[key] *= (1.0 - cfg.RHO)

            # Deposit on the best plan's successful routes: fast, safe plans
            # lay more pheromone, recruiting future ants to those corridors.
            deposit = cfg.Q_DEPOSIT / max(self.best_cost, 1e-6)
            for route in self.best_routes:
                if route is None or route[-1] not in inst.shelter_set:
                    continue
                for u, v in zip(route[:-1], route[1:]):
                    self.tau[(u, v)] += deposit
                    self.tau[(v, u)] += deposit

            self.history.append(self.best_cost)
            if self.record_frames and it % max(1, cfg.ITERATIONS
                                               // cfg.ANIM_FRAMES) == 0:
                self.frames.append((it, dict(self.tau),
                                    [None if r is None else list(r)
                                     for r in self.best_routes],
                                    self.best_cost))
        return self.best_routes, self.best_cost
