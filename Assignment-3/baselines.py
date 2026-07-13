"""
Baselines compared against ACO: same feasibility rule, different routing
logic. Used only to show why fire-aware and fire-blind alternatives underperform.
"""

import heapq

import numpy as np

from problem import build_route, plan_cost


def greedy_plan(inst, cfg):
    """Fire-aware greedy: at each intersection take the feasible road that
    minimizes edge time plus straight-line time to the nearest shelter."""
    def pick(cur, cands, t_now):
        return min(cands, key=lambda c: c[1] + inst.h_time[c[0]])
    routes = [build_route(inst, cfg, o, pick) for o in inst.origins]
    return routes, plan_cost(routes, inst, cfg)


def static_plan(inst, cfg):
    """Fire-BLIND shortest path: plan by travel time only, ignoring the fire,
    then face reality. Shows why static navigation fails in a wildfire."""
    def dijkstra_static(src):
        dist = np.full(inst.n, np.inf)
        prev = np.full(inst.n, -1, dtype=int)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            t, u = heapq.heappop(pq)
            if t > dist[u]:
                continue
            for v, te in inst.adj[u]:
                if t + te < dist[v]:
                    dist[v] = t + te
                    prev[v] = u
                    heapq.heappush(pq, (t + te, v))
        return dist, prev

    routes = []
    for o in inst.origins:
        dist, prev = dijkstra_static(o)
        best_s = min(inst.shelters, key=lambda s: dist[s])
        if not np.isfinite(dist[best_s]):
            routes.append(None)
            continue
        path = [int(best_s)]
        while path[-1] != o:
            path.append(int(prev[path[-1]]))
        routes.append(path[::-1])
    return routes, plan_cost(routes, inst, cfg)
