"""Uniform-Cost Search (a.k.a. Dijkstra on the road graph).

Optimal in g-cost (summed edge length) since all edge costs are non-negative.
Equivalent to A* with h == 0.
"""
import heapq
import itertools
import time

from cost_model import distance_cost
from metrics import SearchResult


def ucs(G, start, goal, edge_cost=distance_cost):
    t0 = time.perf_counter()

    counter = itertools.count()
    g_score = {start: 0.0}
    parent = {start: None}
    heap = [(0.0, next(counter), start)]
    closed = set()
    expansion_order = {}

    expanded = 0
    generated = 1
    max_frontier = 1

    while heap:
        max_frontier = max(max_frontier, len(heap))
        g, _, u = heapq.heappop(heap)

        if u in closed:
            continue
        if u == goal:
            expansion_order[u] = expanded
            return _done(u, parent, g, expanded, generated, max_frontier,
                         t0, expansion_order, True)

        closed.add(u)
        expansion_order[u] = expanded
        expanded += 1

        for v in G.successors(u):
            if v in closed:
                continue
            tentative = g + edge_cost(G, u, v)
            if tentative < g_score.get(v, float("inf")):
                g_score[v] = tentative
                parent[v] = u
                heapq.heappush(heap, (tentative, next(counter), v))
                generated += 1

    return _done(None, parent, float("inf"), expanded, generated,
                 max_frontier, t0, expansion_order, False)


def _done(end, parent, cost, expanded, generated, max_frontier, t0,
          expansion_order, found):
    path = []
    if found:
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return SearchResult(
        path=path, cost=cost, expanded=expanded, generated=generated,
        max_frontier=max_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=max(0, len(path) - 1), found=found,
        extra={"expansion_order": expansion_order},
    )
