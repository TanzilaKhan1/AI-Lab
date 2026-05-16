"""Weighted A*: f(n) = g(n) + w * h(n), w >= 1.

At w = 1 this is standard A*. At w > 1 the heuristic is amplified, so the
search becomes more goal-directed and usually faster, but it loses the
optimality guarantee even when h is admissible (theory.tex sec. 1424-1448:
"Weighted A* ... may lose optimality, because the heuristic is being
intentionally amplified"). At large w it approaches Greedy Best-First.
"""
import heapq
import itertools
import time

from cost_model import distance_cost
from metrics import SearchResult


def weighted_astar(G, start, goal, heuristic, edge_cost=distance_cost,
                   weight=1.5):
    t0 = time.perf_counter()

    counter = itertools.count()
    g_score = {start: 0.0}
    parent = {start: None}
    open_heap = [(weight * heuristic(start, goal), next(counter), start)]
    closed = set()
    expansion_order = {}

    expanded = 0
    generated = 1
    max_frontier = 1

    while open_heap:
        max_frontier = max(max_frontier, len(open_heap))
        f, _, u = heapq.heappop(open_heap)

        if u in closed:
            continue
        if u == goal:
            expansion_order[u] = expanded
            return _done(u, parent, g_score[u], expanded, generated,
                         max_frontier, t0, found=True, f=f,
                         expansion_order=expansion_order, weight=weight)

        closed.add(u)
        expansion_order[u] = expanded
        expanded += 1

        for v in G.successors(u):
            if v in closed:
                continue
            tentative_g = g_score[u] + edge_cost(G, u, v)
            if tentative_g < g_score.get(v, float("inf")):
                g_score[v] = tentative_g
                parent[v] = u
                heapq.heappush(
                    open_heap,
                    (tentative_g + weight * heuristic(v, goal),
                     next(counter), v),
                )
                generated += 1

    return _done(None, parent, float("inf"), expanded, generated,
                 max_frontier, t0, found=False, f=None,
                 expansion_order=expansion_order, weight=weight)


def _done(end, parent, cost, expanded, generated, max_frontier, t0, *,
          found, f, expansion_order, weight):
    path = []
    if found:
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    extra = {"expansion_order": expansion_order, "weight": weight}
    if f is not None:
        extra["final_f"] = f
    return SearchResult(
        path=path,
        cost=cost,
        expanded=expanded,
        generated=generated,
        max_frontier=max_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=max(0, len(path) - 1),
        found=found,
        extra=extra,
    )
