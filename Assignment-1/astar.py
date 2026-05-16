"""A* search over an OSMnx MultiDiGraph.

    f(n) = g(n) + h(n)
    g(n) = accumulated distance_cost from start
    h(n) = heuristic estimate (meters) — supplied by caller

Interface mirrored by every other search file, so you can swap algorithms:
    result = astar(G, start, goal, heuristic)
    result = bfs(G, start, goal)             # (once bfs.py exists)
"""
import heapq
import itertools
import time

from cost_model import distance_cost
from metrics import SearchResult


def astar(G, start, goal, heuristic, edge_cost=distance_cost):
    t0 = time.perf_counter()

    counter = itertools.count()            # tie-breaker so heap never compares nodes
    g_score = {start: 0.0}
    parent = {start: None}
    open_heap = [(heuristic(start, goal), next(counter), start)]
    closed = set()
    expansion_order = {}                   # node -> step index (for heatmap)

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
                         expansion_order=expansion_order)

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
                    (tentative_g + heuristic(v, goal), next(counter), v),
                )
                generated += 1

    return _done(None, parent, float("inf"), expanded, generated,
                 max_frontier, t0, found=False, f=None,
                 expansion_order=expansion_order)


def _done(end, parent, cost, expanded, generated, max_frontier, t0, *,
          found, f, expansion_order):
    path = []
    if found:
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    return SearchResult(
        path=path,
        cost=cost,
        expanded=expanded,
        generated=generated,
        max_frontier=max_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=max(0, len(path) - 1),
        found=found,
        extra={"final_f": f, "expansion_order": expansion_order}
              if f is not None else {"expansion_order": expansion_order},
    )
