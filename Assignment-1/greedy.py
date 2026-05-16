"""Greedy Best-First Search: expand the node with the smallest h(n).

Fast but NOT optimal --- ignores the accumulated cost g entirely.
"""
import heapq
import itertools
import time

from cost_model import distance_cost
from metrics import SearchResult


def greedy(G, start, goal, heuristic, edge_cost=distance_cost):
    t0 = time.perf_counter()

    counter = itertools.count()
    parent = {start: None}
    heap = [(heuristic(start, goal), next(counter), start)]
    visited = {start}
    expansion_order = {}

    expanded = 0
    generated = 1
    max_frontier = 1

    while heap:
        max_frontier = max(max_frontier, len(heap))
        _, _, u = heapq.heappop(heap)
        expansion_order[u] = expanded
        expanded += 1

        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur); cur = parent[cur]
            path.reverse()
            cost = sum(edge_cost(G, a, b) for a, b in zip(path[:-1], path[1:]))
            return SearchResult(
                path=path, cost=cost, expanded=expanded, generated=generated,
                max_frontier=max_frontier,
                runtime_sec=time.perf_counter() - t0,
                depth=len(path) - 1, found=True,
                extra={"expansion_order": expansion_order},
            )

        for v in G.successors(u):
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            heapq.heappush(heap, (heuristic(v, goal), next(counter), v))
            generated += 1

    return SearchResult(
        path=[], cost=float("inf"), expanded=expanded, generated=generated,
        max_frontier=max_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=0, found=False,
        extra={"expansion_order": expansion_order},
    )
