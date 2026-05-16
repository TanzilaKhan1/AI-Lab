"""Breadth-First Search.

Finds the path with the fewest hops (optimal in hop count, not in distance).
The reported g-cost is the sum of edge lengths along the BFS-found path, so
the number is comparable to UCS / A*.
"""
from collections import deque
import time

from cost_model import distance_cost
from metrics import SearchResult


def bfs(G, start, goal, edge_cost=distance_cost):
    t0 = time.perf_counter()
    if start == goal:
        return _pack([start], 0.0, 0, 1, 1, t0, {start: 0}, True)

    parent = {start: None}
    frontier = deque([start])
    visited = {start}
    expansion_order = {}
    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        u = frontier.popleft()
        expansion_order[u] = expanded
        expanded += 1

        for v in G.successors(u):
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            generated += 1
            if v == goal:
                path = _reconstruct(parent, v)
                cost = _path_cost(G, path, edge_cost)
                expansion_order[v] = expanded
                return _pack(path, cost, expanded, generated, max_frontier,
                             t0, expansion_order, True)
            frontier.append(v)

    return _pack([], float("inf"), expanded, generated, max_frontier, t0,
                 expansion_order, False)


def _reconstruct(parent, end):
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _path_cost(G, path, edge_cost):
    return sum(edge_cost(G, u, v) for u, v in zip(path[:-1], path[1:]))


def _pack(path, cost, expanded, generated, max_frontier, t0, order, found):
    return SearchResult(
        path=path, cost=cost, expanded=expanded, generated=generated,
        max_frontier=max_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=max(0, len(path) - 1), found=found,
        extra={"expansion_order": order},
    )
