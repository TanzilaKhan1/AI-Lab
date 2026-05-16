"""Depth-First Search (iterative, stack-based).

Not optimal. Marks a node as visited when it is pushed onto the stack, so a
node is expanded at most once. A `depth_limit` can be supplied; at no limit,
DFS follows any reachable branch until dead-end or goal.
"""
import time

from cost_model import distance_cost
from metrics import SearchResult


def dfs(G, start, goal, edge_cost=distance_cost, depth_limit=None):
    t0 = time.perf_counter()
    if start == goal:
        return _pack([start], 0.0, 0, 1, 1, t0, {start: 0}, True)

    parent = {start: None}
    depth = {start: 0}
    stack = [start]
    visited = {start}
    expansion_order = {}
    expanded = 0
    generated = 1
    max_frontier = 1

    while stack:
        max_frontier = max(max_frontier, len(stack))
        u = stack.pop()
        expansion_order[u] = expanded
        expanded += 1

        if u == goal:
            path = _reconstruct(parent, u)
            cost = sum(edge_cost(G, a, b) for a, b in zip(path[:-1], path[1:]))
            return _pack(path, cost, expanded, generated, max_frontier, t0,
                         expansion_order, True)

        if depth_limit is not None and depth[u] >= depth_limit:
            continue

        for v in G.successors(u):
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            depth[v] = depth[u] + 1
            generated += 1
            stack.append(v)

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


def _pack(path, cost, expanded, generated, max_frontier, t0, order, found):
    return SearchResult(
        path=path, cost=cost, expanded=expanded, generated=generated,
        max_frontier=max_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=max(0, len(path) - 1), found=found,
        extra={"expansion_order": order},
    )
