"""Iterative Deepening Search: repeated depth-limited DFS.

Uses the textbook tree-search DLS with path-ancestor cycle checking: a node
can be re-expanded on different branches during the same iteration, but not
twice on the same path. This is what makes IDS complete (for finite b) and
optimal in hop count.

Optimal in hop count (same as BFS) but with DFS-level memory. The price is
repeated regeneration of shallow nodes. On a city-scale graph (solution depth
~250-400 hops and b~3-4) the exponential per-iteration work means IDS
typically does not finish within a sensible time budget --- that is the
comparison point.
"""
import time

from cost_model import distance_cost
from metrics import SearchResult


def ids(G, start, goal, edge_cost=distance_cost,
        max_depth=500, time_limit_sec=30.0):
    t0 = time.perf_counter()
    total_expanded = 0
    total_generated = 0
    peak_frontier = 0
    last_order = {}
    last_iter = 0
    timed_out = False

    for limit in range(max_depth + 1):
        last_iter = limit
        remaining = time_limit_sec - (time.perf_counter() - t0)
        if remaining <= 0:
            timed_out = True
            break

        deadline = time.perf_counter() + remaining
        result = _dls(G, start, goal, limit, edge_cost, deadline=deadline)

        total_expanded += result["expanded"]
        total_generated += result["generated"]
        peak_frontier = max(peak_frontier, result["max_frontier"])
        last_order = result["order"]

        if result["found"]:
            return SearchResult(
                path=result["path"], cost=result["cost"],
                expanded=total_expanded, generated=total_generated,
                max_frontier=peak_frontier,
                runtime_sec=time.perf_counter() - t0,
                depth=len(result["path"]) - 1, found=True,
                extra={"expansion_order": result["order"],
                       "iterations": limit},
            )

        if result["timed_out"]:
            timed_out = True
            break

    return SearchResult(
        path=[], cost=float("inf"),
        expanded=total_expanded, generated=total_generated,
        max_frontier=peak_frontier,
        runtime_sec=time.perf_counter() - t0,
        depth=0, found=False,
        extra={"expansion_order": last_order, "iterations": last_iter,
               "timed_out": timed_out},
    )


def _dls(G, start, goal, limit, edge_cost, *, deadline=None):
    """Tree-search depth-limited DFS with path-ancestor cycle check.

    A node is blocked only if it already lies on the current DFS path (cycle
    avoidance). Siblings and distant branches may revisit it, which is what
    keeps IDS complete under a depth cutoff.
    """
    if start == goal:
        return {"found": True, "path": [start], "cost": 0.0,
                "expanded": 1, "generated": 1, "max_frontier": 1,
                "order": {start: 0}, "timed_out": False}

    expansion_order = {start: 0}
    expanded = 1
    generated = 1
    max_frontier = 1

    current_path = [start]
    on_path = {start}
    stack = [(start, 0, iter(G.successors(start)))]

    while stack:
        if deadline is not None and time.perf_counter() >= deadline:
            return {"found": False, "path": [], "cost": float("inf"),
                    "expanded": expanded, "generated": generated,
                    "max_frontier": max_frontier, "order": expansion_order,
                    "timed_out": True}

        u, du, succ_iter = stack[-1]
        v = next(succ_iter, None)

        if v is None:
            stack.pop()
            on_path.discard(u)
            current_path.pop()
            continue

        generated += 1

        if v == goal:
            path = current_path + [v]
            cost = sum(edge_cost(G, a, b) for a, b in zip(path[:-1], path[1:]))
            return {"found": True, "path": path, "cost": cost,
                    "expanded": expanded, "generated": generated,
                    "max_frontier": max_frontier, "order": expansion_order,
                    "timed_out": False}

        if v in on_path:
            continue
        if du + 1 > limit:
            continue

        expansion_order[v] = expanded
        expanded += 1
        on_path.add(v)
        current_path.append(v)
        stack.append((v, du + 1, iter(G.successors(v))))
        if len(stack) > max_frontier:
            max_frontier = len(stack)

    return {"found": False, "path": [], "cost": float("inf"),
            "expanded": expanded, "generated": generated,
            "max_frontier": max_frontier, "order": expansion_order,
            "timed_out": False}
