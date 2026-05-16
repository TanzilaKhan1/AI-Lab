"""Shared SearchResult container + Effective Branching Factor."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchResult:
    path: List[int]
    cost: float              # g-cost of solution (sum of distance_cost over edges)
    expanded: int            # number of nodes popped / expanded
    generated: int           # number of nodes ever pushed onto the frontier
    max_frontier: int        # peak frontier size (space usage proxy)
    runtime_sec: float
    depth: int               # number of edges in the solution
    found: bool
    extra: dict = field(default_factory=dict)


def effective_branching_factor(expanded, depth, tol=1e-4):
    """Solve N + 1 = 1 + b* + b*^2 + ... + b*^d  for b*, via binary search.

    Interpretation: a uniform tree with branching b* and depth d would have
    visited the same number of nodes as this search.
    """
    if depth <= 0 or expanded <= 0:
        return 0.0
    N = expanded

    def total_nodes(b):
        if abs(b - 1.0) < 1e-9:
            return float(depth + 1)
        try:
            return (b ** (depth + 1) - 1.0) / (b - 1.0)
        except OverflowError:
            return float("inf")

    lo = 1.0
    # b* is bounded above by N**(1/d); add a small margin so hi brackets the root.
    hi = max(2.0, N ** (1.0 / max(depth, 1)) + 1.0)
    while total_nodes(hi) < N + 1 and hi < 1e3:
        hi *= 2
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if total_nodes(mid) < N + 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
