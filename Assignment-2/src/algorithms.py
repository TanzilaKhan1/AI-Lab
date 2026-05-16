"""Five CSP/COP solvers for the Dhaka fuel allocation problem.

  1. Basic Backtracking
  2. Backtracking + MRV   (Minimum Remaining Values, dynamic)
  3. Backtracking + LCV   (Least Constraining Value-ordering)
  4. Backtracking + Forward Checking
  5. Min-Conflicts        (local search; cost-aware tie-breaking)

Each solver returns a SolveResult with runtime, nodes expanded, conflicts
left, total travel distance, total wait minutes and the COP objective J.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable

from .problem import (
    Instance,
    SERVICE_MINUTES,
    cost,
    count_conflicts,
    travel_km,
    wait_minutes,
)


@dataclass
class SolveResult:
    name: str
    solved: bool
    assignment: dict[str, tuple[str, int]]
    nodes: int
    time_s: float
    conflicts: int
    travel_km: float
    wait_min: float
    cost: float


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _capacity_table(instance: Instance) -> dict[tuple[str, int], int]:
    caps: dict[tuple[str, int], int] = {}
    for s in instance.stations:
        for t in s.open_slots:
            caps[(s.sid, t)] = s.cap(t)
    return caps


def _is_consistent(value: tuple[str, int],
                   assignment: dict[str, tuple[str, int]],
                   caps: dict[tuple[str, int], int]) -> bool:
    used = 1
    for st in assignment.values():
        if st == value:
            used += 1
    return used <= caps.get(value, 0)


def _finalise(name: str, solved: bool, instance: Instance,
              assignment: dict[str, tuple[str, int]], nodes: int,
              elapsed: float) -> SolveResult:
    if solved:
        return SolveResult(
            name=name, solved=True, assignment=dict(assignment),
            nodes=nodes, time_s=elapsed,
            conflicts=count_conflicts(instance, assignment),
            travel_km=travel_km(instance, assignment),
            wait_min=wait_minutes(instance, assignment),
            cost=cost(instance, assignment),
        )
    return SolveResult(
        name=name, solved=False, assignment={},
        nodes=nodes, time_s=elapsed,
        conflicts=int(1e9),
        travel_km=math.inf, wait_min=math.inf, cost=math.inf,
    )


# ---------------------------------------------------------------------------
# 1. Basic backtracking
# ---------------------------------------------------------------------------

def backtracking(instance: Instance, time_limit: float = 60.0) -> SolveResult:
    caps = _capacity_table(instance)
    order = [v.vid for v in instance.vehicles]
    domain = {vid: list(instance.domain[vid]) for vid in order}
    assignment: dict[str, tuple[str, int]] = {}
    nodes = [0]
    aborted = [False]
    start = time.perf_counter()
    deadline = start + time_limit

    def recurse(i: int) -> bool:
        if time.perf_counter() > deadline:
            aborted[0] = True
            return False
        if aborted[0]:
            return False
        if i == len(order):
            return True
        vid = order[i]
        for val in domain[vid]:
            nodes[0] += 1
            if _is_consistent(val, assignment, caps):
                assignment[vid] = val
                if recurse(i + 1):
                    return True
                del assignment[vid]
        return False

    solved = recurse(0) and not aborted[0]
    return _finalise("Backtracking", solved, instance, assignment, nodes[0],
                     time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 2. Backtracking + MRV
# ---------------------------------------------------------------------------

def backtracking_mrv(instance: Instance, time_limit: float = 60.0) -> SolveResult:
    caps = _capacity_table(instance)
    domain = {v.vid: list(instance.domain[v.vid]) for v in instance.vehicles}
    assignment: dict[str, tuple[str, int]] = {}
    nodes = [0]
    aborted = [False]
    start = time.perf_counter()
    deadline = start + time_limit

    def remaining(vid: str) -> int:
        return sum(1 for val in domain[vid] if _is_consistent(val, assignment, caps))

    def recurse() -> bool:
        if time.perf_counter() > deadline:
            aborted[0] = True
            return False
        if aborted[0]:
            return False
        unassigned = [v.vid for v in instance.vehicles if v.vid not in assignment]
        if not unassigned:
            return True
        vid = min(unassigned, key=lambda x: (remaining(x), -len(domain[x])))
        for val in domain[vid]:
            nodes[0] += 1
            if _is_consistent(val, assignment, caps):
                assignment[vid] = val
                if recurse():
                    return True
                del assignment[vid]
        return False

    solved = recurse() and not aborted[0]
    return _finalise("BT + MRV", solved, instance, assignment, nodes[0],
                     time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 3. Backtracking + LCV
# ---------------------------------------------------------------------------

def backtracking_lcv(instance: Instance, time_limit: float = 60.0) -> SolveResult:
    caps = _capacity_table(instance)
    order = [v.vid for v in instance.vehicles]
    domain = {vid: list(instance.domain[vid]) for vid in order}
    assignment: dict[str, tuple[str, int]] = {}
    nodes = [0]
    aborted = [False]
    start = time.perf_counter()
    deadline = start + time_limit

    value_load: dict[tuple[str, int], int] = {}
    for d in domain.values():
        for val in d:
            value_load[val] = value_load.get(val, 0) + 1

    def lcv_order(vid: str) -> list[tuple[str, int]]:
        return sorted(domain[vid], key=lambda val: value_load.get(val, 0))

    def recurse(i: int) -> bool:
        if time.perf_counter() > deadline:
            aborted[0] = True
            return False
        if aborted[0]:
            return False
        if i == len(order):
            return True
        vid = order[i]
        for val in lcv_order(vid):
            nodes[0] += 1
            if _is_consistent(val, assignment, caps):
                assignment[vid] = val
                if recurse(i + 1):
                    return True
                del assignment[vid]
        return False

    solved = recurse(0) and not aborted[0]
    return _finalise("BT + LCV", solved, instance, assignment, nodes[0],
                     time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 4. Backtracking + Forward Checking
# ---------------------------------------------------------------------------

def backtracking_fc(instance: Instance, time_limit: float = 60.0) -> SolveResult:
    caps = _capacity_table(instance)
    order = [v.vid for v in instance.vehicles]
    domain = {vid: list(instance.domain[vid]) for vid in order}
    assignment: dict[str, tuple[str, int]] = {}
    nodes = [0]
    aborted = [False]
    start = time.perf_counter()
    deadline = start + time_limit

    def forward_check(curr_val: tuple[str, int], current_idx: int,
                      cur_domain: dict[str, list[tuple[str, int]]]
                      ) -> dict[str, list[tuple[str, int]]] | None:
        occ = 0
        for st in assignment.values():
            if st == curr_val:
                occ += 1
        cap = caps.get(curr_val, 0)
        new_dom = {k: list(v) for k, v in cur_domain.items()}
        if occ >= cap:
            for j in range(current_idx + 1, len(order)):
                vid = order[j]
                if curr_val in new_dom[vid]:
                    new_dom[vid] = [v for v in new_dom[vid] if v != curr_val]
                if not new_dom[vid]:
                    return None
        return new_dom

    def recurse(i: int, cur_domain: dict[str, list[tuple[str, int]]]) -> bool:
        if time.perf_counter() > deadline:
            aborted[0] = True
            return False
        if aborted[0]:
            return False
        if i == len(order):
            return True
        vid = order[i]
        for val in cur_domain[vid]:
            nodes[0] += 1
            if _is_consistent(val, assignment, caps):
                assignment[vid] = val
                pruned = forward_check(val, i, cur_domain)
                if pruned is not None and recurse(i + 1, pruned):
                    return True
                del assignment[vid]
        return False

    solved = recurse(0, domain) and not aborted[0]
    return _finalise("BT + Forward Checking", solved, instance, assignment,
                     nodes[0], time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 5. Min-Conflicts local search — cost-aware tie-breaking
# ---------------------------------------------------------------------------

def min_conflicts(instance: Instance, max_steps: int = 20_000,
                  time_limit: float = 60.0, seed: int = 0) -> SolveResult:
    rng = random.Random(seed)
    caps = _capacity_table(instance)
    vb = instance.vehicle_by_id
    sb = instance.station_by_id
    domain = instance.domain
    assignment: dict[str, tuple[str, int]] = {}
    occ: dict[tuple[str, int], int] = {}
    nodes = 0
    start = time.perf_counter()
    deadline = start + time_limit

    # Greedy initial assignment: minimise (overflow + distance) for each pick.
    for v in instance.vehicles:
        best = min(
            domain[v.vid],
            key=lambda val: (
                max(0, occ.get(val, 0) + 1 - caps.get(val, 0)),
                _km(vb[v.vid], sb[val[0]]),
            ),
        )
        assignment[v.vid] = best
        occ[best] = occ.get(best, 0) + 1
        nodes += 1

    def conflicted_vars() -> list[str]:
        return [vid for vid, val in assignment.items()
                if occ[val] > caps.get(val, 0)]

    def score(vid: str, val: tuple[str, int]) -> tuple[int, float]:
        old = assignment[vid]
        new_overflow = max(0, occ.get(val, 0) + 1 - caps.get(val, 0))
        if val == old:
            new_overflow = max(0, occ[old] - caps.get(old, 0))
        # how many leftover violations remain on the *old* cell after the move
        old_left = max(0, occ[old] - 1 - caps.get(old, 0)) if old != val else new_overflow
        # cost-aware tie-break: distance + wait proxy
        dist = _km(vb[vid], sb[val[0]])
        new_occ_after = occ.get(val, 0) + (0 if val == old else 1)
        wait_proxy = max(0, new_occ_after - caps.get(val, 0)) * 4.0
        return (new_overflow + old_left, dist + wait_proxy)

    for _ in range(max_steps):
        if time.perf_counter() > deadline:
            break
        bad = conflicted_vars()
        if not bad:
            # already feasible — keep optimising cost
            vid = rng.choice([v.vid for v in instance.vehicles])
        else:
            vid = rng.choice(bad)
        old_val = assignment[vid]
        cands = list(domain[vid])
        rng.shuffle(cands)
        best_val = old_val
        best_score = score(vid, old_val)
        for val in cands:
            nodes += 1
            s = score(vid, val)
            if s < best_score:
                best_score = s
                best_val = val
        if best_val != old_val:
            occ[old_val] -= 1
            occ[best_val] = occ.get(best_val, 0) + 1
            assignment[vid] = best_val

    elapsed = time.perf_counter() - start
    conflicts = count_conflicts(instance, assignment)
    return SolveResult(
        name="Min-Conflicts",
        solved=conflicts == 0,
        assignment=dict(assignment),
        nodes=nodes,
        time_s=elapsed,
        conflicts=conflicts,
        travel_km=travel_km(instance, assignment),
        wait_min=wait_minutes(instance, assignment),
        cost=cost(instance, assignment),
    )


def _km(v, s):
    # local haversine import to avoid circular reference
    from .problem import haversine_km
    return haversine_km(v.lat, v.lon, s.lat, s.lon)


ALGORITHMS: dict[str, Callable[..., SolveResult]] = {
    "Backtracking": backtracking,
    "BT + MRV": backtracking_mrv,
    "BT + LCV": backtracking_lcv,
    "BT + Forward Checking": backtracking_fc,
    "Min-Conflicts": min_conflicts,
}
