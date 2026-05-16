"""Dhaka fuel allocation CSP/COP — problem formulation + instance generator.

================================================================
Real-world scenario
================================================================
During the Bangladesh 2026 fuel crisis (Iran–US war supply shock),
the Bangladesh Petroleum Corporation (BPC) launched a digital **Fuel-Pass**
to ration petroleum. Each pump operates with reduced throughput, vehicles
have a class-specific weekly quota, and queues stretched 1+ km at some
stations. We model the daily dispatch of N vehicles to M stations.

================================================================
Formal model
================================================================

  Variables (X)
    x_v  ∈  S × {0,…,T-1}                       ∀ v ∈ vehicles

  Domain (D)
    (s,t) ∈ D(x_v) iff
        fuel(v)   ∈ types(s)            – fuel-type match
        dist(v,s) ≤ R_v                 – reachability radius (km)
        t         ∈ open(s)             – station open in slot t
        t         ∈ window(v)           – vehicle available in slot t
        slot t ≥ priority-floor(v)      – emergency vehicles can pre-empt

  Constraints (C)
    C1 (hard) capacity:    Σ 1[x_v=(s,t)]  ≤  cap(s,t)         ∀ s,t
    C2 (soft) quota:       fill_amount(v) ≤ weekly_quota(v)    ∀ v
    C3 (soft) priority:    emergency v should land in low-congestion slot

  CSP variant: hard-enforce C1; report a feasible assignment.
  COP variant: minimise

        J(x)  =  α · Σ_v  travel_km(v, S(x_v))
              +  β · Σ_v  wait_minutes(x_v)
              +  γ · Σ_{s,t} max(0, n(s,t) − cap(s,t))
              +  δ · Σ_{v∈em} (n(S(x_v),T(x_v)) − 1)_+

      where wait_minutes(x_v) = position-in-queue × service_minutes.
      We use weights (α,β,γ,δ) = (1.0, 2.0, 25.0, 10.0); γ is heavy so the
      optimiser still rejects infeasibility.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --- BPC March 2026 policy parameters
QUOTAS_WEEKLY = {"motorcycle": 8, "three_wheeler": 20, "car": 25, "emergency": 9999}
PER_FILL_LIMIT = {"motorcycle": 2, "three_wheeler": 5, "car": 10, "emergency": 50}
VEHICLE_MIX = [("motorcycle", 0.55), ("car", 0.30), ("three_wheeler", 0.12), ("emergency", 0.03)]
FUEL_TYPE_FOR = {"motorcycle": "octane", "car": "octane", "three_wheeler": "cng", "emergency": "diesel"}
SERVICE_MINUTES = 8        # mean service time per vehicle once at the pump

# --- crisis-day operating regime
SLOTS_PER_DAY = 6                            # 6 × 2-h windows, 06:00–18:00
SLOT_HOURS = 2
DEFAULT_PUMPS_PER_STATION = (1, 3)           # tight crisis caps
THROUGHPUT_PER_PUMP_PER_SLOT = 1             # 1 vehicle per pump per slot
N_ACTIVE_STATIONS = 18                       # active subset of 248 OSM stations


@dataclass(frozen=True)
class Station:
    sid: str
    name: str
    lat: float
    lon: float
    pumps: int
    fuel_types: tuple[str, ...]
    open_slots: tuple[int, ...]

    def cap(self, slot: int) -> int:
        return self.pumps * THROUGHPUT_PER_PUMP_PER_SLOT if slot in self.open_slots else 0


@dataclass(frozen=True)
class Vehicle:
    vid: str
    vclass: str
    fuel: str
    lat: float
    lon: float
    radius_km: float
    window: tuple[int, ...]
    priority: int           # 0 = normal, 1 = emergency


@dataclass
class Instance:
    vehicles: list[Vehicle]
    stations: list[Station]
    domain: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return len(self.vehicles)

    @property
    def station_by_id(self) -> dict[str, Station]:
        return {s.sid: s for s in self.stations}

    @property
    def vehicle_by_id(self) -> dict[str, Vehicle]:
        return {v.vid: v for v in self.vehicles}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _sample_class(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for cls, p in VEHICLE_MIX:
        cum += p
        if r <= cum:
            return cls
    return VEHICLE_MIX[-1][0]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_stations_from_osm(min_pumps: int = DEFAULT_PUMPS_PER_STATION[0],
                           max_pumps: int = DEFAULT_PUMPS_PER_STATION[1],
                           n_active: int = N_ACTIVE_STATIONS,
                           seed: int = 0) -> list[Station]:
    csv = DATA / "dhaka_fuel_stations.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Run src/fetch_osm.py first — missing {csv}")
    df = pd.read_csv(csv)
    rng = random.Random(seed)
    if n_active < len(df):
        df = df.sample(n=n_active, random_state=seed).reset_index(drop=True)
    stations: list[Station] = []
    for _, row in df.iterrows():
        pumps = rng.randint(min_pumps, max_pumps)
        types = ["octane", "diesel"] + (["cng"] if rng.random() < 0.55 else [])
        # crisis-day: ~70% of stations skip the early-morning slot
        all_slots = list(range(SLOTS_PER_DAY))
        n_closed = rng.randint(0, 1)
        closed = set(rng.sample(all_slots, n_closed))
        open_slots = tuple(t for t in all_slots if t not in closed)
        name_val = row.get("name") if isinstance(row.get("name"), str) else None
        if name_val is None:
            brand = row.get("brand") if isinstance(row.get("brand"), str) else None
            name_val = brand or f"Station-{row['station_id']}"
        stations.append(Station(
            sid=str(row["station_id"]),
            name=str(name_val),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            pumps=pumps,
            fuel_types=tuple(types),
            open_slots=open_slots,
        ))
    return stations


def generate_vehicles(n: int, stations: list[Station], seed: int = 0) -> list[Vehicle]:
    rng = random.Random(seed)
    lats = [s.lat for s in stations]
    lons = [s.lon for s in stations]
    lat_lo, lat_hi = min(lats), max(lats)
    lon_lo, lon_hi = min(lons), max(lons)
    vehicles: list[Vehicle] = []
    for i in range(n):
        cls = _sample_class(rng)
        lat = rng.uniform(lat_lo, lat_hi)
        lon = rng.uniform(lon_lo, lon_hi)
        radius = rng.uniform(2.0, 5.0)
        win_len = rng.randint(2, 5)
        win_start = rng.randint(0, SLOTS_PER_DAY - win_len)
        window = tuple(range(win_start, win_start + win_len))
        priority = 1 if cls == "emergency" else 0
        vehicles.append(Vehicle(
            vid=f"V{i:03d}", vclass=cls, fuel=FUEL_TYPE_FOR[cls],
            lat=lat, lon=lon, radius_km=radius, window=window, priority=priority,
        ))
    return vehicles


def build_domains(vehicles: list[Vehicle],
                  stations: list[Station]) -> dict[str, list[tuple[str, int]]]:
    domain: dict[str, list[tuple[str, int]]] = {}
    for v in vehicles:
        opts: list[tuple[str, int]] = []
        for s in stations:
            if v.fuel not in s.fuel_types:
                continue
            d = haversine_km(v.lat, v.lon, s.lat, s.lon)
            if d > v.radius_km:
                continue
            for t in s.open_slots:
                if t in v.window:
                    opts.append((s.sid, t))
        domain[v.vid] = opts
    return domain


def make_instance(n_vehicles: int, seed: int = 0,
                  stations: list[Station] | None = None) -> Instance:
    if stations is None:
        stations = load_stations_from_osm(seed=seed)
    served = set().union(*[set(s.fuel_types) for s in stations])
    if "cng" not in served:
        stations = [Station(s.sid, s.name, s.lat, s.lon, s.pumps,
                            tuple(set(s.fuel_types) | ({"cng"} if i % 3 == 0 else set())),
                            s.open_slots)
                    for i, s in enumerate(stations)]
    oversample = int(n_vehicles * 1.5) + 4
    vehicles = generate_vehicles(oversample, stations, seed=seed)
    domain = build_domains(vehicles, stations)
    fixed: list[Vehicle] = []
    for v in vehicles:
        if domain[v.vid]:
            fixed.append(v)
        else:
            fixed.append(Vehicle(
                vid=v.vid, vclass=v.vclass, fuel=v.fuel, lat=v.lat, lon=v.lon,
                radius_km=25.0, window=tuple(range(SLOTS_PER_DAY)), priority=v.priority,
            ))
    vehicles = fixed
    domain = build_domains(vehicles, stations)
    survivors = [v for v in vehicles if domain[v.vid]][:n_vehicles]
    if len(survivors) < n_vehicles:
        pad = []
        i = 0
        while len(survivors) + len(pad) < n_vehicles:
            src = survivors[i % len(survivors)]
            pad.append(Vehicle(
                vid=f"V{(len(survivors)+len(pad)):03d}",
                vclass=src.vclass, fuel=src.fuel, lat=src.lat, lon=src.lon,
                radius_km=src.radius_km, window=src.window, priority=src.priority,
            ))
            i += 1
        survivors = survivors + pad
    renamed = [Vehicle(vid=f"V{i:03d}", vclass=v.vclass, fuel=v.fuel,
                       lat=v.lat, lon=v.lon, radius_km=v.radius_km,
                       window=v.window, priority=v.priority)
               for i, v in enumerate(survivors)]
    domain = build_domains(renamed, stations)
    return Instance(vehicles=renamed, stations=stations, domain=domain)


# ---------------------------------------------------------------------------
# COP metrics
# ---------------------------------------------------------------------------

def occupancy(assignment: dict[str, tuple[str, int]]) -> dict[tuple[str, int], int]:
    occ: dict[tuple[str, int], int] = {}
    for st in assignment.values():
        occ[st] = occ.get(st, 0) + 1
    return occ


def count_conflicts(instance: Instance, assignment: dict[str, tuple[str, int]]) -> int:
    sb = instance.station_by_id
    occ = occupancy(assignment)
    return sum(max(0, n - sb[sid].cap(t)) for (sid, t), n in occ.items())


def travel_km(instance: Instance, assignment: dict[str, tuple[str, int]]) -> float:
    sb = instance.station_by_id
    vb = instance.vehicle_by_id
    return sum(haversine_km(vb[vid].lat, vb[vid].lon, sb[sid].lat, sb[sid].lon)
               for vid, (sid, _) in assignment.items())


def wait_minutes(instance: Instance, assignment: dict[str, tuple[str, int]]) -> float:
    """Mean wait inside the slot, assuming each pump serves one vehicle per
    SERVICE_MINUTES and the queue forms FIFO. Vehicle in position k waits
    (k − 1) × service_minutes / pumps."""
    sb = instance.station_by_id
    queue: dict[tuple[str, int], int] = {}
    total = 0.0
    for sid, t in assignment.values():
        pumps = max(sb[sid].pumps, 1)
        queue[(sid, t)] = queue.get((sid, t), 0) + 1
        position = queue[(sid, t)]
        total += (position - 1) * SERVICE_MINUTES / pumps
    return total


def cost(instance: Instance, assignment: dict[str, tuple[str, int]],
         alpha: float = 1.0, beta: float = 2.0,
         gamma: float = 25.0, delta: float = 10.0) -> float:
    sb = instance.station_by_id
    vb = instance.vehicle_by_id
    occ = occupancy(assignment)
    travel = sum(haversine_km(vb[vid].lat, vb[vid].lon, sb[sid].lat, sb[sid].lon)
                 for vid, (sid, _) in assignment.items())
    overflow = sum(max(0, n - sb[sid].cap(t)) for (sid, t), n in occ.items())
    wait = wait_minutes(instance, assignment)
    em_pen = sum(max(0, occ[(sid, t)] - 1)
                 for vid, (sid, t) in assignment.items()
                 if vb[vid].priority == 1)
    return alpha * travel + beta * wait + gamma * overflow + delta * em_pen
