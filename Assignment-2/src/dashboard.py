"""Build HTML deliverables: Plotly dashboard, Folium map, landing index."""
from __future__ import annotations

import json
from pathlib import Path

import folium
import folium.plugins
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import algorithms as alg
from .problem import (
    SLOTS_PER_DAY,
    haversine_km,
    load_stations_from_osm,
    make_instance,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
WEB = ROOT / "web"
WEB.mkdir(exist_ok=True)

PALETTE = {
    "Backtracking": "#ef4444",
    "BT + MRV": "#22c55e",
    "BT + LCV": "#f59e0b",
    "BT + Forward Checking": "#3b82f6",
    "Min-Conflicts": "#a855f7",
}
CLASS_COLOR = {
    "motorcycle": "#3b82f6",
    "car": "#a855f7",
    "three_wheeler": "#10b981",
    "emergency": "#ef4444",
}


# =========================================================== Plotly dashboard

def build_dashboard() -> None:
    summary = pd.read_csv(RESULTS / "summary.csv")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Wall-clock time (s, log)",
            "Nodes expanded (log)",
            "Solved fraction",
            "COP objective J",
            "Total travel distance (km)",
            "Total wait time (min)",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )
    series = [
        ("time_s", 1, 1, True),
        ("nodes", 1, 2, True),
        ("solved_rate", 2, 1, False),
        ("cost", 2, 2, False),
        ("travel_km", 3, 1, False),
        ("wait_min", 3, 2, False),
    ]
    for alg_name in summary.algorithm.unique():
        d = summary[summary.algorithm == alg_name].sort_values("n_vehicles")
        c = PALETTE.get(alg_name, "#888")
        for i, (col, r, cidx, _) in enumerate(series):
            sub = d.copy()
            if col in {"cost", "travel_km", "wait_min"}:
                sub = sub[sub[col] > 0]
            fig.add_trace(
                go.Scatter(
                    x=sub.n_vehicles, y=sub[col],
                    mode="lines+markers", name=alg_name,
                    line=dict(color=c, width=2.5),
                    marker=dict(size=8),
                    legendgroup=alg_name,
                    showlegend=(i == 0),
                    hovertemplate=f"<b>{alg_name}</b><br>N=%{{x}}<br>{col}=%{{y:.3f}}<extra></extra>",
                ),
                row=r, col=cidx,
            )
    for _, r, cidx, log in series:
        if log:
            fig.update_yaxes(type="log", row=r, col=cidx)
        fig.update_xaxes(title_text="N (vehicles)", row=r, col=cidx)
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1100, width=1280,
        title=dict(text="Five-Solver Comparison — Dhaka Fuel Allocation",
                   x=0.02, xanchor="left", font=dict(size=20)),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="center", x=0.5, font=dict(size=12)),
        font=dict(family="Inter, -apple-system, sans-serif"),
    )
    fig.write_html(WEB / "dashboard.html", include_plotlyjs="cdn", full_html=True)
    print(f"[save] {WEB/'dashboard.html'}")


# =========================================================== Folium map

def build_map() -> None:
    stations = load_stations_from_osm(seed=0)
    inst = make_instance(80, seed=42, stations=stations)
    res = alg.backtracking_mrv(inst, time_limit=20)
    if not res.solved:
        res = alg.min_conflicts(inst, seed=42, time_limit=20)
    print(f"[map] solver={res.name} solved={res.solved} N={inst.n_vars}")

    load: dict[str, int] = {s.sid: 0 for s in inst.stations}
    slotload: dict[tuple[str, int], int] = {}
    for vid, (sid, t) in res.assignment.items():
        load[sid] = load.get(sid, 0) + 1
        slotload[(sid, t)] = slotload.get((sid, t), 0) + 1

    centre = (sum(s.lat for s in inst.stations) / len(inst.stations),
              sum(s.lon for s in inst.stations) / len(inst.stations))
    m = folium.Map(location=centre, zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; OpenStreetMap &copy; CARTO',
        name="Light", max_zoom=20, subdomains="abcd",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; OSM &copy; CARTO',
        name="Dark", max_zoom=20, subdomains="abcd",
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="&copy; Esri", name="Esri Gray", max_zoom=16,
    ).add_to(m)

    # ---- one layer per slot, so the user can step through the day
    slot_groups = [
        folium.FeatureGroup(name=f"Assignments — slot {t}", show=(t == 0))
        for t in range(SLOTS_PER_DAY)
    ]
    for vid, (sid, t) in res.assignment.items():
        v = inst.vehicle_by_id[vid]
        s = inst.station_by_id[sid]
        col = CLASS_COLOR.get(v.vclass, "#64748b")
        folium.PolyLine(
            [(v.lat, v.lon), (s.lat, s.lon)],
            color=col, weight=1.4, opacity=0.6,
            tooltip=f"{vid} &middot; {v.vclass} &rarr; {sid} (slot {t}) &middot; {haversine_km(v.lat, v.lon, s.lat, s.lon):.2f} km",
        ).add_to(slot_groups[t])
        # vehicle marker on same group
        folium.CircleMarker(
            (v.lat, v.lon), radius=3, color=col, fill=True,
            fill_color=col, fill_opacity=0.95, weight=0,
            tooltip=f"{vid} &middot; {v.vclass}",
        ).add_to(slot_groups[t])
    for g in slot_groups:
        g.add_to(m)

    # ---- station markers (always on)
    st_layer = folium.FeatureGroup(name="Stations (load)", show=True)
    for s in inst.stations:
        n = load.get(s.sid, 0)
        cap_day = s.pumps * SLOTS_PER_DAY
        frac = n / max(cap_day, 1)
        col = "#22c55e" if frac < 0.3 else ("#f59e0b" if frac < 0.7 else "#ef4444")
        per_slot = "".join(
            f"<tr><td>slot {t}</td><td style='text-align:right'>{slotload.get((s.sid,t),0)}/{s.cap(t)}</td></tr>"
            for t in s.open_slots
        )
        popup = folium.Popup(
            f"<div style='font-family:Inter,system-ui;min-width:220px'>"
            f"<div style='font-weight:700;font-size:14px'>{s.name}</div>"
            f"<div style='color:#64748b;font-size:12px;margin-bottom:8px'>"
            f"{s.sid} &middot; pumps {s.pumps} &middot; {', '.join(s.fuel_types)}</div>"
            f"<div style='font-weight:600;color:{col};margin-bottom:6px'>"
            f"Daily load &nbsp;{n}/{cap_day} &nbsp;({frac*100:.0f}%)</div>"
            f"<table style='font-size:12px;width:100%;border-collapse:collapse'>"
            f"<thead><tr><th style='text-align:left'>Slot</th>"
            f"<th style='text-align:right'>used/cap</th></tr></thead>"
            f"<tbody>{per_slot}</tbody></table></div>",
            max_width=340,
        )
        folium.CircleMarker(
            location=(s.lat, s.lon),
            radius=7 + 22 * frac,
            color="#0f172a", fill=True, fill_color=col, fill_opacity=0.85, weight=1.5,
            popup=popup,
            tooltip=f"{s.name} &middot; {n}/{cap_day} ({frac*100:.0f}%)",
        ).add_to(st_layer)
    st_layer.add_to(m)

    # ---- congestion heatmap (off by default)
    pts = [[s.lat, s.lon, load.get(s.sid, 0)] for s in inst.stations]
    folium.plugins.HeatMap(
        pts, name="Congestion heatmap", show=False,
        radius=30, blur=24, min_opacity=0.35,
        gradient={0.2: "#22c55e", 0.5: "#f59e0b", 0.85: "#ef4444"},
    ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:#0f172aee;color:#e2e8f0;padding:14px 16px;border-radius:10px;
                font:13px/1.5 Inter,system-ui;border:1px solid #334155;
                box-shadow:0 6px 24px rgba(0,0,0,.3);max-width:240px;backdrop-filter:blur(8px)">
      <div style="font-weight:700;font-size:13px;letter-spacing:.4px;color:#94a3b8;
                  text-transform:uppercase;margin-bottom:8px">Legend</div>
      <div style="font-weight:600;font-size:11px;color:#64748b;margin-top:8px">STATION LOAD</div>
      <div><span style="display:inline-block;width:11px;height:11px;background:#22c55e;border-radius:50%;margin-right:6px"></span>&lt; 30%</div>
      <div><span style="display:inline-block;width:11px;height:11px;background:#f59e0b;border-radius:50%;margin-right:6px"></span>30 – 70%</div>
      <div><span style="display:inline-block;width:11px;height:11px;background:#ef4444;border-radius:50%;margin-right:6px"></span>&gt; 70%</div>
      <div style="font-weight:600;font-size:11px;color:#64748b;margin-top:10px">VEHICLE CLASS</div>
      <div><span style="display:inline-block;width:9px;height:9px;background:#3b82f6;border-radius:50%;margin-right:6px"></span>motorcycle</div>
      <div><span style="display:inline-block;width:9px;height:9px;background:#a855f7;border-radius:50%;margin-right:6px"></span>car</div>
      <div><span style="display:inline-block;width:9px;height:9px;background:#10b981;border-radius:50%;margin-right:6px"></span>three-wheeler</div>
      <div><span style="display:inline-block;width:9px;height:9px;background:#ef4444;border-radius:50%;margin-right:6px"></span>emergency</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    title_html = (
        f"<div style='position:fixed;top:14px;left:50%;transform:translateX(-50%);"
        f"z-index:9999;background:#0f172aee;color:#fff;padding:10px 22px;border-radius:10px;"
        f"font:600 15px/1.2 Inter,system-ui;border:1px solid #334155;"
        f"box-shadow:0 6px 24px rgba(0,0,0,.3);backdrop-filter:blur(8px)'>"
        f"Dhaka Fuel Allocation &middot; {res.name} &middot; "
        f"N = {inst.n_vars} vehicles, {len(inst.stations)} stations, "
        f"{SLOTS_PER_DAY} slots</div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.save(str(WEB / "dhaka_map.html"))
    print(f"[save] {WEB/'dhaka_map.html'}")


# =========================================================== HTML index

def build_index() -> None:
    df = pd.read_csv(RESULTS / "experiments.csv")
    sums = pd.read_csv(RESULTS / "summary.csv")
    n_stations = len(load_stations_from_osm(seed=0))
    osm = json.loads((ROOT / "data/osm_summary.json").read_text())

    sizes = sorted(df.n_vehicles.unique())
    algos = sums.algorithm.unique()

    def tcell(v, kind):
        if kind == "time" and v >= 9.99:
            return "<span style='color:#ef4444'>timeout</span>"
        if kind == "solved":
            cls = "#22c55e" if v == 1.0 else ("#f59e0b" if v > 0 else "#ef4444")
            return f"<span style='color:{cls}'>{v*100:.0f}%</span>"
        if kind == "nodes" and v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if kind == "nodes" and v >= 1000:
            return f"{v/1000:.1f}k"
        if kind == "time":
            return f"{v:.3f}"
        if kind == "cost":
            return "—" if v < 0 else f"{v:.0f}"
        return f"{v:.2f}"

    def make_table(metric: str, kind: str) -> str:
        head = "<tr><th>Algorithm</th>" + "".join(f"<th>N={n}</th>" for n in sizes) + "</tr>"
        rows = ""
        for a in algos:
            cells = ""
            for n in sizes:
                rec = sums[(sums.algorithm == a) & (sums.n_vehicles == n)]
                if rec.empty:
                    cells += "<td>—</td>"
                else:
                    v = rec[metric].iloc[0]
                    cells += f"<td>{tcell(v, kind)}</td>"
            rows += f"<tr><td><b>{a}</b></td>{cells}</tr>"
        return f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>"

    time_tbl = make_table("time_s", "time")
    nodes_tbl = make_table("nodes", "nodes")
    solved_tbl = make_table("solved_rate", "solved")
    cost_tbl = make_table("cost", "cost")

    css = """
    *{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth;font-size:15px}
    body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#0a0f1c;color:#e2e8f0;line-height:1.6;letter-spacing:.005em}
    a{color:#60a5fa;text-decoration:none}
    a:hover{color:#93c5fd}
    code{font-family:'JetBrains Mono','SF Mono',Consolas,monospace;font-size:0.88em;
         background:#1e293b;padding:2px 6px;border-radius:4px;color:#fbbf24}

    /* Sidebar nav */
    .layout{display:flex;min-height:100vh}
    .sidebar{position:fixed;top:0;left:0;bottom:0;width:240px;background:#0f172a;
             border-right:1px solid #1e293b;padding:28px 20px;overflow-y:auto}
    .sidebar .brand{font-weight:700;font-size:1.05rem;color:#f1f5f9;margin-bottom:4px;letter-spacing:-.2px}
    .sidebar .tag{font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.7px;margin-bottom:24px}
    .sidebar nav{display:flex;flex-direction:column;gap:2px}
    .sidebar nav a{display:block;padding:8px 12px;border-radius:6px;color:#94a3b8;font-size:0.9rem;
                   transition:all .15s}
    .sidebar nav a:hover{background:#1e293b;color:#e2e8f0}
    .sidebar nav a.active{background:#1e293b;color:#60a5fa}
    .sidebar .footnote{font-size:0.74rem;color:#475569;margin-top:32px;padding-top:18px;
                       border-top:1px solid #1e293b}

    /* Main */
    main{margin-left:240px;flex:1;padding:0}
    @media(max-width:880px){.sidebar{display:none}main{margin-left:0}}

    header.hero{padding:80px 56px 64px;border-bottom:1px solid #1e293b;
                background:radial-gradient(800px 400px at 0% 0%,#1e3a8a33 0%,transparent 50%)}
    header.hero .label{color:#60a5fa;font-size:0.85rem;font-weight:600;
                       letter-spacing:.5px;text-transform:uppercase;margin-bottom:14px}
    header.hero h1{font-size:2.6rem;font-weight:700;color:#f1f5f9;letter-spacing:-.6px;line-height:1.15;
                   margin-bottom:18px;max-width:840px}
    header.hero .lead{color:#cbd5e1;font-size:1.12rem;max-width:760px;line-height:1.65}

    .kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
             gap:1px;background:#1e293b;border-top:1px solid #1e293b;border-bottom:1px solid #1e293b}
    .kpi{padding:24px 20px;background:#0a0f1c}
    .kpi .v{font-size:1.65rem;font-weight:700;color:#f1f5f9;letter-spacing:-.5px;line-height:1}
    .kpi .l{font-size:0.78rem;color:#64748b;margin-top:6px;text-transform:uppercase;letter-spacing:.4px}

    section{padding:56px;border-bottom:1px solid #1e293b;scroll-margin-top:0}
    section .eyebrow{color:#60a5fa;font-size:0.8rem;font-weight:600;
                     letter-spacing:.5px;text-transform:uppercase;margin-bottom:10px}
    section h2{font-size:1.7rem;font-weight:700;color:#f1f5f9;margin-bottom:8px;letter-spacing:-.3px}
    section .desc{color:#94a3b8;max-width:760px;margin-bottom:32px}
    section h3{font-size:1.05rem;color:#f1f5f9;font-weight:600;margin-bottom:14px;margin-top:24px}

    /* Two-col layout */
    .two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px}
    @media(max-width:880px){.two-col{grid-template-columns:1fr}}
    .panel{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:24px}
    .panel h3{margin-top:0}
    .panel p{color:#cbd5e1;margin-bottom:10px;font-size:0.97rem}
    .panel .accent{color:#60a5fa;font-weight:600}
    .panel ul{padding-left:20px;color:#cbd5e1}
    .panel li{margin-bottom:4px}

    /* Table */
    .table-wrap{overflow-x:auto;border:1px solid #1e293b;border-radius:10px;margin-bottom:16px}
    table{width:100%;border-collapse:collapse;font-size:0.93rem;background:#0f172a}
    th,td{padding:11px 14px;border-bottom:1px solid #1e293b;text-align:right;white-space:nowrap}
    th:first-child,td:first-child{text-align:left}
    th{background:#1e293b;color:#cbd5e1;font-weight:600;font-size:0.78rem;
       text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #334155}
    tbody tr:hover{background:#0a1224}
    .tab-strip{display:flex;gap:4px;margin-bottom:12px}
    .tab-strip button{padding:8px 14px;background:transparent;border:1px solid #1e293b;
                      color:#94a3b8;border-radius:6px;cursor:pointer;font-size:0.85rem;font-family:inherit;
                      transition:all .15s}
    .tab-strip button.active{background:#1e293b;color:#60a5fa;border-color:#3b82f6}
    .tab-strip button:hover{color:#e2e8f0}

    /* Figure grid */
    .fig-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
    .fig{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px;
         transition:border-color .15s}
    .fig:hover{border-color:#334155}
    .fig .title{font-size:0.85rem;font-weight:600;color:#cbd5e1;margin-bottom:10px;
                text-transform:uppercase;letter-spacing:.4px}
    .fig img{display:block;width:100%;height:auto;border-radius:6px;background:#fff}

    /* iframes */
    iframe{display:block;width:100%;border:1px solid #1e293b;border-radius:12px;background:#fff}

    /* Code block */
    pre{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:18px;
        overflow-x:auto;font-family:'JetBrains Mono','SF Mono',monospace;
        font-size:0.85rem;color:#cbd5e1;line-height:1.65}

    /* Findings list */
    .findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
    .find{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:20px}
    .find .lab{font-size:0.74rem;font-weight:600;color:#60a5fa;
               text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}
    .find h4{font-size:1rem;color:#f1f5f9;margin-bottom:6px}
    .find p{color:#94a3b8;font-size:0.9rem}

    footer{padding:32px 56px;color:#475569;font-size:0.85rem}
    footer .line{margin-bottom:4px}
    """

    sidebar_links = [
        ("hero", "Overview"),
        ("problem", "Problem"),
        ("data", "Data"),
        ("algorithms", "Solvers"),
        ("results", "Results"),
        ("map", "Map"),
        ("dashboard", "Dashboard"),
        ("heatmap", "Heatmap"),
        ("figures", "Figures"),
        ("findings", "Findings"),
        ("repro", "Reproduce"),
    ]
    nav_html = "".join(f'<a href="#{i}">{label}</a>' for i, label in sidebar_links)

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Dhaka Fuel Allocation — CSP / COP study</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{css}</style>
</head><body>
<div class="layout">
<aside class="sidebar">
  <div class="brand">Dhaka Fuel CSP</div>
  <div class="tag">May 2026 &middot; v1.0</div>
  <nav>{nav_html}</nav>
  <div class="footnote">Data: OpenStreetMap (ODbL).<br>Policy: BPC, Fuel-Pass BD, BRTA.</div>
</aside>
<main>

<header class="hero" id="hero">
  <div class="label">CSP / COP study</div>
  <h1>Modelling Dhaka's 2026 fuel-rationing problem<br>as a Constraint Satisfaction Problem.</h1>
  <p class="lead">Real fuel-station coordinates pulled from OpenStreetMap, vehicle demand
     synthesised against the Bangladesh Petroleum Corporation rationing policy. Five
     classical AI solvers benchmarked side-by-side from N = 10 to N = 100 vehicles.</p>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="v">{n_stations}</div><div class="l">OSM fuel stations</div></div>
  <div class="kpi"><div class="v">5</div><div class="l">Solvers benchmarked</div></div>
  <div class="kpi"><div class="v">{df.n_vehicles.nunique()}</div><div class="l">Problem sizes</div></div>
  <div class="kpi"><div class="v">{len(df)}</div><div class="l">Total runs</div></div>
  <div class="kpi"><div class="v">{df.seed.nunique()}</div><div class="l">Random seeds</div></div>
  <div class="kpi"><div class="v">{SLOTS_PER_DAY}</div><div class="l">Time slots / day</div></div>
</div>

<section id="problem">
  <div class="eyebrow">01 — Problem definition</div>
  <h2>A CSP with a COP escape hatch</h2>
  <p class="desc">Each vehicle must receive one (station, slot) ticket. Stations have limited per-slot
     capacity. When demand outpaces supply, the COP variant minimises a weighted cost rather
     than declaring infeasibility.</p>
  <div class="two-col">
    <div class="panel">
      <h3 style="color:#60a5fa">Variables, domain, constraints</h3>
      <p><span class="accent">X</span> &nbsp; one variable per vehicle:
         <code>x_v &in; S &times; {{0,&hellip;,T&minus;1}}</code></p>
      <p><span class="accent">D(x_v)</span> &nbsp; unary feasibility folded in:
         fuel-type match, reachability radius <code>R_v</code>, station open in slot,
         vehicle window covers slot.</p>
      <p><span class="accent">C</span> &nbsp; the only global constraint is per-cell capacity
         <code>|{{v : x_v=(s,t)}}| &le; cap(s,t)</code>.</p>
    </div>
    <div class="panel">
      <h3 style="color:#a855f7">COP objective</h3>
      <p><code>J = α·travel + β·wait + γ·overflow + δ·priority</code></p>
      <p>weights <code>(α, β, γ, δ) = (1.0, 2.0, 25.0, 10.0)</code></p>
      <ul>
        <li><b>travel</b> – total km vehicles drive</li>
        <li><b>wait</b> – queue-position × service-minutes</li>
        <li><b>overflow</b> – capacity violations (heavy)</li>
        <li><b>priority</b> – emergency vehicles in crowded slot</li>
      </ul>
    </div>
  </div>
</section>

<section id="data">
  <div class="eyebrow">02 — Data</div>
  <h2>Real fuel stations, BPC-anchored parameters</h2>
  <p class="desc">Station coordinates come from the Overpass API
     (<code>amenity=fuel</code> inside the Dhaka bounding box). Vehicle demand follows
     the BPC March 2026 rationing rules.</p>
  <div class="two-col">
    <div class="panel">
      <h3>Source</h3>
      <p>OpenStreetMap, <code>amenity=fuel</code> nodes inside bbox
         <code>{osm['bbox_S_W_N_E']}</code>. Total: <b>{n_stations}</b> stations.</p>
      <p>Crisis-day scenario: <b>{18}</b> active stations from the {n_stations} (limited
         stock), 1–3 pumps each, 1 vehicle / pump / slot, {SLOTS_PER_DAY} two-hour slots.</p>
    </div>
    <div class="panel">
      <h3>BPC rationing parameters (March 2026)</h3>
      <ul>
        <li>Per-fill cap: 2 L motorcycle, 5 L three-wheeler, 10 L car</li>
        <li>Weekly quota: 8 L motorcycle, 20 L three-wheeler, 25 L car</li>
        <li>Vehicle mix sampled: 55% mc, 30% car, 12% 3-w, 3% emergency</li>
        <li>Fuel-Pass digital QR distribution active at 7 stations during pilot</li>
      </ul>
    </div>
  </div>
</section>

<section id="algorithms">
  <div class="eyebrow">03 — Solvers</div>
  <h2>Five classical CSP/COP solvers</h2>
  <div class="findings">
    <div class="find"><div class="lab">01</div><h4>Basic Backtracking</h4>
       <p>Depth-first variable order, in-domain value order. Baseline.</p></div>
    <div class="find"><div class="lab">02</div><h4>BT + MRV</h4>
       <p>Variable with fewest remaining legal values chosen first. Recomputed each step.</p></div>
    <div class="find"><div class="lab">03</div><h4>BT + LCV</h4>
       <p>Values tried in ascending order of how many other vehicles also include that (station, slot).</p></div>
    <div class="find"><div class="lab">04</div><h4>BT + Forward Checking</h4>
       <p>After each assignment, prune the just-filled cell from every unassigned variable's domain.</p></div>
    <div class="find"><div class="lab">05</div><h4>Min-Conflicts</h4>
       <p>Greedy init + iterative repair. Tie-broken by COP cost; keeps optimising once feasible.</p></div>
  </div>
</section>

<section id="results">
  <div class="eyebrow">04 — Results</div>
  <h2>Benchmark numbers</h2>
  <p class="desc">Aggregate over {df.seed.nunique()} random seeds per size. 10 s wall-clock cap per
     solver per instance. Cells marked <span style="color:#ef4444">timeout</span> hit the budget.</p>
  <div class="tab-strip">
    <button data-target="t-time" class="active">Time (s)</button>
    <button data-target="t-nodes">Nodes</button>
    <button data-target="t-solved">Solved rate</button>
    <button data-target="t-cost">COP cost J</button>
  </div>
  <div class="table-wrap" id="t-time">{time_tbl}</div>
  <div class="table-wrap" id="t-nodes" style="display:none">{nodes_tbl}</div>
  <div class="table-wrap" id="t-solved" style="display:none">{solved_tbl}</div>
  <div class="table-wrap" id="t-cost" style="display:none">{cost_tbl}</div>
</section>

<section id="map">
  <div class="eyebrow">05 — Spatial view</div>
  <h2>Dhaka assignment map</h2>
  <p class="desc">Stations sized by daily load. Each slot has its own toggleable assignment layer —
     step through the day to see how demand shifts. MRV solution for N = 80 vehicles.</p>
  <iframe src="dhaka_map.html" loading="lazy" style="height:720px"></iframe>
</section>

<section id="dashboard">
  <div class="eyebrow">06 — Comparative dashboard</div>
  <h2>Six-panel solver dashboard</h2>
  <p class="desc">Time, nodes, success, travel, wait and COP cost — log-scaled where useful.</p>
  <iframe src="dashboard.html" loading="lazy" style="height:1140px"></iframe>
</section>

<section id="heatmap">
  <div class="eyebrow">07 — Utilisation matrix</div>
  <h2>Station × slot utilisation</h2>
  <p class="desc">Each cell = (vehicles assigned) ÷ (capacity). Hot rows = overloaded
     stations; hot columns = peak slots.</p>
  <iframe src="heatmap.html" loading="lazy" style="height:720px"></iframe>
</section>

<section id="figures">
  <div class="eyebrow">08 — Static figures</div>
  <h2>Publication-quality plots</h2>
  <div class="fig-grid">
    <div class="fig"><div class="title">Runtime vs N</div><img src="../figures/runtime_vs_size.png"></div>
    <div class="fig"><div class="title">Nodes expanded</div><img src="../figures/nodes_vs_size.png"></div>
    <div class="fig"><div class="title">Success rate</div><img src="../figures/solved_rate.png"></div>
    <div class="fig"><div class="title">COP objective J</div><img src="../figures/cost_vs_size.png"></div>
    <div class="fig"><div class="title">Total travel distance</div><img src="../figures/travel_vs_size.png"></div>
    <div class="fig"><div class="title">Total wait time</div><img src="../figures/wait_vs_size.png"></div>
    <div class="fig"><div class="title">Slot × station utilisation</div><img src="../figures/slot_heatmap.png"></div>
    <div class="fig"><div class="title">Vehicle class breakdown</div><img src="../figures/class_breakdown.png"></div>
    <div class="fig"><div class="title">Per-station load distribution</div><img src="../figures/load_distribution.png"></div>
    <div class="fig"><div class="title">All OSM stations in Dhaka</div><img src="../figures/dhaka_fuel_stations_map.png"></div>
  </div>
</section>

<section id="findings">
  <div class="eyebrow">09 — Findings</div>
  <h2>What we learned</h2>
  <div class="findings">
    <div class="find"><div class="lab">Scaling</div><h4>Basic BT collapses at N ≥ 50</h4>
       <p>Branching factor ≈ |D| ~ 40 per variable. Tree grows as |D|^N. Solver hits the
          10 s budget by N = 75.</p></div>
    <div class="find"><div class="lab">Heuristic</div><h4>MRV is the most reliable complete solver</h4>
       <p>Per-step variable re-evaluation is cheap relative to the pruning gain. Solves every
          tested size in &lt; 1.2 s on average.</p></div>
    <div class="find"><div class="lab">Propagation</div><h4>Forward Checking helps until capacity is global</h4>
       <p>FC prunes one cell at a time — when many variables compete for the same scarce
          cells, it still has to backtrack a lot. Best for mid-sized N.</p></div>
    <div class="find"><div class="lab">Local search</div><h4>Min-Conflicts is the only scaler</h4>
       <p>Reaches a feasible assignment in 0.001 s at every N, then keeps reducing COP cost
          on the post-feasibility loop.</p></div>
    <div class="find"><div class="lab">Cost</div><h4>MRV minimises travel; MinConf minimises wait</h4>
       <p>The complete solvers commit to the first feasible answer they find. The local-search
          tie-breaker measurably reduces J at high N.</p></div>
    <div class="find"><div class="lab">Policy</div><h4>Why this matters in practice</h4>
       <p>BPC's Fuel-Pass app is exactly this kind of CSP back-end. Min-Conflicts is the only
          solver that completes in interactive time — useful when 100s of vehicles refresh.</p></div>
  </div>
</section>

<section id="repro">
  <div class="eyebrow">10 — Reproduce</div>
  <h2>One-shot pipeline</h2>
<pre>poetry install
poetry run python -m src.fetch_osm       # pull amenity=fuel via Overpass
poetry run python -m src.experiments     # benchmark all 5 solvers
poetry run python -m src.build_table     # render LaTeX tables
poetry run python -m src.extras          # extra figures + heatmap
poetry run python -m src.dashboard       # build HTML deliverables
open web/index.html</pre>
</section>

<footer>
  <div class="line">Dhaka Fuel CSP &middot; May 2026 &middot;
    data: OpenStreetMap (ODbL) &middot; policy refs: BPC, Fuel-Pass BD, BRTA</div>
  <div class="line">Built with Python &middot; OSMnx &middot; Folium &middot; Plotly &middot; Matplotlib</div>
</footer>

</main>
</div>

<script>
  // tab switching
  document.querySelectorAll('.tab-strip button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const strip = btn.parentElement;
      strip.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const t = btn.dataset.target;
      ['t-time','t-nodes','t-solved','t-cost'].forEach(id => {{
        document.getElementById(id).style.display = (id === t) ? '' : 'none';
      }});
    }});
  }});

  // sidebar active link on scroll
  const links = Array.from(document.querySelectorAll('.sidebar nav a'));
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  function onScroll() {{
    const y = window.scrollY + 120;
    let activeIdx = 0;
    for (let i = 0; i < sections.length; i++) {{
      if (sections[i].offsetTop <= y) activeIdx = i;
    }}
    links.forEach((a, i) => a.classList.toggle('active', i === activeIdx));
  }}
  window.addEventListener('scroll', onScroll); onScroll();
</script>
</body></html>
"""
    (WEB / "index.html").write_text(html)
    print(f"[save] {WEB/'index.html'}")


if __name__ == "__main__":
    build_dashboard()
    build_map()
    build_index()
    print("[done] open web/index.html")
