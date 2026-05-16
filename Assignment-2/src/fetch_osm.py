"""Fetch Dhaka fuel stations from OpenStreetMap via Overpass API.

Uses a direct Overpass HTTP query inside the Dhaka administrative bounding box
because the osmnx `features_from_place` pipeline can stall on large geocoded
boundaries. Saves:
  * data/dhaka_fuel_stations.geojson
  * data/dhaka_fuel_stations.csv
  * data/osm_summary.json
  * figures/dhaka_fuel_stations_map.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# Dhaka administrative bounding box (south, west, north, east) — covers
# Dhaka City Corporation North + South and inner suburbs.
DHAKA_BBOX = (23.65, 90.32, 23.92, 90.52)

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

QUERY = f"""
[out:json][timeout:60];
(
  node["amenity"="fuel"]({DHAKA_BBOX[0]},{DHAKA_BBOX[1]},{DHAKA_BBOX[2]},{DHAKA_BBOX[3]});
  way ["amenity"="fuel"]({DHAKA_BBOX[0]},{DHAKA_BBOX[1]},{DHAKA_BBOX[2]},{DHAKA_BBOX[3]});
  relation["amenity"="fuel"]({DHAKA_BBOX[0]},{DHAKA_BBOX[1]},{DHAKA_BBOX[2]},{DHAKA_BBOX[3]});
);
out center tags;
"""


def fetch() -> pd.DataFrame:
    headers = {"User-Agent": "dhaka-fuel-csp/0.1 (research)"}
    js = None
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        print(f"[overpass] POST {url}  bbox={DHAKA_BBOX}")
        try:
            r = requests.post(url, data={"data": QUERY}, headers=headers, timeout=120)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:
            print(f"[overpass]   FAILED: {e}")
            last_err = e
            continue
    if js is None:
        raise RuntimeError(f"All Overpass endpoints failed; last: {last_err}")
    rows = []
    for el in js.get("elements", []):
        tags = el.get("tags", {}) or {}
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            c = el.get("center", {})
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "osmid": el["id"],
            "type": el["type"],
            "name": tags.get("name"),
            "operator": tags.get("operator"),
            "brand": tags.get("brand"),
            "lat": float(lat),
            "lon": float(lon),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    df["station_id"] = [f"S{i:03d}" for i in range(len(df))]
    return df


def save(df: pd.DataFrame) -> None:
    geoms = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(DATA / "dhaka_fuel_stations.geojson", driver="GeoJSON")
    df.to_csv(DATA / "dhaka_fuel_stations.csv", index=False)
    print(f"[save] {DATA/'dhaka_fuel_stations.csv'} ({len(df)} stations)")

    # Build a proper basemap-backed scatter using contextily.
    import contextily as cx
    s, w, n, e = DHAKA_BBOX
    gdf_3857 = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(11, 12))
    gdf_3857.plot(
        ax=ax, color="#ef4444", markersize=46, alpha=0.92,
        edgecolor="white", linewidth=0.9,
        label=f"Fuel stations (n={len(gdf)})",
    )
    # bbox rectangle in web-mercator
    from shapely.geometry import box as _box
    import geopandas as _gpd
    bbox_3857 = _gpd.GeoDataFrame(
        geometry=[_box(w, s, e, n)], crs="EPSG:4326").to_crs(epsg=3857)
    bbox_3857.boundary.plot(ax=ax, color="#0f172a", linewidth=1.8,
                            label="Dhaka bounding box")

    # Pull a real OSM basemap. Falls back to no-tile if offline.
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=11)
    except Exception as exc:
        print(f"[basemap] CartoDB failed ({exc}); trying Esri Gray")
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldGrayCanvas, zoom=11)
        except Exception as exc2:
            print(f"[basemap] Esri failed too ({exc2}); using plain grid")
            ax.grid(alpha=0.3)

    ax.set_title("Dhaka City — Fuel Stations (OSM amenity=fuel)", fontsize=14, pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    leg = ax.legend(loc="upper right", frameon=True, facecolor="white",
                    edgecolor="#cbd5e1")
    for txt in leg.get_texts():
        txt.set_fontsize(11)
    fig.tight_layout()
    out_png = FIGS / "dhaka_fuel_stations_map.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"[save] {out_png}")

    summary = {
        "place": "Dhaka, Bangladesh",
        "bbox_S_W_N_E": list(DHAKA_BBOX),
        "n_stations": int(len(df)),
        "data_lon_range": [float(df.lon.min()), float(df.lon.max())],
        "data_lat_range": [float(df.lat.min()), float(df.lat.max())],
    }
    (DATA / "osm_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    df = fetch()
    if len(df) == 0:
        print("[error] Overpass returned 0 stations — retry later", file=sys.stderr)
        sys.exit(1)
    save(df)
    print(f"[done] {len(df)} fuel stations saved.")
