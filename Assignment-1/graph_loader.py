"""Load the whole Dhaka city road graph and per-factor POIs from OSM."""
import osmnx as ox

ox.settings.use_cache = True

DHAKA_PLACE = "Dhaka, Bangladesh"

FACTOR_TAGS = {
    "police":         ("amenity", "police"),
    "hospital":       ("amenity", "hospital"),
    "fire_station":   ("amenity", "fire_station"),
    "bus_station":    ("amenity", "bus_station"),
    "marketplace":    ("amenity", "marketplace"),
    "level_crossing": ("railway", "level_crossing"),
}

SAFETY_NAMES = {"police", "hospital", "fire_station"}


def load_graph(place=DHAKA_PLACE, network_type="drive", simplify=True):
    """Full Dhaka drivable road graph."""
    G = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    G = ox.distance.add_edge_lengths(G)
    return G


def poi_points(place, tag_key, tag_value):
    """Return a list of (lat, lon) POIs matching tag_key=tag_value within place."""
    try:
        gdf = ox.features_from_place(place, tags={tag_key: tag_value})
    except Exception as e:
        print(f"  [warn] POI fetch failed for {tag_key}={tag_value}: {e}")
        return []
    if gdf is None or len(gdf) == 0:
        return []
    pts = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if hasattr(geom, "x") and hasattr(geom, "y"):
            pts.append((geom.y, geom.x))
        else:
            c = geom.centroid
            pts.append((c.y, c.x))
    return pts
