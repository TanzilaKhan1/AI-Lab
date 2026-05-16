"""Map + route + exploration heatmap.

Safety POIs (police / hospital / fire) are plotted as blue '+' markers,
hazard POIs (bus terminals / markets / level crossings) as red 'X' markers.
The found path is overlaid thick red; expanded nodes are colored by expansion
order via the viridis colormap (yellow = early, purple = late).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import osmnx as ox


def plot_search(G, result, *, pois_safety=None, pois_hazard=None,
                save_path=None, title=""):
    expansion_order = result.extra.get("expansion_order", {})
    path_set = set(result.path)
    max_step = max(expansion_order.values()) if expansion_order else 1

    node_colors, node_sizes = [], []
    for n in G.nodes():
        if n in path_set:
            node_colors.append("#e53935")          # red -- path
            node_sizes.append(6)
        elif n in expansion_order:
            t = expansion_order[n] / max_step
            node_colors.append(cm.viridis(t))      # heatmap by expansion order
            node_sizes.append(2)
        else:
            node_colors.append((0.88, 0.88, 0.88, 0.35))
            node_sizes.append(0.1)

    fig, ax = ox.plot_graph(
        G, node_color=node_colors, node_size=node_sizes,
        edge_color="#dedede", edge_linewidth=0.25,
        bgcolor="white", show=False, close=False, figsize=(14, 14),
    )

    if len(result.path) > 1:
        xs = [G.nodes[n]["x"] for n in result.path]
        ys = [G.nodes[n]["y"] for n in result.path]
        ax.plot(xs, ys, color="#e53935", linewidth=2.8, alpha=0.95, zorder=5)
        ax.scatter(xs[0], ys[0], s=220, c="#43a047", edgecolors="black",
                   linewidth=1.5, marker="o", zorder=6, label="Start")
        ax.scatter(xs[-1], ys[-1], s=260, c="#039be5", edgecolors="black",
                   linewidth=1.5, marker="*", zorder=6, label="Goal")

    if pois_safety:
        lats, lons = zip(*pois_safety)
        ax.scatter(lons, lats, s=35, c="#1e88e5", marker="P", alpha=0.75,
                   edgecolors="white", linewidth=0.4, zorder=4,
                   label="Safety POI (police/hospital/fire)")
    if pois_hazard:
        lats, lons = zip(*pois_hazard)
        ax.scatter(lons, lats, s=32, c="#ef5350", marker="X", alpha=0.75,
                   edgecolors="white", linewidth=0.4, zorder=4,
                   label="Hazard POI (bus/market/crossing)")

    sm = plt.cm.ScalarMappable(cmap=cm.viridis,
                               norm=plt.Normalize(vmin=0, vmax=max_step))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Expansion order (early -> late)", fontsize=9)

    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
