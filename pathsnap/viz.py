"""Folium map building (shared by the map script)."""
import folium
import numpy as np
from folium.plugins import Fullscreen

from .geometry import point_segment


def _arc_lines(trip, arcs):
    a = trip.nodes_ll[trip.arcs[arcs, 0]]
    b = trip.nodes_ll[trip.arcs[arcs, 1]]
    return [[[p[1], p[0]], [q[1], q[0]]] for p, q in zip(a, b)]


def polylines_from_route(trip, route):
    """Merge a route (arc list) into continuous [lat,lon] polylines."""
    lines, cur = [], []
    for a in route:
        u, v = trip.arcs[a]
        pu, pv = trip.nodes_ll[u], trip.nodes_ll[v]
        if cur and cur[-1] == [pu[1], pu[0]]:
            cur.append([pv[1], pv[0]])
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[pu[1], pu[0]], [pv[1], pv[0]]]
    if len(cur) > 1:
        lines.append(cur)
    return lines


def nearby_arcs(trip, max_dist=150.0, step=5):
    """Arcs within max_dist of the (subsampled) track, for drawing the relevant network."""
    from scipy.spatial import cKDTree
    tree = cKDTree(trip.track_xy[::step])
    mid = (trip.arc_a + trip.arc_b) / 2
    d, _ = tree.query(mid, distance_upper_bound=max_dist + trip.arc_len.max() / 2)
    return np.flatnonzero(np.isfinite(d) & (d <= max_dist + trip.arc_len / 2))


def trip_map(trip, layers, title=""):
    """layers: list of (name, route_arcs, colour, weight, dash) drawn as polylines."""
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    Fullscreen().add_to(m)
    net = folium.FeatureGroup(name="Road network (near track)", show=True)
    for line in _arc_lines(trip, nearby_arcs(trip)):
        folium.PolyLine(line, color="#888", weight=1, opacity=.5).add_to(net)
    net.add_to(m)
    for name, route, color, weight, dash in layers:
        fg = folium.FeatureGroup(name=name)
        for line in polylines_from_route(trip, route):
            folium.PolyLine(line, color=color, weight=weight, opacity=.85, dash_array=dash, tooltip=name).add_to(fg)
        fg.add_to(m)
    gps = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(trip.track_ll) // 3000)
    for i in range(0, len(trip.track_ll), step):
        lon, lat = trip.track_ll[i]
        folium.CircleMarker([lat, lon], radius=2, color="#1f5fd1", fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"#{i} t={trip.times[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    s, e = trip.track_ll[0], trip.track_ll[-1]
    folium.Marker([s[1], s[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([e[1], e[0]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    lo, hi = trip.track_ll.min(0), trip.track_ll.max(0)
    m.fit_bounds([[lo[1], lo[0]], [hi[1], hi[0]]])
    items = "".join(f'<div><span style="display:inline-block;width:22px;height:4px;background:{c};'
                    f'vertical-align:middle;margin-right:6px"></span>{n}</div>' for n, _, c, _, _ in layers)
    legend = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;'
              f'border:1px solid #999;border-radius:4px;font:12px sans-serif"><b>{title}</b>'
              f'<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#1f5fd1;'
              f'margin:0 14px 0 7px"></span>GPS observations</div>'
              f'<div><span style="display:inline-block;width:22px;height:2px;background:#888;vertical-align:middle;'
              f'margin-right:6px"></span>Road network</div>{items}</div>')
    m.get_root().html.add_child(folium.Element(legend))
    return m
