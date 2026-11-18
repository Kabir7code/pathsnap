"""Loading raw trip files and projecting coordinates to metres.

Each trip folder <id>/ contains:
  <id>.nodes  lon lat               (row index = node_id)
  <id>.arcs   from_node to_node     (row index = arc_id, directed)
  <id>.track  lon lat t_seconds     (GPS observations)
  <id>.route  arc_id per line       (reference route, chronological)

Projection: local equirectangular projection centred on the trip's node
centroid.  At city scale (tens of km) its distortion is far below GPS noise,
and it lets us use plain Euclidean geometry in metres.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EARTH_R = 6_371_000.0


def _read(path, ncols, dtype=float):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    assert df.shape[1] == ncols, f"{path}: expected {ncols} cols, got {df.shape[1]}"
    return df.to_numpy(dtype=dtype)


def list_trips(split_dir):
    return sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))


@dataclass
class Trip:
    trip_id: str
    nodes_ll: np.ndarray          # (N,2) lon, lat
    arcs: np.ndarray              # (A,2) from, to
    track_ll: np.ndarray          # (P,2) lon, lat
    times: np.ndarray             # (P,)
    route: np.ndarray | None      # (R,) arc ids or None if hidden
    lon0: float = 0.0
    lat0: float = 0.0
    nodes_xy: np.ndarray = field(default=None, repr=False)
    track_xy: np.ndarray = field(default=None, repr=False)

    # ---- projection -------------------------------------------------
    def to_xy(self, ll):
        ll = np.asarray(ll, dtype=float)
        k = np.cos(np.radians(self.lat0))
        x = np.radians(ll[..., 0] - self.lon0) * EARTH_R * k
        y = np.radians(ll[..., 1] - self.lat0) * EARTH_R
        return np.stack([x, y], axis=-1)

    def to_ll(self, xy):
        xy = np.asarray(xy, dtype=float)
        k = np.cos(np.radians(self.lat0))
        lon = np.degrees(xy[..., 0] / (EARTH_R * k)) + self.lon0
        lat = np.degrees(xy[..., 1] / EARTH_R) + self.lat0
        return np.stack([lon, lat], axis=-1)

    # ---- derived arc geometry -----------------------------------------
    @property
    def arc_a(self):
        return self.nodes_xy[self.arcs[:, 0]]

    @property
    def arc_b(self):
        return self.nodes_xy[self.arcs[:, 1]]

    @property
    def arc_len(self):
        if not hasattr(self, "_arc_len"):
            self._arc_len = np.linalg.norm(self.arc_b - self.arc_a, axis=1)
        return self._arc_len

    @property
    def arc_bearing(self):
        """Bearing of each arc in degrees, 0 = east, counter-clockwise."""
        if not hasattr(self, "_arc_bearing"):
            d = self.arc_b - self.arc_a
            self._arc_bearing = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        return self._arc_bearing


def load_trip(split_dir, trip_id, with_route=True) -> Trip:
    d = os.path.join(split_dir, trip_id)
    nodes = _read(os.path.join(d, f"{trip_id}.nodes"), 2)
    arcs = _read(os.path.join(d, f"{trip_id}.arcs"), 2, dtype=np.int64)
    track = _read(os.path.join(d, f"{trip_id}.track"), 3)
    route = None
    rp = os.path.join(d, f"{trip_id}.route")
    if with_route and os.path.exists(rp):
        route = _read(rp, 1, dtype=np.int64).ravel()

    # Basic cleaning: drop non-finite rows, sort by time, drop duplicate timestamps.
    ok = np.isfinite(track).all(axis=1)
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    _, first = np.unique(track[:, 2], return_index=True)
    track = track[np.sort(first)]

    t = Trip(trip_id, nodes, arcs, track[:, :2], track[:, 2], route)
    t.lon0, t.lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    t.nodes_xy = t.to_xy(nodes)
    t.track_xy = t.to_xy(track[:, :2])
    return t
