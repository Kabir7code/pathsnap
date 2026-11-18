"""Feature engineering for (GPS point, candidate arc) pairs.

Problem framing: for every GPS point we generate up to K candidate arcs.
Each (point, candidate) pair is one row; the binary target is
"this candidate is the reference arc for this point".  A classifier scores
every candidate; the highest score is the classifier-only prediction, and
the scores are used as HMM emission probabilities.

Features (all in metres / degrees / seconds):
  dist            perpendicular distance point -> arc
  dist_rank       rank of the candidate by distance (0 = nearest)
  dist_minus_min  dist - nearest candidate's dist
  win_dist        mean distance of points p-W..p+W to the arc (local context)
  heading_diff    |GPS heading - arc bearing|, 0..180 (separates the two
                  directions of a two-way street, which have equal dist)
  heading_valid   1 if the vehicle moved enough to define a heading
  speed           GPS speed estimate (m/s)
  dt_prev         time since previous observation (gap indicator)
  arc_len         arc length
  proj_t          projection position along the arc (0 start, 1 end)
  n_cands         number of candidates for this point
  out_deg_end     out-degree of the arc's end node
  in_deg_start    in-degree of the arc's start node
  has_twin        1 if the reverse arc also exists (two-way road)
"""
import numpy as np
import pandas as pd

from .candidates import ArcIndex
from .geometry import angle_diff, point_segment

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "win_dist", "heading_diff",
    "heading_valid", "speed", "dt_prev", "arc_len", "proj_t", "n_cands",
    "out_deg_end", "in_deg_start", "has_twin",
]


def gps_heading(xy, lag=2, min_move=3.0):
    """Heading at each point from p-lag to p+lag (degrees), and a validity mask."""
    n = len(xy)
    i0 = np.clip(np.arange(n) - lag, 0, n - 1)
    i1 = np.clip(np.arange(n) + lag, 0, n - 1)
    d = xy[i1] - xy[i0]
    move = np.linalg.norm(d, axis=1)
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])), move >= min_move


def trip_graph_stats(trip):
    A = trip.arcs
    N = len(trip.nodes_xy)
    out_deg = np.bincount(A[:, 0], minlength=N)
    in_deg = np.bincount(A[:, 1], minlength=N)
    key = A[:, 0] * N + A[:, 1]
    rev = A[:, 1] * N + A[:, 0]
    has_twin = np.isin(rev, key)
    return out_deg, in_deg, has_twin


def build_trip_frame(trip, radius=50.0, k=8, window=2, true_arc=None, index=None):
    """Return (DataFrame of candidate rows, list of per-point candidate arrays)."""
    xy = trip.track_xy
    n = len(xy)
    index = index or ArcIndex(trip)
    cands, dists, ts = index.query(xy, radius=radius, k=k)
    heading, hvalid = gps_heading(xy)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(trip.times)
    dt_prev = np.concatenate([[1.0], dt])
    speed = np.concatenate([[0.0], step / np.maximum(dt, 1e-6)])
    out_deg, in_deg, has_twin = trip_graph_stats(trip)

    counts = np.array([len(c) for c in cands])
    pidx = np.repeat(np.arange(n), counts)
    arc = np.concatenate(cands) if counts.sum() else np.empty(0, int)
    dist = np.concatenate(dists) if counts.sum() else np.empty(0)
    proj_t = np.concatenate(ts) if counts.sum() else np.empty(0)
    rank = np.concatenate([np.arange(c) for c in counts]) if counts.sum() else np.empty(0, int)
    mind = np.array([d[0] if len(d) else np.nan for d in dists])

    # windowed distance: mean distance of neighbouring points to the same arc
    A, B = trip.arc_a[arc], trip.arc_b[arc]
    acc = np.zeros(len(arc))
    for off in range(-window, window + 1):
        q = np.clip(pidx + off, 0, n - 1)
        d, _ = point_segment(xy[q], A, B)
        acc += d
    win_dist = acc / (2 * window + 1)

    df = pd.DataFrame({
        "trip": trip.trip_id,
        "point": pidx,
        "arc": arc,
        "dist": dist,
        "dist_rank": rank,
        "dist_minus_min": dist - mind[pidx],
        "win_dist": win_dist,
        "heading_diff": np.where(hvalid[pidx], angle_diff(heading[pidx], trip.arc_bearing[arc]), 90.0),
        "heading_valid": hvalid[pidx].astype(int),
        "speed": speed[pidx],
        "dt_prev": dt_prev[pidx],
        "arc_len": trip.arc_len[arc],
        "proj_t": proj_t,
        "n_cands": counts[pidx],
        "out_deg_end": out_deg[trip.arcs[arc, 1]],
        "in_deg_start": in_deg[trip.arcs[arc, 0]],
        "has_twin": has_twin[arc].astype(int),
    })
    if true_arc is not None:
        df["true_arc"] = true_arc[pidx]
        df["label"] = (df["arc"].to_numpy() == df["true_arc"].to_numpy()).astype(int)
    return df, cands, dists, ts
