"""Candidate generation: for each GPS point, the K nearest arcs within a radius.

Arcs are densified into sample points every `step` metres and indexed in a
KD-tree; a ball query returns arcs that *might* be close, then the exact
point-to-segment distance is used to rank and filter them.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import point_segment


class ArcIndex:
    def __init__(self, trip, step=10.0):
        self.trip = trip
        self.step = step
        a, b, L = trip.arc_a, trip.arc_b, trip.arc_len
        n = np.maximum(1, np.ceil(L / step).astype(int)) + 1
        arc_id = np.repeat(np.arange(len(L)), n)
        # fraction 0..1 along each arc for each sample
        offs = np.concatenate([[0], np.cumsum(n)[:-1]])
        k = np.arange(n.sum()) - np.repeat(offs, n)
        frac = k / np.repeat(n - 1, n)
        pts = a[arc_id] + frac[:, None] * (b[arc_id] - a[arc_id])
        self.sample_arc = arc_id
        self.tree = cKDTree(pts)

    def query(self, xy, radius=50.0, k=8):
        """Return (cand_arcs, cand_dist, cand_t) as lists of arrays, one per point.

        Candidates are sorted by distance. A point with no arc inside the
        radius gets an empty array (handled downstream as 'missing candidates').
        """
        trip = self.trip
        hits = self.tree.query_ball_point(xy, r=radius + self.step / 2)
        arcs_out, dist_out, t_out = [], [], []
        A, B = trip.arc_a, trip.arc_b
        for i, h in enumerate(hits):
            if not h:
                arcs_out.append(np.empty(0, int)); dist_out.append(np.empty(0)); t_out.append(np.empty(0))
                continue
            c = np.unique(self.sample_arc[h])
            d, t = point_segment(xy[i][None, :], A[c], B[c])
            keep = d <= radius
            c, d, t = c[keep], d[keep], t[keep]
            o = np.argsort(d, kind="stable")[:k]
            arcs_out.append(c[o]); dist_out.append(d[o]); t_out.append(t[o])
        return arcs_out, dist_out, t_out
