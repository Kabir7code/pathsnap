"""Deriving per-GPS-point labels from the reference route.

The .route file lists the arcs travelled, in order, but does NOT say which
GPS point belongs to which arc.  We align the two sequences with a
monotonic dynamic program (similar to DTW):

    cost[p, r] = dist(point p, route arc r) + min_{r' <= r} cost[p-1, r']

i.e. each point is assigned to a route arc, the assigned route index may
never go backwards, and the total point-to-arc distance is minimised.
Points whose assigned arc is further than `max_dist` metres are marked as
unlabelled (-1): they are GPS outliers and their 'true' arc is ambiguous.
"""
import numpy as np

from .geometry import point_segment


def align_track_to_route(trip, max_dist=60.0):
    R = trip.route
    P = trip.track_xy
    a, b = trip.arc_a[R], trip.arc_b[R]
    nP, nR = len(P), len(R)
    back = np.empty((nP, nR), dtype=np.int32 if nR > 32000 else np.int16)
    prev = None
    for p in range(nP):
        d, _ = point_segment(P[p][None, :], a, b)
        if prev is None:
            # first point may start anywhere, but prefer early route arcs
            cur = d.copy()
            back[p] = np.arange(nR)
        else:
            # running minimum of prev with its argmin
            cm = np.minimum.accumulate(prev)
            idx = np.arange(nR)
            is_new = np.concatenate([[True], prev[1:] < cm[:-1]])
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cur = d + cm
            back[p] = arg
        prev = cur
    # backtrack (the last point may end on any route arc)
    r = int(np.argmin(prev))
    assign = np.empty(nP, dtype=np.int64)
    for p in range(nP - 1, -1, -1):
        assign[p] = r
        r = int(back[p, r])
    true_arc = R[assign]
    d, _ = point_segment(P, trip.arc_a[true_arc], trip.arc_b[true_arc])
    true_arc = np.where(d <= max_dist, true_arc, -1)
    return true_arc, assign, d
