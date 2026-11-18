"""HMM map matching with Viterbi decoding.

Hidden states   : at GPS point p, state = one of its candidate arcs c_p,k
                  (together with the projection position t on that arc).
Emission        : log P(obs p | state) = w * log(classifier probability that
                  the candidate is the true arc).
Transition      : Newson & Krumm (2009) style.  For consecutive points p, q
                  with straight-line distance d_gc, and network distance
                  d_route between their projections on candidate arcs i, j:
                      log P(j | i) = -|d_route - d_gc| / beta
                  d_route is computed with a cached, cut-off Dijkstra over
                  the directed arc graph (arc lengths as weights), so
                  one-way streets and connectivity are respected.
Unreachable     : if j cannot be reached from i within the cut-off, the
                  transition gets a fixed floor log-prob (unreachable_logp)
                  instead of -inf, so the decoder never "breaks" -- it can
                  still jump if the evidence is overwhelming.
Missing cands   : points without any candidate are skipped by the decoder
                  (prediction -1); the next point is linked to the last point
                  that had candidates.
GPS gaps        : if the time gap exceeds gap_seconds, or the straight-line
                  distance exceeds half the Dijkstra cut-off, transitions are
                  set uniform (log 0): the chain effectively restarts there,
                  relying on emissions only.
Route output    : the decoded arc sequence is de-duplicated and consecutive
                  arcs that are not adjacent are joined by the shortest path.
"""
import heapq

import numpy as np


class RoadGraph:
    def __init__(self, trip, cutoff=1500.0):
        self.trip = trip
        self.cutoff = cutoff
        A = trip.arcs
        self.n_nodes = len(trip.nodes_xy)
        order = np.argsort(A[:, 0], kind="stable")
        self.out_arcs = order
        self.out_ptr = np.searchsorted(A[order, 0], np.arange(self.n_nodes + 1))
        self.arc_to = A[:, 1]
        self.arc_from = A[:, 0]
        self.length = trip.arc_len
        self._cache = {}

    def sssp(self, src):
        """Dijkstra from node src up to self.cutoff metres. Returns (dist, pred_arc) dicts."""
        r = self._cache.get(src)
        if r is not None:
            return r
        dist = {src: 0.0}
        pred = {}
        heap = [(0.0, src)]
        ptr, oa, to, L, cut = self.out_ptr, self.out_arcs, self.arc_to, self.length, self.cutoff
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for e in oa[ptr[u]:ptr[u + 1]]:
                v = to[e]
                nd = d + L[e]
                if nd <= cut and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = e
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (dist, pred)
        return dist, pred

    def route_distance(self, ai, ti, aj, tj, backward_tol=15.0):
        Li, Lj = self.length[ai], self.length[aj]
        if ai == aj:
            delta = (tj - ti) * Li
            return max(delta, 0.0) if delta >= -backward_tol else None
        dist, _ = self.sssp(self.arc_to[ai])
        d = dist.get(self.arc_from[aj])
        if d is None:
            return None
        return (1 - ti) * Li + d + tj * Lj

    def shortest_arc_path(self, ai, aj):
        """Arcs strictly between ai and aj on the shortest path (None if unreachable)."""
        u, v = self.arc_to[ai], self.arc_from[aj]
        if u == v:
            return []
        dist, pred = self.sssp(u)
        if v not in dist:
            return None
        path = []
        while v != u:
            e = pred[v]
            path.append(int(e))
            v = self.arc_from[e]
        return path[::-1]


def viterbi(trip, cands, ts, probs, graph, beta=10.0, emission_weight=1.0,
            unreachable_logp=-30.0, backward_tol=15.0, gap_seconds=30.0):
    """Decode the most likely candidate sequence.

    cands, ts, probs: per-point arrays of candidate arc ids, projection t, and
    classifier probabilities.  Returns an array of predicted arc ids (-1 for
    points without candidates).
    """
    n = len(cands)
    xy, times = trip.track_xy, trip.times
    valid = [i for i in range(n) if len(cands[i])]
    pred = np.full(n, -1, dtype=np.int64)
    if not valid:
        return pred
    emis = [emission_weight * np.log(np.clip(probs[i], 1e-6, 1.0)) for i in range(n)]
    delta = emis[valid[0]].copy()
    backs = {}
    stats = {"uniform_steps": 0, "unreachable_pairs": 0, "pairs": 0}
    for prev, cur in zip(valid, valid[1:]):
        d_gc = float(np.linalg.norm(xy[cur] - xy[prev]))
        gap = times[cur] - times[prev]
        cp, cc = cands[prev], cands[cur]
        if gap > gap_seconds or d_gc > graph.cutoff / 2:
            trans = np.zeros((len(cp), len(cc)))
            stats["uniform_steps"] += 1
        else:
            trans = np.empty((len(cp), len(cc)))
            for a in range(len(cp)):
                for b in range(len(cc)):
                    d = graph.route_distance(cp[a], ts[prev][a], cc[b], ts[cur][b], backward_tol)
                    if d is None:
                        trans[a, b] = unreachable_logp
                        stats["unreachable_pairs"] += 1
                    else:
                        trans[a, b] = -abs(d - d_gc) / beta
            stats["pairs"] += trans.size
        score = delta[:, None] + trans
        bp = np.argmax(score, axis=0)
        delta = score[bp, np.arange(len(cc))] + emis[cur]
        backs[cur] = (prev, bp)
    # backtrack
    k = int(np.argmax(delta))
    cur = valid[-1]
    while True:
        pred[cur] = cands[cur][k]
        if cur not in backs:
            break
        prev, bp = backs[cur]
        k = int(bp[k]); cur = prev
    viterbi.last_stats = stats
    return pred


def reconstruct_route(point_arcs, graph):
    """Collapse per-point arcs into a route; fill non-adjacent jumps with shortest paths."""
    seq = []
    for a in point_arcs:
        if a >= 0 and (not seq or seq[-1] != a):
            seq.append(int(a))
    route = seq[:1]
    gaps = 0
    for a, b in zip(seq, seq[1:]):
        mid = graph.shortest_arc_path(a, b)
        if mid is None:
            gaps += 1
            mid = []
        route.extend(mid)
        route.append(b)
    return route, gaps
