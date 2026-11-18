"""End-to-end inference for one trip: candidates -> classifier -> HMM -> route.

Used by the HMM evaluation script, the Folium maps and the API, so every
consumer runs exactly the same code path.  The reference route is never an
input here; scoring is done separately (pathsnap.metrics).
"""
import time

import numpy as np

from .config import CFG
from .features import FEATURES, build_trip_frame
from .hmm import RoadGraph, reconstruct_route, viterbi


def split_by_point(values, cands):
    return np.split(values, np.cumsum([len(c) for c in cands])[:-1])


def predict_trip(trip, model, hmm_params=None, use_hmm=True):
    hp = dict(CFG["hmm"]); hp.update(hmm_params or {})
    timing = {}
    t0 = time.time()
    df, cands, _, ts = build_trip_frame(trip, CFG["radius"], CFG["k"], CFG["window"])
    timing["features_s"] = time.time() - t0
    t0 = time.time()
    p = model.predict_proba(df[FEATURES].to_numpy())[:, 1] if len(df) else np.empty(0)
    timing["classifier_s"] = time.time() - t0
    probs = split_by_point(p, cands)
    cls_pred = np.array([c[np.argmax(q)] if len(c) else -1 for c, q in zip(cands, probs)], dtype=np.int64)
    out = {"cands": cands, "probs": probs, "cls_pred": cls_pred, "timing": timing}
    graph = RoadGraph(trip, cutoff=hp["dijkstra_cutoff"])
    if use_hmm:
        t0 = time.time()
        hmm_pred = viterbi(trip, cands, ts, probs, graph, beta=hp["beta"],
                           emission_weight=hp["emission_weight"], unreachable_logp=hp["unreachable_logp"],
                           backward_tol=hp["backward_tolerance"], gap_seconds=hp.get("gap_seconds", 30.0))
        timing["hmm_s"] = time.time() - t0
        out["hmm_pred"] = hmm_pred
        out["hmm_stats"] = dict(viterbi.last_stats)
        out["route"], out["route_gaps"] = reconstruct_route(hmm_pred, graph)
    out["cls_route"], out["cls_route_gaps"] = reconstruct_route(cls_pred, graph)
    timing["total_s"] = sum(timing.values())
    return out


def score_trip(trip, out, true_arc):
    """Metrics for one trip given per-point reference arcs (from labels.align_track_to_route)."""
    from .metrics import continuity, route_overlap
    lab = true_arc >= 0
    res = {"points": int(len(true_arc)), "labelled": int(lab.sum())}
    for key, pred, route in [("cls", out["cls_pred"], out["cls_route"]),
                             ("hmm", out.get("hmm_pred"), out.get("route"))]:
        if pred is None:
            continue
        res[f"{key}_top1"] = float((pred[lab] == true_arc[lab]).mean())
        res[f"{key}_continuity"] = continuity(pred, trip.arcs)
        res.update({f"{key}_{k}": v for k, v in route_overlap(route, trip.route).items()})
    return res
