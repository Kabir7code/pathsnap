"""Shared experiment protocol (used by model comparison and Bayesian optimisation)."""
import os
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from .config import CFG
from .data import load_trip
from .dataset import subsample_points
from .features import FEATURES
from .metrics import candidate_metrics, continuity, top1_accuracy, top1_predictions


def evaluate(model, df, pts):
    t0 = time.time()
    p = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
    infer = time.time() - t0
    m = candidate_metrics(df.label.to_numpy(), p)
    d = df[["trip", "point", "arc"]].copy(); d["score"] = p
    pred = top1_predictions(d)
    m["top1"] = top1_accuracy(pts, pred)
    m["infer_s"] = infer
    m["infer_ms_per_point"] = 1000 * infer / df[["trip", "point"]].drop_duplicates().shape[0]
    return m, pred


def cross_validate(make, tr, tr_pts, folds=5, max_rows=300_000, seed=42, verbose=True):
    out = []
    for f, (i_tr, i_va) in enumerate(GroupKFold(n_splits=folds).split(tr, groups=tr.trip)):
        fit = subsample_points(tr.iloc[i_tr], max_rows, seed + f)
        va = tr.iloc[i_va]
        model = make()
        t0 = time.time(); model.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy()); fit_s = time.time() - t0
        m, _ = evaluate(model, va, tr_pts[tr_pts.trip.isin(va.trip.unique())])
        m["fit_s"] = fit_s
        out.append(m)
        if verbose:
            print(f"  fold {f}: F1={m['macro_f1']:.4f} top1={m['top1']:.4f} ll={m['log_loss']:.4f}", flush=True)
    return out


_ARCS = {}


def fit_and_test(make, tr, te, te_pts, max_rows=300_000, seed=42):
    fit = subsample_points(tr, max_rows, seed)
    model = make()
    t0 = time.time(); model.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy()); fit_s = time.time() - t0
    m, pred = evaluate(model, te, te_pts)
    conts = []
    for tid, g in pred.groupby("trip"):
        if tid not in _ARCS:
            _ARCS[tid] = load_trip(os.path.join(CFG["data_dir"], "student_test"), tid, with_route=False).arcs
        conts.append(continuity(g.sort_values("point").pred_arc.to_numpy(), _ARCS[tid]))
    m["continuity"] = float(np.mean(conts))
    m["fit_s"] = fit_s
    m["train_rows"] = len(fit)
    return model, m
