"""Train and compare candidate classifiers.

Protocol
  * 5-fold GroupKFold over the training trips (a trip never appears in both
    the fitting and validation part of a fold -> no leakage between points
    of the same trip).
  * Each fold fits on a point-level subsample (max_rows) for tractability,
    and is scored on ALL candidates of the held-out trips.
  * The model is then refitted on the training split and scored on the ten
    student-testing trips.
Outputs: results/model_comparison.csv, results/model_runs.json,
         artifacts/models/<name>.joblib
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import CFG
from pathsnap.dataset import load_split, subsample_points
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, continuity, top1_accuracy, top1_predictions
from pathsnap.models import MODEL_INFO, make_model
from pathsnap.data import load_trip

DEFAULT = ["logreg", "gnb", "knn", "dtree", "rf", "extratrees", "histgb", "lightgbm", "mlp"]
# KNN prediction cost grows with training size, so it gets a smaller sample
MAX_ROWS = {"knn": 100_000}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT)
    ap.add_argument("--max_rows", type=int, default=300_000)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    seed = CFG["seed"]
    tr, tr_pts = load_split("train")
    te, te_pts = load_split("student_test", exclude_bad=False)
    os.makedirs("artifacts/models", exist_ok=True)
    arcs_cache = {}
    runs = json.load(open("results/model_runs.json")) if os.path.exists("results/model_runs.json") else {}

    for name in args.models:
        print(f"=== {name}", flush=True)
        mr = MAX_ROWS.get(name, args.max_rows)
        folds = []
        gkf = GroupKFold(n_splits=args.folds)
        for f, (i_tr, i_va) in enumerate(gkf.split(tr, groups=tr.trip)):
            fit = subsample_points(tr.iloc[i_tr], mr, seed + f)
            va = tr.iloc[i_va]
            model = make_model(name, seed)
            t0 = time.time(); model.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy()); fit_s = time.time() - t0
            m, _ = evaluate(model, va, tr_pts[tr_pts.trip.isin(va.trip.unique())])
            m["fit_s"] = fit_s
            folds.append(m)
            print(f"  fold {f}: F1={m['macro_f1']:.4f} top1={m['top1']:.4f} ll={m['log_loss']:.4f}", flush=True)

        fit = subsample_points(tr, mr, seed)
        model = make_model(name, seed)
        t0 = time.time(); model.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy()); fit_s = time.time() - t0
        test_m, pred = evaluate(model, te, te_pts)
        # classifier-only route continuity on each student-test trip
        conts = []
        for tid, g in pred.groupby("trip"):
            if tid not in arcs_cache:
                arcs_cache[tid] = load_trip(os.path.join(CFG["data_dir"], "student_test"), tid, with_route=False).arcs
            conts.append(continuity(g.sort_values("point").pred_arc.to_numpy(), arcs_cache[tid]))
        test_m["continuity"] = float(np.mean(conts))
        test_m["fit_s"] = fit_s
        joblib.dump(model, f"artifacts/models/{name}.joblib", compress=3)
        runs[name] = {"family": MODEL_INFO[name][0], "config": MODEL_INFO[name][1], "seed": seed,
                      "train_rows": len(fit), "cv_folds": folds, "student_test": test_m}
        print(f"  TEST: F1={test_m['macro_f1']:.4f} top1={test_m['top1']:.4f} ll={test_m['log_loss']:.4f} "
              f"cont={test_m['continuity']:.4f} infer={test_m['infer_s']:.2f}s", flush=True)
        json.dump(runs, open("results/model_runs.json", "w"), indent=1)

    rows = []
    for name, r in runs.items():
        cv = pd.DataFrame([{k: v for k, v in f.items() if k != "confusion"} for f in r["cv_folds"]])
        t = r["student_test"]
        rows.append({"model": name, "family": r["family"], "config": r["config"],
                     "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
                     "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
                     "test_top1": t["top1"], "test_macro_f1": t["macro_f1"], "test_precision": t["precision"],
                     "test_recall": t["recall"], "test_log_loss": t["log_loss"], "test_continuity": t["continuity"],
                     "fit_s": t["fit_s"], "infer_ms_per_point": t["infer_ms_per_point"]})
    out = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    out.to_csv("results/model_comparison.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
