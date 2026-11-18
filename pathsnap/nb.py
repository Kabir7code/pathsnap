"""Helpers used by the Jupyter notebooks in notebooks/.

Every function here runs the real pipeline (no numbers are hard-coded):
feature building, GroupKFold cross-validation, student-test evaluation,
plots, and classifier-vs-HMM decoding.
"""
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from .config import CFG, ROOT
from .dataset import ART, MODELS, load_split, subsample_points
from .experiment import cross_validate, evaluate
from .features import FEATURES
from .metrics import continuity

NB_RESULTS = os.path.join(ROOT, "results", "02_model_training", "notebook_runs")


def show(fig=None):
    """Display a figure inline regardless of the active matplotlib backend."""
    from IPython.display import display
    fig = fig or plt.gcf()
    display(fig)
    plt.close(fig)


# ----------------------------------------------------------------- data
def ensure_features():
    """Build the candidate/feature tables if they are missing (about 35 s)."""
    need = [os.path.join(ART, f"{k}_{s}.parquet") for k in ("features", "points") for s in ("train", "student_test")]
    lab = [os.path.join(ROOT, "results", "01_eda_preprocessing", f"labelling_{s}.csv") for s in ("train", "student_test")]
    if all(os.path.exists(p) for p in need + lab):
        print("feature tables found in data/features/")
        return
    if not os.path.isdir(os.path.join(CFG["data_dir"], "train")):
        raise FileNotFoundError("Put the dataset in data/train/<id>/ and data/student_test/<id>/ first.")
    print("building feature tables (about 35 s)...")
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "01_build_dataset.py")], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    print(f"done in {time.time() - t0:.0f}s")


def load_data():
    ensure_features()
    tr, tr_pts = load_split("train")
    te, te_pts = load_split("student_test", exclude_bad=False)
    print(f"train: {tr.trip.nunique()} trips, {len(tr):,} candidate rows, positive rate {tr.label.mean():.3f}")
    print(f"student_test: {te.trip.nunique()} trips, {len(te):,} candidate rows")
    return tr, tr_pts, te, te_pts


# ----------------------------------------------------------- experiments
def run_cv(make, tr, tr_pts, max_rows=300_000, folds=5):
    t0 = time.time()
    res = cross_validate(make, tr, tr_pts, folds=folds, max_rows=max_rows, seed=CFG["seed"], verbose=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"} for r in res])
    df.index.name = "fold"
    print(f"CV time {time.time() - t0:.0f}s | mean macro F1 {df.macro_f1.mean():.4f} ± {df.macro_f1.std():.4f}")
    return df


def fit_and_evaluate(make, tr, tr_pts, te, te_pts, max_rows=300_000):
    """Fit on the training split; score on the training sample (for overfitting checks) and on student test."""
    fit = subsample_points(tr, max_rows, CFG["seed"])
    model = make()
    t0 = time.time()
    model.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())
    fit_s = time.time() - t0
    fit_pts = tr_pts.merge(fit[["trip", "point"]].drop_duplicates(), on=["trip", "point"])
    train_m, _ = evaluate(model, fit, fit_pts)
    test_m, pred = evaluate(model, te, te_pts)
    conts = []
    from .data import load_trip
    for tid, g in pred.groupby("trip"):
        arcs = load_trip(os.path.join(CFG["data_dir"], "student_test"), tid, with_route=False).arcs
        conts.append(continuity(g.sort_values("point").pred_arc.to_numpy(), arcs))
    test_m["continuity"] = float(np.mean(conts))
    test_m["fit_s"] = fit_s
    p_test = model.predict_proba(te[FEATURES].to_numpy())[:, 1]
    print(f"fit {fit_s:.1f}s on {len(fit):,} rows")
    return model, train_m, test_m, p_test


def summary_table(name, cv, train_m, test_m):
    keys = ["macro_f1", "precision", "recall", "log_loss", "top1"]
    t = pd.DataFrame({
        "train (fit sample)": [train_m[k] for k in keys],
        "CV mean (5 folds)": [cv[k].mean() for k in keys],
        "student test": [test_m[k] for k in keys],
    }, index=["macro F1", "precision", "recall", "log loss", "Top-1 accuracy"])
    t.columns.name = name
    return t.round(4)


def save_results(name, cv, train_m, test_m, extra=None):
    os.makedirs(NB_RESULTS, exist_ok=True)
    out = {"model": name, "cv_folds": cv.to_dict(orient="records"), "train": train_m, "student_test": test_m}
    out.update(extra or {})
    json.dump(out, open(os.path.join(NB_RESULTS, f"{name}.json"), "w"), indent=1, default=float)
    print("saved", os.path.relpath(os.path.join(NB_RESULTS, f"{name}.json"), ROOT))


# ----------------------------------------------------------------- plots
def plot_cv_folds(cvs: dict):
    """cvs: {model_name: cv DataFrame}"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, cv in cvs.items():
        ax[0].plot(cv.index, cv.macro_f1, "o-", label=name)
        ax[1].plot(cv.index, cv.top1, "o-", label=name)
    ax[0].set(title="CV macro F1 per fold", xlabel="fold", ylabel="macro F1")
    ax[1].set(title="CV Top-1 accuracy per fold", xlabel="fold", ylabel="Top-1")
    ax[0].legend(); fig.tight_layout(); show(fig)


def plot_overfit(rows: dict):
    """rows: {model_name: (train_m, cv_df, test_m)} -> grouped bars of macro F1 and Top-1."""
    names = list(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, key, title in [(ax[0], "macro_f1", "macro F1"), (ax[1], "top1", "Top-1 accuracy")]:
        x = np.arange(len(names)); w = .27
        a.bar(x - w, [rows[n][0][key] for n in names], w, label="train (fit sample)")
        a.bar(x, [rows[n][1][key].mean() for n in names], w, label="CV")
        a.bar(x + w, [rows[n][2][key] for n in names], w, label="student test")
        a.set_xticks(x, names); a.set(title=f"Overfitting check: {title}", ylim=(0.75, 1.0))
    ax[0].legend(loc="lower left"); fig.tight_layout(); show(fig)


def plot_confusion(models_cms: dict):
    n = len(models_cms)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.3))
    axs = np.atleast_1d(axs)
    for a, (name, cm) in zip(axs, models_cms.items()):
        cm = np.array(cm); cmn = cm / cm.sum(axis=1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=9,
                       color="white" if cmn[i, j] > .5 else "black")
        a.set_xticks([0, 1], ["not ref", "ref arc"]); a.set_yticks([0, 1], ["not ref", "ref arc"])
        a.set(title=f"{name} (student test)", xlabel="predicted", ylabel="actual")
    fig.tight_layout(); show(fig)


def plot_calibration(y, probs: dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in probs.items():
        fy, fx = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(fx, fy, "o-", ms=3, label=name)
    ax.set(xlabel="predicted probability", ylabel="observed frequency", title="Calibration (student test)")
    ax.legend(); fig.tight_layout(); show(fig)


def importance(model, te, n=20_000, seed=42):
    """Native importances where the model has them, otherwise permutation importance (drop in log loss)."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "booster_"):
        g = est.booster_.feature_importance("gain"); return pd.Series(g / g.sum(), FEATURES), "share of split gain"
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, FEATURES), "impurity importance"
    if hasattr(est, "coef_"):
        return pd.Series(est.coef_[0], FEATURES), "coefficient (standardised features)"
    s = te.sample(min(n, len(te)), random_state=seed)
    r = permutation_importance(model, s[FEATURES].to_numpy(), s.label.to_numpy(), scoring="neg_log_loss",
                               n_repeats=3, random_state=seed, n_jobs=1)
    return pd.Series(r.importances_mean, FEATURES), "permutation importance (log-loss increase)"


def plot_importance(model, te, name):
    imp, kind = importance(model, te)
    imp = imp.reindex(imp.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(imp.index, imp.values, color=["#d62728" if v < 0 else "#1f5fd1" for v in imp.values])
    ax.set(title=f"{name}: feature importance", xlabel=kind)
    fig.tight_layout(); show(fig)
    return imp


# ------------------------------------------------------------------- HMM
def _hmm_tools():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from importlib import import_module
    return import_module("04_hmm_eval")


def hmm_compare(model, te, name, hp=None):
    """Classifier-only vs classifier + HMM/Viterbi on every student-test trip."""
    from .data import load_trip
    from .hmm import RoadGraph
    from .labels import align_track_to_route
    tools = _hmm_tools()
    hp = hp or json.load(open(os.path.join(MODELS, "hmm_params.json")))
    rows = []
    for tid, g in te.groupby("trip"):
        trip = load_trip(os.path.join(CFG["data_dir"], "student_test"), tid)
        ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
        p = model.predict_proba(g[FEATURES].to_numpy())[:, 1]
        c, t, pr = tools.lists_from_frame(g, len(ta), p)
        r = tools.decode_and_score(trip, ta, c, t, pr, RoadGraph(trip, hp["dijkstra_cutoff"]), hp)
        r.update(trip=tid, model=name)
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["cls_top1", "hmm_top1", "cls_route_f1", "hmm_route_f1", "cls_continuity", "hmm_continuity"]
    print(f"{name}: mean over {len(df)} trips")
    print(df[cols].mean().round(4).to_string())
    return df


def plot_hmm(per_trip: pd.DataFrame, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    t = per_trip.sort_values("trip"); x = np.arange(len(t)); w = .4
    for a, (m, ttl) in zip(ax, [("top1", "Point Top-1"), ("route_f1", "Route F1"), ("continuity", "Continuity")]):
        a.bar(x - w / 2, t[f"cls_{m}"], w, label="classifier only", color="#f28e1c")
        a.bar(x + w / 2, t[f"hmm_{m}"], w, label="classifier + HMM", color="#d62728")
        a.set_xticks(x, t.trip.str[-2:]); a.set(title=ttl, xlabel="student-test trip", ylim=(0.5, 1.02))
    ax[0].legend(loc="lower left"); fig.suptitle(title); fig.tight_layout(); show(fig)
