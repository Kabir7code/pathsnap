"""HMM / Viterbi route reconstruction: decoder tuning and evaluation.

1. Out-of-fold (OOF) classifier probabilities on the training trips
   (5-fold GroupKFold with the final model configuration), so the decoder is
   tuned on probabilities that look like real test-time probabilities.
2. Bayesian optimisation (Optuna TPE) of decoder parameters
   (beta, emission_weight, unreachable_logp) on training trips only.
   Objective: mean point-level Top-1 accuracy after decoding.
3. Student-test comparison per trip: classifier-only vs classifier+HMM
   (default decoder params) vs classifier+HMM (tuned params).
Outputs: results/05_hmm_viterbi/hmm_trials.csv, results/05_hmm_viterbi/hmm_student_test.csv,
         results/05_hmm_viterbi/hmm_comparison.png, models/hmm_params.json
"""
import argparse
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import CFG
from pathsnap.data import load_trip
from pathsnap.dataset import ART, MODELS, subsample_points
from pathsnap.features import FEATURES
from pathsnap.hmm import RoadGraph, reconstruct_route, viterbi
from pathsnap.labels import align_track_to_route
from pathsnap.metrics import continuity, route_overlap
from pathsnap.models import make_model


def lists_from_frame(g, n_points, p):
    """Rebuild per-point candidate lists from the feature table of one trip."""
    cands = [np.empty(0, int)] * n_points
    ts = [np.empty(0)] * n_points
    probs = [np.empty(0)] * n_points
    pts = g.point.to_numpy()
    bounds = np.flatnonzero(np.diff(pts)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(pts)]):
        i = pts[s]
        cands[i] = g.arc.to_numpy()[s:e]; ts[i] = g.proj_t.to_numpy()[s:e]; probs[i] = p[s:e]
    return cands, ts, probs


def decode_and_score(trip, true_arc, cands, ts, probs, graph, hp):
    cls = np.array([c[np.argmax(q)] if len(c) else -1 for c, q in zip(cands, probs)])
    t0 = time.time()
    hmm = viterbi(trip, cands, ts, probs, graph, beta=hp["beta"], emission_weight=hp["emission_weight"],
                  unreachable_logp=hp["unreachable_logp"], backward_tol=hp["backward_tolerance"],
                  gap_seconds=hp.get("gap_seconds", 30.0))
    dec_s = time.time() - t0
    lab = true_arc >= 0
    r_cls, _ = reconstruct_route(cls, graph)
    r_hmm, gaps = reconstruct_route(hmm, graph)
    res = {"cls_top1": (cls[lab] == true_arc[lab]).mean(), "hmm_top1": (hmm[lab] == true_arc[lab]).mean(),
           "cls_continuity": continuity(cls, trip.arcs), "hmm_continuity": continuity(hmm, trip.arcs),
           "decode_s": dec_s, "route_gaps": gaps}
    res.update({f"cls_{k}": v for k, v in route_overlap(r_cls, trip.route).items()})
    res.update({f"hmm_{k}": v for k, v in route_overlap(r_hmm, trip.route).items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max_points", type=int, default=5000, help="skip huge train trips during tuning (speed)")
    args = ap.parse_args()
    seed = CFG["seed"]
    meta = json.load(open(os.path.join(MODELS, "final_model.json")))
    model_name, params = meta["model"], meta["params"]

    # ---------- 1. OOF probabilities on training trips --------------------
    lab = pd.read_csv("results/01_eda_preprocessing/labelling_train.csv", dtype={"trip": str})
    good = lab[(lab.labelled_frac >= CFG["min_trip_label_frac"]) & (lab.points <= args.max_points)].trip.tolist()
    full = pd.read_parquet(os.path.join(ART, "features_train.parquet"))
    full = full[full.trip.isin(lab[lab.labelled_frac >= CFG["min_trip_label_frac"]].trip)]
    full["oof"] = np.nan
    for f, (i_tr, i_va) in enumerate(GroupKFold(5).split(full, groups=full.trip)):
        fit = subsample_points(full.iloc[i_tr][full.iloc[i_tr].true_arc >= 0], 300_000, seed + f)
        m = make_model(model_name, seed, **params).fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())
        full.iloc[i_va, full.columns.get_loc("oof")] = m.predict_proba(full.iloc[i_va][FEATURES].to_numpy())[:, 1]
        print("oof fold", f, flush=True)

    tune = []
    for tid in good:
        trip = load_trip(os.path.join(CFG["data_dir"], "train"), tid)
        ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
        g = full[full.trip == tid]
        c, t, p = lists_from_frame(g, len(ta), g.oof.to_numpy())
        tune.append((trip, ta, c, t, p, RoadGraph(trip, CFG["hmm"]["dijkstra_cutoff"])))
    print(f"tuning on {len(tune)} training trips", flush=True)

    # ---------- 2. Bayesian optimisation of decoder parameters ------------
    def objective(trial):
        hp = dict(CFG["hmm"])
        hp.update(beta=trial.suggest_float("beta", 1.0, 100.0, log=True),
                  emission_weight=trial.suggest_float("emission_weight", 0.2, 3.0),
                  unreachable_logp=trial.suggest_float("unreachable_logp", -60.0, -5.0))
        rs = [decode_and_score(*x[:5], x[5], hp) for x in tune]
        trial.set_user_attr("route_f1", float(np.mean([r["hmm_route_f1"] for r in rs])))
        trial.set_user_attr("cls_top1", float(np.mean([r["cls_top1"] for r in rs])))
        v = float(np.mean([r["hmm_top1"] for r in rs]))
        print(f"hmm trial {trial.number}: top1={v:.4f} routeF1={trial.user_attrs['route_f1']:.4f}", flush=True)
        return v

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=6),
                                storage="sqlite:///results/05_hmm_viterbi/optuna_hmm.db", study_name="pathsnap_hmm", load_if_exists=True)
    study.enqueue_trial({k: CFG["hmm"][k] for k in ["beta", "emission_weight", "unreachable_logp"]})
    left = args.trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    study.trials_dataframe().to_csv("results/05_hmm_viterbi/hmm_trials.csv", index=False)
    tuned = dict(CFG["hmm"]); tuned.update(study.best_params)
    json.dump(tuned, open(os.path.join(MODELS, "hmm_params.json"), "w"), indent=1)
    print("best decoder params", tuned)

    # ---------- 3. Student-test comparison --------------------------------
    model = joblib.load(os.path.join(MODELS, "final.joblib"))
    te = pd.read_parquet(os.path.join(ART, "features_student_test.parquet"))
    rows = []
    for tid, g in te.groupby("trip"):
        trip = load_trip(os.path.join(CFG["data_dir"], "student_test"), tid)
        ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
        p = model.predict_proba(g[FEATURES].to_numpy())[:, 1]
        c, t, pr = lists_from_frame(g, len(ta), p)
        graph = RoadGraph(trip, CFG["hmm"]["dijkstra_cutoff"])
        for variant, hp in [("default", CFG["hmm"]), ("tuned", tuned)]:
            r = decode_and_score(trip, ta, c, t, pr, graph, hp)
            r.update(trip=tid, decoder=variant, points=len(ta))
            rows.append(r)
            print(tid, variant, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("results/05_hmm_viterbi/hmm_student_test.csv", index=False)
    summ = res.groupby("decoder")[[c for c in res if c.startswith(("cls_", "hmm_")) or c == "decode_s"]].mean()
    summ.to_csv("results/05_hmm_viterbi/hmm_summary.csv")
    print(summ.round(4).T.to_string())

    # plot: per-trip top1 & route F1, classifier vs HMM (tuned)
    t = res[res.decoder == "tuned"].sort_values("trip")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(t)); w = .4
    for a, (m, title) in zip(ax, [("top1", "Point Top-1 accuracy"), ("route_f1", "Route F1 (arc overlap)"),
                                  ("continuity", "Route continuity")]):
        a.bar(x - w / 2, t[f"cls_{m}"], w, label="classifier only")
        a.bar(x + w / 2, t[f"hmm_{m}"], w, label="classifier + HMM")
        a.set_xticks(x, t.trip.str[-2:]); a.set(title=title, xlabel="student-test trip", ylim=(0.5, 1.0))
    ax[0].legend(); fig.tight_layout(); fig.savefig("results/05_hmm_viterbi/hmm_comparison.png", dpi=130)


if __name__ == "__main__":
    main()
