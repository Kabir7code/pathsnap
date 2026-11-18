"""Bayesian hyperparameter tuning of the selected final model, with the HMM in the loop.

The first search (scripts/05_tune_lightgbm_bayesian.py) optimised the classifier's CV macro F1 alone and
found no gain. The deployed system is classifier + HMM/Viterbi, so this search optimises the
metric that is actually delivered:

  objective = mean per-trip HMM Top-1 on training trips, using OUT-OF-FOLD probabilities
              (3-fold GroupKFold by trip; each trip is decoded with probabilities from a model
              that never saw it), decoded with the model's own tuned decoder.

DEVELOPMENT DATA ONLY: training trips for the search; the 10 student-test trips for the
final tuned-vs-untuned comparison. The evaluator-test partition is never read.

Afterwards untuned and tuned are compared with the same full protocol:
  * classifier: 5-fold GroupKFold CV macro F1 / Top-1 / log loss + student-test metrics
  * system: classifier-only vs + HMM on 78 OOF training trips + 10 student-test trips,
    with a paired bootstrap CI.
The tuned model replaces the deployed one (models/final.joblib) only if it is
SIGNIFICANTLY better on the primary metric (mean HMM Top-1 over the 88 development trips:
the 95% bootstrap CI of the paired difference must be entirely above 0). Otherwise the
current deployed model is kept, so noise never changes the deployed system.

Usage: python scripts/11_retune_lightgbm_with_hmm.py --model lightgbm|rf [--trials 20]
Outputs: results/09_final_tuning_lightgbm_with_hmm/
"""
import argparse
import json
import os
import shutil
import sys
import time
from importlib import import_module

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from pathsnap.config import CFG
from pathsnap.data import load_trip
from pathsnap.dataset import ART, MODELS, load_split, subsample_points
from pathsnap.experiment import cross_validate, fit_and_test
from pathsnap.features import FEATURES
from pathsnap.hmm import RoadGraph
from pathsnap.labels import align_track_to_route
from pathsnap.models import make_model

H = import_module("04_hmm_eval")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = CFG["seed"]
OUT = "results/09_final_tuning_lightgbm_with_hmm"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def space(trial, name):
    if name == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if name == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 8, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]))
    raise ValueError(name)


def build(name, params):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        p = dict(n_estimators=200, max_depth=16, min_samples_leaf=20, max_features="sqrt"); p.update(params)
        return RandomForestClassifier(n_jobs=-1, random_state=SEED, **p)
    return make_model(name, SEED, **params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lightgbm", "rf"])
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    sel = json.load(open("results/08_final_model_selection/decision.json"))
    label = {"lightgbm": "LightGBM (tuned)", "rf": "Random Forest"}[args.model]
    hp = sel["own_decoder_params"][label]
    base_params = json.load(open(os.path.join(MODELS, "final_model.json")))["params"] if args.model == "lightgbm" else {}

    lab = pd.read_csv("results/01_eda_preprocessing/labelling_train.csv", dtype={"trip": str})
    good = lab[lab.labelled_frac >= CFG["min_trip_label_frac"]]
    full = pd.read_parquet(os.path.join(ART, "features_train.parquet"))
    full = full[full.trip.isin(good.trip)].reset_index(drop=True)
    te_full = pd.read_parquet(os.path.join(ART, "features_student_test.parquet"))
    bundles = {}
    for split, tids in [("train", good.trip), ("student_test", sorted(te_full.trip.unique()))]:
        for tid in tids:
            trip = load_trip(os.path.join(CFG["data_dir"], split), tid)
            ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
            bundles[(split, tid)] = (trip, ta, RoadGraph(trip, hp["dijkstra_cutoff"]))
    tune_trips = [t for t in good.trip if len(bundles[("train", t)][1]) <= 3000]
    log(f"model={args.model} | decoder={ {k: round(v, 3) for k, v in hp.items()} } | tuning trips={len(tune_trips)}")

    def oof_probs(params, folds, rows):
        oof = np.full(len(full), np.nan)
        for f, (i_tr, i_va) in enumerate(GroupKFold(folds).split(full, groups=full.trip)):
            fit = subsample_points(full.iloc[i_tr][full.iloc[i_tr].true_arc >= 0], rows, SEED + f)
            m = build(args.model, params).fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())
            oof[i_va] = m.predict_proba(full.iloc[i_va][FEATURES].to_numpy())[:, 1]
        return oof

    def decode(df, probs, split, tids, decoder):
        out = []
        for tid in tids:
            trip, ta, graph = bundles[(split, tid)]
            g = df[df.trip == tid]
            c, t, p = H.lists_from_frame(g, len(ta), probs[g.index.to_numpy()])
            r = H.decode_and_score(trip, ta, c, t, p, graph, decoder)
            r.update(trip=tid, split=split); out.append(r)
        return pd.DataFrame(out)

    # ------------------------------------------------ Bayesian search
    def objective(trial):
        params = space(trial, args.model)
        t = time.time()
        res = decode(full, oof_probs(params, 3, 150_000), "train", tune_trips, hp)
        trial.set_user_attr("cls_top1", float(res.cls_top1.mean()))
        trial.set_user_attr("hmm_route_f1", float(res.hmm_route_f1.mean()))
        v = float(res.hmm_top1.mean())
        log(f"trial {trial.number:2d}: HMM Top-1 {v:.4f} (classifier {res.cls_top1.mean():.4f}) {time.time() - t:.0f}s {params}")
        return v

    study = optuna.create_study(direction="maximize", study_name=f"final_{args.model}_hmm_objective",
                                storage=f"sqlite:///{OUT}/optuna_{args.model}.db", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6))
    if not study.trials:
        study.enqueue_trial(base_params if args.model == "lightgbm" else
                            dict(n_estimators=200, max_depth=16, min_samples_leaf=20, max_features="sqrt"))
    left = args.trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    done = [t for t in study.trials if t.datetime_complete]
    search_s = (max(t.datetime_complete for t in done) - min(t.datetime_start for t in done)).total_seconds()  # from the saved trials
    tri = study.trials_dataframe(); tri.to_csv(f"{OUT}/trials_{args.model}.csv", index=False)
    best = study.best_params
    log("best", round(study.best_value, 4), best)

    # --------------------------------- full comparison: untuned vs tuned
    tr, tr_pts = load_split("train")
    te, te_pts = load_split("student_test", exclude_bad=False)
    variants = {"untuned": base_params if args.model == "lightgbm" else {}, "tuned": best}
    if args.model == "lightgbm":
        variants = {"current (tuned for classifier F1)": base_params, "re-tuned (HMM objective)": best}
    rows, per, models = [], [], {}
    for vname, params in variants.items():
        log("evaluating", vname)
        folds = cross_validate(lambda p=params: build(args.model, p), tr, tr_pts, folds=5, seed=SEED, verbose=False)
        model, test = fit_and_test(lambda p=params: build(args.model, p), tr, te, te_pts, seed=SEED)
        models[vname] = model
        oof = oof_probs(params, 5, 300_000)
        p_te = model.predict_proba(te_full[FEATURES].to_numpy())[:, 1]
        d_tr = decode(full, oof, "train", list(good.trip), hp)
        d_te = decode(te_full, p_te, "student_test", sorted(te_full.trip.unique()), hp)
        d = pd.concat([d_tr, d_te]); d["variant"] = vname; per.append(d)
        rows.append({"variant": vname, "params": json.dumps(params, default=str),
                     "cv_macro_f1": np.mean([f["macro_f1"] for f in folds]), "cv_macro_f1_std": np.std([f["macro_f1"] for f in folds]),
                     "cv_top1": np.mean([f["top1"] for f in folds]), "cv_log_loss": np.mean([f["log_loss"] for f in folds]),
                     "test_top1": test["top1"], "test_macro_f1": test["macro_f1"], "test_log_loss": test["log_loss"],
                     "test_infer_ms_per_point": test["infer_ms_per_point"], "fit_s": test["fit_s"],
                     "train_oof_cls_top1": d_tr.cls_top1.mean(), "train_oof_hmm_top1": d_tr.hmm_top1.mean(),
                     "train_oof_hmm_route_f1": d_tr.hmm_route_f1.mean(),
                     "test_cls_top1_trip_mean": d_te.cls_top1.mean(), "test_hmm_top1_trip_mean": d_te.hmm_top1.mean(),
                     "test_hmm_route_f1_trip_mean": d_te.hmm_route_f1.mean(),
                     "dev88_hmm_top1": d.hmm_top1.mean(), "dev88_hmm_route_f1": d.hmm_route_f1.mean()})
    comp = pd.DataFrame(rows).set_index("variant")
    comp.to_csv(f"{OUT}/tuned_vs_untuned_{args.model}.csv")
    per = pd.concat(per); per.to_csv(f"{OUT}/per_trip_{args.model}.csv", index=False)

    v0, v1 = list(variants)
    a = per[per.variant == v0].set_index(["split", "trip"]).hmm_top1
    b = per[per.variant == v1].set_index(["split", "trip"]).hmm_top1.reindex(a.index)
    diff = (b - a).to_numpy()
    boots = np.random.default_rng(SEED).choice(diff, (10_000, len(diff))).mean(axis=1)
    ci = np.percentile(boots, [2.5, 97.5])
    deploy = v1 if ci[0] > 0 else v0
    decision = {"model": args.model, "search": {"trials": len(study.trials), "seconds": search_s, "objective":
                "mean per-trip HMM Top-1, 3-fold OOF on training trips <= 3000 points", "best_value": study.best_value,
                "best_params": best}, "tuned_minus_untuned_dev88_hmm_top1": float(diff.mean()),
                "ci95": [float(ci[0]), float(ci[1])], "deployed_variant": deploy,
                "deployed_params": variants[deploy], "decoder_params": hp, "evaluator_test_used": False}
    json.dump(decision, open(f"{OUT}/decision_{args.model}.json", "w"), indent=1, default=str)

    # ------------------------------------------- deploy the chosen model
    if deploy == v0:
        log("tuned model is not significantly better -> deployed model unchanged")
    else:
        _deploy(models, deploy, variants, args, study, search_s, decision, hp)
    _plots(tri, study, comp, args)
    print(comp.T.to_string())
    print(json.dumps(decision, indent=1, default=str))


def _deploy(models, deploy, variants, args, study, search_s, decision, hp):
    joblib.dump(models[deploy], os.path.join(MODELS, "final.joblib"), compress=3)
    json.dump({"model": args.model, "params": variants[deploy], "seed": SEED, "search_seconds": search_s,
               "trials": len(study.trials), "best_cv3_macro_f1": None,
               "selection": "scripts/10_final_model_selection.py", "tuning": "scripts/11_retune_lightgbm_with_hmm.py",
               "objective": decision["search"]["objective"]},
              open(os.path.join(MODELS, "final_model.json"), "w"), indent=1, default=str)
    json.dump(hp, open(os.path.join(MODELS, "hmm_params.json"), "w"), indent=1)
    log("deployed", deploy, "-> models/final.joblib, final_model.json, hmm_params.json")


def _plots(tri, study, comp, args):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(tri.number, tri.value, "o", label="trial")
    ax.plot(tri.number, tri.value.cummax(), "r-", label="best so far")
    ax.set(xlabel="trial", ylabel="mean HMM Top-1 (OOF training trips)", title=f"Bayesian optimisation ({args.model}, HMM objective)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/optuna_history_{args.model}.png", dpi=130); plt.close(fig)

    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(6, 3.4)); ax.barh(list(imp)[::-1], list(imp.values())[::-1])
    ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance"); fig.tight_layout()
    fig.savefig(f"{OUT}/param_importance_{args.model}.png", dpi=130); plt.close(fig)

    ms = [("cv_macro_f1", "CV macro F1"), ("test_top1", "test Top-1\n(classifier, pooled)"),
          ("test_hmm_top1_trip_mean", "test Top-1\n+ HMM (trip mean)"), ("dev88_hmm_top1", "88 dev trips\nTop-1 + HMM"),
          ("dev88_hmm_route_f1", "88 dev trips\nroute F1 + HMM")]
    fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(ms)); w = .38
    for i, v in enumerate(comp.index):
        vals = [comp.loc[v, m] for m, _ in ms]
        ax.bar(x + (i - .5) * w, vals, w, label=v)
        for xi, val in zip(x, vals):
            ax.text(xi + (i - .5) * w, val + .002, f"{val:.4f}", ha="center", fontsize=7)
    ax.set_xticks(x, [l for _, l in ms]); ax.set(ylim=(0.85, 0.98), title=f"{args.model}: before vs after tuning")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/tuned_vs_untuned_{args.model}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
