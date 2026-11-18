"""Bayesian hyperparameter optimisation (Optuna TPE sampler) for the chosen model.

Objective : mean macro F1 over 3-fold GroupKFold on the 80 training trips
            (trip-grouped, so no point of a validation trip is ever fitted).
Record    : every trial is stored in results/04_bayesian_optimisation/optuna_<model>.db (Optuna study)
            and exported to results/04_bayesian_optimisation/optuna_trials_<model>.csv.
Afterwards: the best configuration and the untuned default are both
            evaluated with the SAME 5-fold CV and the SAME student-test
            protocol as scripts/02_train_models.py -> results/04_bayesian_optimisation/tuned_vs_untuned.csv
            The tuned model is saved as models/final.joblib.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import CFG
from pathsnap.dataset import load_split
from pathsnap.experiment import cross_validate, fit_and_test
from pathsnap.models import make_model


def space(trial, name):
    if name == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if name == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
        )
    if name == "extratrees":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
        )
    raise ValueError(f"no search space for {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--search_rows", type=int, default=150_000)
    args = ap.parse_args()
    seed = CFG["seed"]
    tr, tr_pts = load_split("train")
    te, te_pts = load_split("student_test", exclude_bad=False)

    def objective(trial):
        params = space(trial, args.model)
        folds = cross_validate(lambda: make_model(args.model, seed, **params), tr, tr_pts,
                               folds=3, max_rows=args.search_rows, seed=seed, verbose=False)
        trial.set_user_attr("cv_top1", float(np.mean([f["top1"] for f in folds])))
        trial.set_user_attr("cv_log_loss", float(np.mean([f["log_loss"] for f in folds])))
        f1 = float(np.mean([f["macro_f1"] for f in folds]))
        print(f"trial {trial.number}: F1={f1:.4f} {params}", flush=True)
        return f1

    storage = f"sqlite:///results/04_bayesian_optimisation/optuna_{args.model}.db"
    study = optuna.create_study(study_name=f"pathsnap_{args.model}", direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
                                storage=storage, load_if_exists=True)
    t0 = time.time()
    remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(f"results/04_bayesian_optimisation/optuna_trials_{args.model}.csv", index=False)
    best = study.best_params
    print("best", study.best_value, best)

    # optimisation history plot
    df = study.trials_dataframe()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.number, df.value, "o", alpha=.6, label="trial")
    ax.plot(df.number, df.value.cummax(), "-", color="C3", label="best so far")
    ax.set(xlabel="trial", ylabel="3-fold CV macro F1", title=f"Bayesian optimisation ({args.model}, TPE)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"results/04_bayesian_optimisation/optuna_history_{args.model}.png", dpi=130)

    rows = []
    for label, params in [("untuned", {}), ("tuned", best)]:
        print(f"=== {label}", flush=True)
        make = lambda p=params: make_model(args.model, seed, **p)
        folds = cross_validate(make, tr, tr_pts, folds=5, seed=seed)
        model, test = fit_and_test(make, tr, te, te_pts, seed=seed)
        rows.append({"variant": label, "params": json.dumps(params),
                     "cv_macro_f1": np.mean([f["macro_f1"] for f in folds]),
                     "cv_macro_f1_std": np.std([f["macro_f1"] for f in folds]),
                     "cv_top1": np.mean([f["top1"] for f in folds]),
                     "cv_log_loss": np.mean([f["log_loss"] for f in folds]),
                     "test_top1": test["top1"], "test_macro_f1": test["macro_f1"],
                     "test_log_loss": test["log_loss"], "test_continuity": test["continuity"],
                     "fit_s": test["fit_s"], "infer_ms_per_point": test["infer_ms_per_point"]})
        if label == "tuned":
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/final.joblib", compress=3)
            json.dump({"model": args.model, "params": best, "seed": seed, "search_seconds": search_s,
                       "trials": len(study.trials), "best_cv3_macro_f1": study.best_value},
                      open("models/final_model.json", "w"), indent=1)
    out = pd.DataFrame(rows)
    out.to_csv("results/04_bayesian_optimisation/tuned_vs_untuned.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"search time: {search_s:.0f}s")


if __name__ == "__main__":
    main()
