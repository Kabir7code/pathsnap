"""Final model selection: LightGBM (tuned) vs Random Forest, before and after HMM/Viterbi.

Uses DEVELOPMENT DATA ONLY (training trips + student-test trips). The evaluator-test
partition is NOT read by this script (brief: it must not be used to choose the model).

Evidence base (88 trips instead of 10):
  * 78 training trips: out-of-fold probabilities from 5-fold GroupKFold (each trip is
    scored by a model that never saw it)
  * 10 student-test trips: models fitted on the training split
Fairness: each model gets its OWN HMM decoder parameters, tuned with Optuna on its own
out-of-fold probabilities (training trips only). Results with the shared, deployed
decoder parameters are reported too.

PRE-DECLARED DECISION RULE (fixed before results are seen):
  primary   = mean per-trip HMM Top-1 over the 88 development trips (each model with its own decoder)
  if the 95% bootstrap CI of the paired difference contains 0 -> statistical tie ->
  secondary = mean per-trip HMM route F1; if that CI also contains 0 ->
  tertiary  = practical criteria: probability quality (log loss), inference speed, model size.

Outputs: results/08_final_model_selection/  (per-trip CSV, summary CSV, decision JSON, plots)
"""
import json
import os
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
from scipy.stats import wilcoxon
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from pathsnap.config import CFG
from pathsnap.data import load_trip
from pathsnap.dataset import ART, MODELS, subsample_points
from pathsnap.features import FEATURES
from pathsnap.hmm import RoadGraph
from pathsnap.labels import align_track_to_route
from pathsnap.models import make_model

H = import_module("04_hmm_eval")
OUT = "results/08_final_model_selection"
os.makedirs(OUT, exist_ok=True)
SEED = CFG["seed"]
FINAL_PARAMS = json.load(open(os.path.join(MODELS, "final_model.json")))["params"]
CANDIDATES = {
    "LightGBM (tuned)": dict(make=lambda: make_model("lightgbm", SEED, **FINAL_PARAMS), file="final.joblib"),
    "Random Forest": dict(make=lambda: make_model("rf", SEED), file="rf.joblib"),
}
SHARED_HP = json.load(open(os.path.join(MODELS, "hmm_params.json")))
optuna.logging.set_verbosity(optuna.logging.WARNING)
t_start = time.time()


def log(*a):
    print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------ load data
lab = pd.read_csv("results/01_eda_preprocessing/labelling_train.csv", dtype={"trip": str})
good = lab[lab.labelled_frac >= CFG["min_trip_label_frac"]]
full_tr = pd.read_parquet(os.path.join(ART, "features_train.parquet"))
full_tr = full_tr[full_tr.trip.isin(good.trip)].reset_index(drop=True)
full_te = pd.read_parquet(os.path.join(ART, "features_student_test.parquet"))


def trip_bundle(split, tid):
    trip = load_trip(os.path.join(CFG["data_dir"], split), tid)
    ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
    return trip, ta, RoadGraph(trip, SHARED_HP["dijkstra_cutoff"])


log("loading trips and road graphs")
TRAIN = {tid: trip_bundle("train", tid) for tid in good.trip}
TEST = {tid: trip_bundle("student_test", tid) for tid in sorted(full_te.trip.unique())}

# --------------------------------------------- probabilities per model
probs_tr, probs_te, ll = {}, {}, {}
for name, spec in CANDIDATES.items():
    oof = np.full(len(full_tr), np.nan)
    for f, (i_tr, i_va) in enumerate(GroupKFold(5).split(full_tr, groups=full_tr.trip)):
        fit = subsample_points(full_tr.iloc[i_tr][full_tr.iloc[i_tr].true_arc >= 0], 300_000, SEED + f)
        m = spec["make"]().fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())
        oof[i_va] = m.predict_proba(full_tr.iloc[i_va][FEATURES].to_numpy())[:, 1]
        log(name, "OOF fold", f)
    probs_tr[name] = oof
    model = joblib.load(os.path.join(MODELS, spec["file"]))
    t0 = time.time()
    probs_te[name] = model.predict_proba(full_te[FEATURES].to_numpy())[:, 1]
    infer = time.time() - t0
    lt = full_tr.true_arc.to_numpy() >= 0
    ll[name] = {"oof_log_loss": log_loss(full_tr.label[lt], np.clip(oof[lt], 1e-7, 1 - 1e-7)),
                "test_log_loss": log_loss(full_te.label, np.clip(probs_te[name], 1e-7, 1 - 1e-7)),
                "test_infer_ms_per_point": 1000 * infer / full_te[["trip", "point"]].drop_duplicates().shape[0],
                "model_file_mb": os.path.getsize(os.path.join(MODELS, spec["file"])) / 1e6}


def lists(df, p, tid, n):
    g = df[df.trip == tid]
    return H.lists_from_frame(g, n, p[g.index.to_numpy()])


def run(name, split, hp):
    """Decode every trip of a split with the given decoder params."""
    df, P, B = (full_tr, probs_tr[name], TRAIN) if split == "train_oof" else (full_te, probs_te[name], TEST)
    rows = []
    for tid, (trip, ta, graph) in B.items():
        c, t, p = lists(df, P, tid, len(ta))
        r = H.decode_and_score(trip, ta, c, t, p, graph, hp)
        r.update(model=name, split=split, trip=tid, points=int((ta >= 0).sum()))
        rows.append(r)
    return rows


# --------------------------- per-model decoder tuning (train trips only)
TUNE_TRIPS = [tid for tid in good.trip if len(TRAIN[tid][1]) <= 3000]
own_hp = {}
for name in CANDIDATES:
    def objective(trial):
        hp = dict(SHARED_HP)
        hp.update(beta=trial.suggest_float("beta", 1.0, 100.0, log=True),
                  emission_weight=trial.suggest_float("emission_weight", 0.2, 4.0),
                  unreachable_logp=trial.suggest_float("unreachable_logp", -60.0, -5.0))
        s = []
        for tid in TUNE_TRIPS:
            trip, ta, graph = TRAIN[tid]
            c, t, p = lists(full_tr, probs_tr[name], tid, len(ta))
            s.append(H.decode_and_score(trip, ta, c, t, p, graph, hp)["hmm_top1"])
        return float(np.mean(s))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6))
    st.enqueue_trial({k: SHARED_HP[k] for k in ["beta", "emission_weight", "unreachable_logp"]})
    st.optimize(objective, n_trials=15)
    own_hp[name] = dict(SHARED_HP); own_hp[name].update(st.best_params)
    st.trials_dataframe().to_csv(f"{OUT}/decoder_trials_{ {'LightGBM (tuned)': 'lightgbm', 'Random Forest': 'rf'}[name]}.csv", index=False)
    log(name, "decoder tuned:", {k: round(v, 3) for k, v in st.best_params.items()}, "train top1", round(st.best_value, 4))

# ----------------------------------------------------------- evaluate
rows = []
for name in CANDIDATES:
    for split in ["train_oof", "student_test"]:
        for dec, hp in [("own", own_hp[name]), ("shared", SHARED_HP)]:
            for r in run(name, split, hp):
                r["decoder"] = dec; rows.append(r)
        log(name, split, "decoded")
per = pd.DataFrame(rows)
per.to_csv(f"{OUT}/per_trip_all_models.csv", index=False)

metrics = ["cls_top1", "hmm_top1", "cls_route_f1", "hmm_route_f1", "cls_continuity", "hmm_continuity"]
summary = per.groupby(["decoder", "split", "model"])[metrics].mean()
dev = per.assign(split="all_88_dev_trips").groupby(["decoder", "split", "model"])[metrics].mean()
summary = pd.concat([summary, dev]).round(4)
summary.to_csv(f"{OUT}/summary_lgbm_vs_rf.csv")


# ------------------------------------------------- paired statistics
def paired(metric, decoder="own"):
    a = per[(per.decoder == decoder) & (per.model == "LightGBM (tuned)")].set_index(["split", "trip"])[metric]
    b = per[(per.decoder == decoder) & (per.model == "Random Forest")].set_index(["split", "trip"])[metric]
    d = (b - a.reindex(b.index)).to_numpy()
    rng = np.random.default_rng(SEED)
    boots = rng.choice(d, (10_000, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"metric": metric, "decoder": decoder, "n_trips": len(d), "mean_RF_minus_LGBM": d.mean(),
            "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": p, "RF_better_trips": int((d > 0).sum()),
            "LGBM_better_trips": int((d < 0).sum()), "ties": int((d == 0).sum())}, d


stats = []
diffs = {}
for m in ["hmm_top1", "hmm_route_f1", "cls_top1", "cls_route_f1"]:
    for dec in ["own", "shared"]:
        s, d = paired(m, dec); stats.append(s); diffs[(m, dec)] = d
stats = pd.DataFrame(stats)
stats.to_csv(f"{OUT}/paired_stats_lgbm_vs_rf.csv", index=False)

# --------------------------------------------------------- decision
prim = stats[(stats.metric == "hmm_top1") & (stats.decoder == "own")].iloc[0]
sec = stats[(stats.metric == "hmm_route_f1") & (stats.decoder == "own")].iloc[0]
tie = lambda s: s.ci95_low <= 0 <= s.ci95_high
if not tie(prim):
    winner = "Random Forest" if prim.mean_RF_minus_LGBM > 0 else "LightGBM (tuned)"
    reason = f"primary metric (HMM Top-1) differs significantly: RF−LGBM = {prim.mean_RF_minus_LGBM:+.4f} [{prim.ci95_low:+.4f}, {prim.ci95_high:+.4f}]"
elif not tie(sec):
    winner = "Random Forest" if sec.mean_RF_minus_LGBM > 0 else "LightGBM (tuned)"
    reason = f"HMM Top-1 tied; secondary (HMM route F1) differs: RF−LGBM = {sec.mean_RF_minus_LGBM:+.4f} [{sec.ci95_low:+.4f}, {sec.ci95_high:+.4f}]"
else:
    lg, rf = ll["LightGBM (tuned)"], ll["Random Forest"]
    pts = sum([lg["oof_log_loss"] < rf["oof_log_loss"], lg["test_infer_ms_per_point"] < rf["test_infer_ms_per_point"],
               lg["model_file_mb"] < rf["model_file_mb"]])
    winner = "LightGBM (tuned)" if pts >= 2 else "Random Forest"
    reason = (f"HMM Top-1 and route F1 statistically tied (CIs contain 0); decided on practical criteria "
              f"(log loss, inference speed, model size): LightGBM wins {pts}/3")
decision = {"winner": winner, "reason": reason, "decision_rule": __doc__.split("PRE-DECLARED DECISION RULE")[1].split("Outputs")[0].strip(),
            "own_decoder_params": own_hp, "probability_quality_and_cost": ll, "runtime_s": time.time() - t_start,
            "evaluator_test_used": False}
json.dump(decision, open(f"{OUT}/decision.json", "w"), indent=1, default=float)

# ------------------------------------------------------------- plots
c = {"LightGBM (tuned)": "#1f5fd1", "Random Forest": "#2ca02c"}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
for a, split, title in zip(axs, ["train_oof", "student_test", "all_88_dev_trips"],
                           ["78 training trips (out-of-fold)", "10 student-test trips", "All 88 development trips"]):
    s = summary.loc["own", split]
    x = np.arange(2); w = .38
    for i, m in enumerate(s.index):
        a.bar(i - w / 2, s.loc[m, "cls_top1"], w, color=c[m], alpha=.45)
        a.bar(i + w / 2, s.loc[m, "hmm_top1"], w, color=c[m])
        a.text(i - w / 2, s.loc[m, "cls_top1"] + .003, f"{s.loc[m, 'cls_top1']:.3f}", ha="center", fontsize=8)
        a.text(i + w / 2, s.loc[m, "hmm_top1"] + .003, f"{s.loc[m, 'hmm_top1']:.3f}", ha="center", fontsize=8)
    a.set_xticks(x, s.index); a.set(title=title, ylim=(0.8, 1.0), ylabel="mean per-trip Top-1")
axs[0].bar([], [], color="grey", alpha=.45, label="classifier only"); axs[0].bar([], [], color="grey", label="+ HMM")
axs[0].legend(loc="lower left"); fig.suptitle("Point Top-1 before vs after HMM (each model with its own tuned decoder)")
fig.tight_layout(); fig.savefig(f"{OUT}/lgbm_vs_rf_before_after_hmm.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for a, m, t in zip(axs, ["hmm_top1", "hmm_route_f1"], ["HMM Top-1", "HMM route F1"]):
    o = per[per.decoder == "own"]
    A = o[o.model == "LightGBM (tuned)"].set_index(["split", "trip"])[m]
    Bv = o[o.model == "Random Forest"].set_index(["split", "trip"])[m].reindex(A.index)
    for sp, mk in [("train_oof", "o"), ("student_test", "^")]:
        sel = A.index.get_level_values(0) == sp
        a.scatter(A[sel], Bv[sel], marker=mk, alpha=.7, label=sp)
    lo = min(A.min(), Bv.min()) - .01
    a.plot([lo, 1], [lo, 1], "k--", lw=1); a.set(xlabel=f"LightGBM {t}", ylabel=f"Random Forest {t}",
                                             title=f"Per-trip {t} (points above line = RF better)")
    a.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/lgbm_vs_rf_per_trip_scatter.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
sel = stats[stats.decoder == "own"].reset_index(drop=True)
ax.errorbar(sel.mean_RF_minus_LGBM, range(len(sel)), xerr=[sel.mean_RF_minus_LGBM - sel.ci95_low, sel.ci95_high - sel.mean_RF_minus_LGBM],
            fmt="o", capsize=4)
ax.axvline(0, color="k", lw=1); ax.set_yticks(range(len(sel)), sel.metric)
ax.set(xlabel="mean paired difference, Random Forest − LightGBM (95% bootstrap CI, 88 trips)",
       title="Is either model significantly better?  (CI crossing 0 = no significant difference)")
fig.tight_layout(); fig.savefig(f"{OUT}/lgbm_vs_rf_difference_ci.png", dpi=130); plt.close(fig)

log("DONE")
print(summary.to_string())
print(stats.round(4).to_string(index=False))
print(json.dumps({k: decision[k] for k in ["winner", "reason"]}, indent=1))
