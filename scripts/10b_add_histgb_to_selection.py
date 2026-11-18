"""Add HistGradientBoosting to the final model selection (same protocol as scripts/10_final_model_selection.py).

Development data only (78 OOF training trips + 10 student-test trips); evaluator-test is never read.
HistGB gets its own HMM decoder tuned the same way (15 Optuna trials on training trips <= 3000 points).
Appends to results/08_final_model_selection/per_trip_all_models.csv and writes paired stats vs LightGBM and Random Forest.
"""
import json, os, sys, time
from importlib import import_module
import joblib, numpy as np, optuna, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
from pathsnap.config import CFG
from pathsnap.data import load_trip
from pathsnap.dataset import ART, MODELS, subsample_points
from pathsnap.features import FEATURES
from pathsnap.hmm import RoadGraph
from pathsnap.labels import align_track_to_route
from pathsnap.models import make_model
H = import_module("04_hmm_eval"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED, OUT, NAME = CFG["seed"], "results/08_final_model_selection", "HistGB"
SHARED = json.load(open(os.path.join(MODELS, "hmm_params.json"))); T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)

lab = pd.read_csv("results/01_eda_preprocessing/labelling_train.csv", dtype={"trip": str})
good = lab[lab.labelled_frac >= CFG["min_trip_label_frac"]]
tr = pd.read_parquet(os.path.join(ART, "features_train.parquet")); tr = tr[tr.trip.isin(good.trip)].reset_index(drop=True)
te = pd.read_parquet(os.path.join(ART, "features_student_test.parquet"))
B = {}
for split, tids in [("train", good.trip), ("student_test", sorted(te.trip.unique()))]:
    for tid in tids:
        trip = load_trip(os.path.join(CFG["data_dir"], split), tid); ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
        B[(split, tid)] = (trip, ta, RoadGraph(trip, SHARED["dijkstra_cutoff"]))
oof = np.full(len(tr), np.nan)
for f, (a, b) in enumerate(GroupKFold(5).split(tr, groups=tr.trip)):
    fit = subsample_points(tr.iloc[a][tr.iloc[a].true_arc >= 0], 300_000, SEED + f)
    oof[b] = make_model("histgb", SEED).fit(fit[FEATURES].to_numpy(), fit.label.to_numpy()).predict_proba(tr.iloc[b][FEATURES].to_numpy())[:, 1]
log("OOF done")
model = joblib.load(os.path.join(MODELS, "histgb.joblib"))
t0 = time.time(); pte = model.predict_proba(te[FEATURES].to_numpy())[:, 1]; inf = time.time() - t0
lt = tr.true_arc.to_numpy() >= 0
cost = {"oof_log_loss": log_loss(tr.label[lt], np.clip(oof[lt], 1e-7, 1 - 1e-7)), "test_log_loss": log_loss(te.label, np.clip(pte, 1e-7, 1 - 1e-7)),
        "test_infer_ms_per_point": 1000 * inf / te[["trip", "point"]].drop_duplicates().shape[0],
        "model_file_mb": os.path.getsize(os.path.join(MODELS, "histgb.joblib")) / 1e6}

def dec(df, P, split, tids, hp):
    rows = []
    for tid in tids:
        trip, ta, g = B[(split, tid)]; sub = df[df.trip == tid]
        c, t, p = H.lists_from_frame(sub, len(ta), P[sub.index.to_numpy()])
        r = H.decode_and_score(trip, ta, c, t, p, g, hp); r.update(trip=tid, split=split, points=int((ta >= 0).sum())); rows.append(r)
    return rows

tune = [t for t in good.trip if len(B[("train", t)][1]) <= 3000]
def obj(trial):
    hp = dict(SHARED); hp.update(beta=trial.suggest_float("beta", 1.0, 100.0, log=True),
                                 emission_weight=trial.suggest_float("emission_weight", 0.2, 4.0),
                                 unreachable_logp=trial.suggest_float("unreachable_logp", -60.0, -5.0))
    return float(np.mean([r["hmm_top1"] for r in dec(tr, oof, "train", tune, hp)]))
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6))
st.enqueue_trial({k: SHARED[k] for k in ["beta", "emission_weight", "unreachable_logp"]}); st.optimize(obj, n_trials=15)
hp = dict(SHARED); hp.update(st.best_params); st.trials_dataframe().to_csv(f"{OUT}/decoder_trials_histgb.csv", index=False)
log("decoder tuned", {k: round(v, 3) for k, v in st.best_params.items()}, round(st.best_value, 4))

rows = []
for d, h in [("own", hp), ("shared", SHARED)]:
    for r in dec(tr, oof, "train", list(good.trip), h) + dec(te, pte, "student_test", sorted(te.trip.unique()), h):
        r.update(model=NAME, decoder=d, split=r["split"] if r["split"] == "student_test" else "train_oof"); rows.append(r)
per = pd.read_csv(f"{OUT}/per_trip_all_models.csv", dtype={"trip": str}); per = per[per.model != NAME]
per = pd.concat([per, pd.DataFrame(rows)]); per.to_csv(f"{OUT}/per_trip_all_models.csv", index=False)

ms = ["cls_top1", "hmm_top1", "cls_route_f1", "hmm_route_f1", "cls_continuity", "hmm_continuity"]
o = per[per.decoder == "own"]
summ = pd.concat([o.groupby(["split", "model"])[ms].mean(), o.assign(split="all_88").groupby(["split", "model"])[ms].mean()]).round(4)
summ.to_csv(f"{OUT}/summary_three_models.csv")
stats = []
for other in ["LightGBM (tuned)", "Random Forest"]:
    for m in ["hmm_top1", "hmm_route_f1", "cls_top1", "cls_route_f1"]:
        a = o[o.model == other].set_index(["split", "trip"])[m]; b = o[o.model == NAME].set_index(["split", "trip"])[m].reindex(a.index)
        d = (b - a).to_numpy(); bo = np.random.default_rng(SEED).choice(d, (10_000, len(d))).mean(axis=1); lo, hi = np.percentile(bo, [2.5, 97.5])
        stats.append({"comparison": f"HistGB - {other}", "metric": m, "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi,
                      "wilcoxon_p": wilcoxon(d).pvalue if np.any(d) else 1.0, "HistGB_better": int((d > 0).sum()), "other_better": int((d < 0).sum())})
stats = pd.DataFrame(stats); stats.to_csv(f"{OUT}/paired_stats_histgb_vs_others.csv", index=False)
dj = json.load(open(f"{OUT}/decision.json")); dj["probability_quality_and_cost"][NAME] = cost; dj["own_decoder_params"][NAME] = hp
json.dump(dj, open(f"{OUT}/decision.json", "w"), indent=1, default=float)

c = {"LightGBM (tuned)": "#1f5fd1", "Random Forest": "#2ca02c", "HistGB": "#9467bd"}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
for ax, sp, t in zip(axs, ["train_oof", "student_test", "all_88"], ["78 training trips (OOF)", "10 student-test trips", "All 88 development trips"]):
    s = summ.loc[sp]
    for i, m in enumerate(["LightGBM (tuned)", "Random Forest", "HistGB"]):
        for j, (k, al) in enumerate([("cls_top1", .45), ("hmm_top1", 1)]):
            v = s.loc[m, k]; x = i + (j - .5) * .38
            ax.bar(x, v, .38, color=c[m], alpha=al); ax.text(x, v + .002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), ["LightGBM", "Random Forest", "HistGB"]); ax.set(title=t, ylim=(0.85, 0.97), ylabel="mean per-trip Top-1")
axs[0].bar([], [], color="grey", alpha=.45, label="classifier only"); axs[0].bar([], [], color="grey", label="+ HMM (own decoder)"); axs[0].legend(loc="upper left")
fig.suptitle("Final candidates: point Top-1 before vs after HMM"); fig.tight_layout(); fig.savefig(f"{OUT}/three_models_before_after_hmm.png", dpi=130)
print(summ.to_string()); print(stats.round(4).to_string(index=False)); print(json.dumps(cost, indent=1))
