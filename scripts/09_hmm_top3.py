"""Classifier-only vs classifier + HMM/Viterbi for the top-3 classifiers (by student-test Top-1).

All three use the SAME decoder parameters (models/hmm_params.json, tuned on
training trips for LightGBM) and the same student-test trips, so the only thing
that changes is the classifier producing the emission probabilities.
Output: results/05_hmm_viterbi/hmm_top3.csv, results/05_hmm_viterbi/hmm_top3_per_trip.csv, results/05_hmm_viterbi/hmm_top3.png
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import CFG
from pathsnap.data import load_trip
from pathsnap.dataset import ART, MODELS
from pathsnap.features import FEATURES
from pathsnap.hmm import RoadGraph
from pathsnap.labels import align_track_to_route

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
hmm_eval = import_module("04_hmm_eval")

TOP3 = {"lightgbm (tuned, final)": "final", "mlp": "mlp", "histgb": "histgb"}
hp = json.load(open(os.path.join(MODELS, "hmm_params.json")))
te = pd.read_parquet(os.path.join(ART, "features_student_test.parquet"))
trips = {}
for tid, g in te.groupby("trip"):
    trip = load_trip(os.path.join(CFG["data_dir"], "student_test"), tid)
    ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
    trips[tid] = (trip, ta, g, RoadGraph(trip, hp["dijkstra_cutoff"]))

rows = []
for label, fname in TOP3.items():
    model = joblib.load(os.path.join(MODELS, f"{fname}.joblib"))
    for tid, (trip, ta, g, graph) in trips.items():
        p = model.predict_proba(g[FEATURES].to_numpy())[:, 1]
        c, t, pr = hmm_eval.lists_from_frame(g, len(ta), p)
        r = hmm_eval.decode_and_score(trip, ta, c, t, pr, graph, hp)
        r.update(model=label, trip=tid)
        rows.append(r)
    print("done", label, flush=True)
per = pd.DataFrame(rows)
per.to_csv("results/05_hmm_viterbi/hmm_top3_per_trip.csv", index=False)
cols = ["cls_top1", "hmm_top1", "cls_route_f1", "hmm_route_f1", "cls_route_precision", "hmm_route_precision",
        "cls_continuity", "hmm_continuity", "decode_s"]
summ = per.groupby("model", sort=False)[cols].mean()
summ.to_csv("results/05_hmm_viterbi/hmm_top3.csv")
print(summ.round(4).to_string())

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
x = np.arange(len(summ)); w = .38
for a, (m, title) in zip(ax, [("top1", "Point Top-1 accuracy"), ("route_f1", "Route F1"), ("continuity", "Route continuity")]):
    a.bar(x - w / 2, summ[f"cls_{m}"], w, label="classifier only", color="#f28e1c")
    a.bar(x + w / 2, summ[f"hmm_{m}"], w, label="classifier + HMM", color="#d62728")
    for i, (u, v) in enumerate(zip(summ[f"cls_{m}"], summ[f"hmm_{m}"])):
        a.text(i - w / 2, u + .005, f"{u:.3f}", ha="center", fontsize=8); a.text(i + w / 2, v + .005, f"{v:.3f}", ha="center", fontsize=8)
    a.set_xticks(x, [s.split(" ")[0] for s in summ.index]); a.set(title=title, ylim=(0.7, 1.02))
ax[0].legend(loc="lower left")
fig.suptitle("Top-3 classifiers: before vs after HMM/Viterbi (mean of 10 student-test trips)")
fig.tight_layout(); fig.savefig("results/05_hmm_viterbi/hmm_top3.png", dpi=130)
