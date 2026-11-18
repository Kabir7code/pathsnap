"""Comparison plots from the recorded experiment results (results/*.json|csv).

Every plot is drawn from numbers written by scripts 02-04; nothing is typed in.
Outputs in results/02_model_training/plots/ (tuned_vs_untuned.png in results/04_bayesian_optimisation/).
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/02_model_training/plots"
os.makedirs(OUT, exist_ok=True)
runs = json.load(open("results/02_model_training/model_runs.json"))
names = sorted(runs, key=lambda n: -np.mean([f["macro_f1"] for f in runs[n]["cv_folds"]]))

# 1. CV macro F1 per model (mean +- std over folds, individual folds as dots)
fig, ax = plt.subplots(figsize=(9, 4))
for i, n in enumerate(names):
    v = [f["macro_f1"] for f in runs[n]["cv_folds"]]
    ax.bar(i, np.mean(v), yerr=np.std(v), color="C0", alpha=.75, capsize=3)
    ax.scatter([i] * len(v), v, color="k", s=8, zorder=3)
ax.set_xticks(range(len(names)), names, rotation=30)
ax.set(ylabel="macro F1", title="5-fold GroupKFold CV macro F1 (train trips)", ylim=(0.75, 1.0))
fig.tight_layout(); fig.savefig(f"{OUT}/cv_macro_f1.png", dpi=130); plt.close(fig)

# 2. student-test top-1 and macro F1
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(names)); w = .38
ax.bar(x - w / 2, [runs[n]["student_test"]["top1"] for n in names], w, label="Top-1 accuracy")
ax.bar(x + w / 2, [runs[n]["student_test"]["macro_f1"] for n in names], w, label="macro F1")
ax.set_xticks(x, names, rotation=30); ax.set(ylim=(0.75, 1.0), title="Student-test performance (classifier only)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/student_test_top1_f1.png", dpi=130); plt.close(fig)

# 3. log loss (CV and test)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w / 2, [np.mean([f["log_loss"] for f in runs[n]["cv_folds"]]) for n in names], w, label="CV")
ax.bar(x + w / 2, [runs[n]["student_test"]["log_loss"] for n in names], w, label="student test")
ax.set_xticks(x, names, rotation=30); ax.set(ylabel="log loss (lower is better)", title="Log loss")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/log_loss.png", dpi=130); plt.close(fig)

# 4. inference time vs top-1
fig, ax = plt.subplots(figsize=(6, 4))
for n in names:
    t = runs[n]["student_test"]
    ax.scatter(t["infer_ms_per_point"], t["top1"]); ax.annotate(n, (t["infer_ms_per_point"], t["top1"]), fontsize=8)
ax.set(xscale="log", xlabel="inference ms per GPS point (log)", ylabel="student-test Top-1", title="Accuracy vs speed")
fig.tight_layout(); fig.savefig(f"{OUT}/speed_vs_accuracy.png", dpi=130); plt.close(fig)

# 5. confusion matrices (student test, threshold 0.5)
cols = 3; rows = int(np.ceil(len(names) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3.3 * cols, 3 * rows))
for a, n in zip(axs.ravel(), names):
    cm = np.array(runs[n]["student_test"]["confusion"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=8,
                   color="white" if cmn[i, j] > .5 else "black")
    a.set_xticks([0, 1], ["not ref", "ref arc"]); a.set_yticks([0, 1], ["not ref", "ref arc"])
    a.set(title=n, xlabel="predicted", ylabel="actual")
for a in axs.ravel()[len(names):]:
    a.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrices.png", dpi=130); plt.close(fig)

# 6. epoch-wise curves: only the MLP records a training history
if os.path.exists("models/mlp.joblib"):
    mlp = joblib.load("models/mlp.joblib")[-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(mlp.loss_curve_); ax[0].set(xlabel="epoch", ylabel="training log loss", title="MLP training loss")
    ax[1].plot(mlp.validation_scores_); ax[1].set(xlabel="epoch", ylabel="validation accuracy",
                                                   title="MLP validation (early-stopping split)")
    fig.tight_layout(); fig.savefig(f"{OUT}/mlp_training_curves.png", dpi=130); plt.close(fig)

# 7. tuned vs untuned
if os.path.exists("results/04_bayesian_optimisation/tuned_vs_untuned.csv"):
    t = pd.read_csv("results/04_bayesian_optimisation/tuned_vs_untuned.csv")
    ms = ["cv_macro_f1", "test_top1", "test_macro_f1", "test_log_loss"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xx = np.arange(len(ms))
    for k, (_, r) in enumerate(t.iterrows()):
        ax.bar(xx + (k - .5) * .38, [r[m] for m in ms], .38, label=r.variant)
    ax.set_xticks(xx, ms); ax.set(title="Untuned vs Bayesian-tuned final model"); ax.legend()
    fig.tight_layout(); fig.savefig("results/04_bayesian_optimisation/tuned_vs_untuned.png", dpi=130); plt.close(fig)
print("plots written to", OUT)
