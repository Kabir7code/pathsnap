#!/usr/bin/env python3
"""Build PathSnap_Models/: a summary of all 9 models plus one folder per model
with its code, results and saved model.

Usage, from the pathsnap repository root with the venv active:
    python build_models_folder.py [output_dir]      (default: PathSnap_Models)

It only reads files that already exist in results/, artifacts/, notebooks/,
pathsnap/ and scripts/. It trains nothing and changes nothing in the repo.
Every number it writes is copied from those files.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.getcwd()
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "PathSnap_Models")

# key, folder, display name, notebook, one-line description of the algorithm
MODELS = [
    ("logreg", "01_logistic_regression", "Logistic Regression", "02_Logistic_Regression_and_Naive_Bayes.ipynb",
     "A linear model: it weights the standardised features and passes the sum through the logistic function "
     "to get P(candidate is the true road arc)."),
    ("gnb", "02_gaussian_naive_bayes", "Gaussian Naive Bayes", "02_Logistic_Regression_and_Naive_Bayes.ipynb",
     "A probabilistic model that assumes each feature is normally distributed within each class and that "
     "features are independent of each other given the class."),
    ("knn", "03_knn", "K-Nearest Neighbours", "03_KNN.ipynb",
     "Instance-based: it stores training rows and scores a candidate by a distance-weighted vote of the "
     "25 most similar training rows. Nothing is fitted."),
    ("dtree", "04_decision_tree", "Decision Tree", "04_Decision_Tree_Random_Forest_Extra_Trees.ipynb",
     "A single tree of if-then splits on the features; each leaf holds the share of true arcs among the "
     "training rows that reach it."),
    ("rf", "05_random_forest", "Random Forest", "04_Decision_Tree_Random_Forest_Extra_Trees.ipynb",
     "Bagging: 200 trees, each trained on a bootstrap sample with random feature subsets at every split; "
     "the forest averages their votes."),
    ("extratrees", "06_extra_trees", "Extra Trees", "04_Decision_Tree_Random_Forest_Extra_Trees.ipynb",
     "Like a random forest, but split thresholds are drawn at random, which makes trees more varied and "
     "faster to build."),
    ("histgb", "07_hist_gradient_boosting", "Histogram Gradient Boosting", "05_HistGradientBoosting_and_LightGBM.ipynb",
     "Boosting: trees are added one at a time, each correcting the errors of the ones before; features are "
     "binned into histograms for speed."),
    ("lightgbm", "08_lightgbm", "LightGBM", "05_HistGradientBoosting_and_LightGBM.ipynb",
     "Gradient boosting with leaf-wise tree growth (it splits the leaf that reduces the loss most). "
     "The best model here, and the one deployed after tuning."),
    ("mlp", "09_mlp", "MLP Neural Network", "06_MLP_Neural_Network.ipynb",
     "A feed-forward neural network with two hidden layers (64 and 32 ReLU units), trained with Adam and "
     "early stopping on standardised features."),
]
NAME = {k: d for k, _, d, _, _ in MODELS}
FOLDER = {k: f for k, f, _, _, _ in MODELS}

REQUIRED = ["results/concurrent/model_comparison.csv", "results/concurrent/model_runs.json",
            "results/model_cards.md", "pathsnap/models.py", "pathsnap/features.py"]


def need(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        sys.exit(f"missing {path}: run this from the pathsnap repo root")
    return p


def cp(src, dst_dir, name=None):
    """Copy a repo file or folder if it exists; return True if copied."""
    s = os.path.join(ROOT, src)
    if not os.path.exists(s):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    d = os.path.join(dst_dir, name or os.path.basename(src.rstrip("/")))
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(s, d)
    return True


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.4f}"


def pct(x):
    return f"{100 * x:.2f}%"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def model_card_sections():
    text = open(need("results/model_cards.md")).read()
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return {p.split("\n", 1)[0].strip(): "## " + p.strip() for p in parts}


def model_definition_code(key):
    """The exact make_model branch for one model, as a standalone function."""
    src = open(need("pathsnap/models.py")).read().splitlines()
    header = src[:next(i for i, l in enumerate(src) if l.startswith("def make_model"))]
    imports, cont = [], False
    for l in header:
        if l.startswith(("from ", "import ")) or (cont and l.startswith(" ")):
            imports.append(l)
            cont = l.rstrip().endswith((",", "("))
        else:
            cont = False
    start = next(i for i, l in enumerate(src) if l.strip() == f'if name == "{key}":')
    end = next(i for i in range(start + 1, len(src)) if src[i].startswith("    if name ==") or src[i].startswith("    raise"))
    body = [l[4:] if l.startswith("    ") else l for l in src[start + 1:end]]
    return "\n".join([
        f'"""Exact model definition used for {NAME[key]}.',
        "Copied from pathsnap/models.py, make_model(name=\"" + key + "\"). Use make_model() from the",
        'package in real runs; this file is for reading and quick experiments."""',
        *imports, "", "",
        f"def make_{key}(seed=42, n_jobs=-1, **params):",
        *body, "", "",
        'if __name__ == "__main__":',
        f"    print(make_{key}())", "",
    ])


def confusion_png(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    labels = ["not true arc (0)", "true arc (1)"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted (threshold 0.5)"); ax.set_ylabel("actual")
    top = max(max(r) for r in cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > top / 2 else "black", fontsize=11)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    cmp_ = pd.read_csv(need("results/concurrent/model_comparison.csv"))
    runs = json.load(open(need("results/concurrent/model_runs.json")))
    cards = model_card_sections()
    tvu = pd.read_csv(os.path.join(ROOT, "results/tuned_vs_untuned.csv")) if os.path.exists("results/tuned_vs_untuned.csv") else None
    hmm = pd.read_csv("results/hmm_summary.csv").set_index("decoder") if os.path.exists("results/hmm_summary.csv") else None
    top3 = pd.read_csv("results/hmm_top3.csv") if os.path.exists("results/hmm_top3.csv") else None
    top3_trip = pd.read_csv("results/hmm_top3_per_trip.csv", dtype={"trip": str}) if os.path.exists("results/hmm_top3_per_trip.csv") else None
    final_meta = json.load(open("artifacts/final_model.json")) if os.path.exists("artifacts/final_model.json") else None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = ""
    commit = commit or "unknown"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    ranked = cmp_.sort_values("test_top1", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    rank = dict(zip(ranked.model, ranked["rank"]))
    best = ranked.iloc[0]

    # ---------- summary/ ----------
    S = os.path.join(OUT, "summary"); os.makedirs(os.path.join(S, "plots"))
    summary = ranked[["rank", "model", "family", "config", "cv_macro_f1", "cv_macro_f1_std", "cv_top1",
                      "test_top1", "test_macro_f1", "test_precision", "test_recall", "test_log_loss",
                      "test_continuity", "fit_s", "infer_ms_per_point"]].copy()
    summary.insert(2, "name", summary.model.map(NAME))
    summary.to_csv(os.path.join(S, "all_models_summary.csv"), index=False)
    cp("results/concurrent/model_comparison.csv", S)
    cp("results/models_dashboard.html", S)

    # all-9 plots drawn from the comparison table
    order = ranked.model.tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(order)); w = 0.38
    ax.bar([i - w / 2 for i in x], ranked.test_top1, w, label="Top-1 accuracy")
    ax.bar([i + w / 2 for i in x], ranked.test_macro_f1, w, label="macro F1")
    for i, (a, b) in enumerate(zip(ranked.test_top1, ranked.test_macro_f1)):
        ax.text(i - w / 2, a + 0.003, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, b + 0.003, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(list(x), [NAME[m] for m in order], rotation=25, ha="right")
    ax.set_ylim(0.75, 1.0); ax.legend(loc="upper right")
    ax.set_title("All 9 models: student-test performance (classifier only)")
    fig.tight_layout(); fig.savefig(os.path.join(S, "plots", "all9_student_test.png"), dpi=130); plt.close(fig)

    cv = cmp_.sort_values("cv_macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(cv)), cv.cv_macro_f1, yerr=cv.cv_macro_f1_std, capsize=4, color="#4c78a8")
    for i, v in enumerate(cv.cv_macro_f1):
        ax.text(i, v + 0.022, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(cv)), [NAME[m] for m in cv.model], rotation=25, ha="right")
    ax.set_ylim(0.75, 1.0); ax.set_ylabel("macro F1")
    ax.set_title("All 9 models: 5-fold GroupKFold CV macro F1 (mean ± std)")
    fig.tight_layout(); fig.savefig(os.path.join(S, "plots", "all9_cv_macro_f1.png"), dpi=130); plt.close(fig)

    for p in ["cv_macro_f1", "student_test_top1_f1", "log_loss", "speed_vs_accuracy", "confusion_matrices",
              "mlp_training_curves", "tuned_vs_untuned", "hmm_top3"]:
        cp(f"results/plots/{p}.png", os.path.join(S, "plots"))
    cp("results/concurrent/timeline.png", os.path.join(S, "plots"), "concurrent_training_timeline.png")

    # ---------- one folder per model ----------
    for key, folder, disp, nb, desc in MODELS:
        if key not in runs:
            print(f"skipping {key}: no results"); continue
        D = os.path.join(OUT, folder)
        C, R, M = (os.path.join(D, s) for s in ("code", "results", "model"))
        for d in (C, R, M):
            os.makedirs(d)
        r = runs[key]; t = r["student_test"]
        folds = pd.DataFrame([{k: v for k, v in f.items() if k != "confusion"} for f in r["cv_folds"]])
        folds.insert(0, "fold", range(len(folds)))
        folds.to_csv(os.path.join(R, "cv_folds.csv"), index=False)
        pd.DataFrame([{k: v for k, v in t.items() if k != "confusion"}]).to_csv(os.path.join(R, "student_test_metrics.csv"), index=False)
        json.dump(r, open(os.path.join(R, "metrics.json"), "w"), indent=1)
        cm = t["confusion"]
        confusion_png(cm, f"{disp}: student-test confusion", os.path.join(R, "confusion_matrix.png"))
        card = cards.get(key, "")
        open(os.path.join(R, "model_card.md"), "w").write(card + "\n")

        open(os.path.join(C, "model_definition.py"), "w").write(model_definition_code(key))
        sh = os.path.join(C, "run_this_model.sh")
        open(sh, "w").write(
            "#!/usr/bin/env bash\n"
            f"# Retrain and re-evaluate only {disp} with the project protocol\n"
            "# (5-fold GroupKFold on the training trips, then the 10 student-test trips).\n"
            "# Run from the pathsnap repo root with the venv active and the dataset in data/.\n"
            f"# Updates artifacts/models/{key}.joblib, results/model_runs.json and results/model_comparison.csv.\n"
            f"python scripts/02_train_models.py --models {key}\n")
        os.chmod(sh, 0o755)
        cp(f"notebooks/{nb}", C)
        cp(f"artifacts/models/{key}.joblib", M)

        hm = ""
        if top3 is not None:
            row = top3[top3.model.str.split().str[0] == key]
            if len(row):
                row.to_csv(os.path.join(R, "hmm_before_after.csv"), index=False)
                if top3_trip is not None:
                    top3_trip[top3_trip.model.str.split().str[0] == key].to_csv(os.path.join(R, "hmm_per_trip.csv"), index=False)
                h = row.iloc[0]
                hm = ("\n## With HMM/Viterbi decoding\n\nThis model is in the top 3, so it was also run through the HMM "
                      "decoder with the same tuned settings (mean of the 10 student-test trips, `results/hmm_top3.csv`):\n\n"
                      "| Metric | Classifier only | + HMM/Viterbi |\n| --- | --- | --- |\n"
                      f"| Top-1 accuracy | {f4(h.cls_top1)} | **{f4(h.hmm_top1)}** |\n"
                      f"| Route F1 | {f4(h.cls_route_f1)} | **{f4(h.hmm_route_f1)}** |\n"
                      f"| Route continuity | {f4(h.cls_continuity)} | **{f4(h.hmm_continuity)}** |\n\n"
                      "These are per-trip averages, so they differ from the pooled Top-1 in the table above.\n")
                if "tuned" in str(h.model):
                    hm += ("\nThe HMM row uses the **tuned** model (`final.joblib`, see `../10_lightgbm_tuning/`), "
                           "the one the API deploys. The table above is the untuned run.\n")

        cvm, cvs = folds.mean(numeric_only=True), folds.std(numeric_only=True)
        res = ("| Metric | 5-fold CV (mean ± std) | Student test (10 trips) |\n| --- | --- | --- |\n"
               f"| Top-1 accuracy | {f4(cvm.top1)} ± {f4(cvs.top1)} | **{f4(t['top1'])}** |\n"
               f"| Macro F1 | {f4(cvm.macro_f1)} ± {f4(cvs.macro_f1)} | **{f4(t['macro_f1'])}** |\n"
               f"| Precision | {f4(cvm.precision)} | {f4(t['precision'])} |\n"
               f"| Recall | {f4(cvm.recall)} | {f4(t['recall'])} |\n"
               f"| Log loss | {f4(cvm.log_loss)} | {f4(t['log_loss'])} |\n"
               f"| Route continuity | not measured in CV | {f4(t.get('continuity'))} |\n"
               f"| Fit time (s) | {cvm.fit_s:.2f} per fold | {t['fit_s']:.2f} |\n"
               f"| Inference (ms per GPS point) | {cvm.infer_ms_per_point:.4f} | {t['infer_ms_per_point']:.4f} |\n")
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        readme = f"""# {disp}

{desc}

**Rank {rank[key]} of 9** by student-test Top-1 accuracy. Family: {r['family']}. Configuration: {r['config']}.
Trained on {r['train_rows']:,} candidate rows, seed {r['seed']}.

## Results

{res}
Top-1 accuracy = share of GPS points whose highest-scored candidate is the true road arc (all test points pooled).

### Confusion matrix (student test, threshold 0.5)

![confusion matrix](results/confusion_matrix.png)

| | predicted: not true arc | predicted: true arc |
| --- | --- | --- |
| **actual: not true arc** | {tn:,} | {fp:,} |
| **actual: true arc** | {fn:,} | {tp:,} |
{hm}
## What the model learned

{card.split(chr(10), 1)[1].strip() if card else 'No model card found.'}

## Files

| Path | Contents |
| --- | --- |
| `code/model_definition.py` | the exact model definition (from `pathsnap/models.py`) |
| `code/run_this_model.sh` | retrain and re-evaluate only this model |
| `code/{nb}` | the notebook for this model family |
| `results/metrics.json` | every recorded metric, per fold and on the test trips |
| `results/cv_folds.csv`, `results/student_test_metrics.csv` | the same as tables |
| `results/confusion_matrix.png` | the confusion matrix above |
| `results/model_card.md` | hyperparameters and learned structure |
| `model/{key}.joblib` | the trained model (load with `joblib.load`) |

The shared pipeline every model uses (candidates, features, training protocol) is in `../shared_code/`.

## How to run

```bash
cd ~/pathsnap && source .venv/bin/activate
python -c "import joblib; m = joblib.load('artifacts/models/{key}.joblib'); print(m)"
python scripts/02_train_models.py --models {key}
```
The first command loads the saved model; the second retrains and re-evaluates it (needs the dataset in `data/`).
"""
        open(os.path.join(D, "README.md"), "w").write(readme)

    # ---------- shared code ----------
    SC = os.path.join(OUT, "shared_code")
    cp("pathsnap", SC)
    for s in ["01_build_dataset.py", "02_train_models.py", "02b_train_concurrent.py", "06_plots.py", "07_model_cards.py"]:
        cp(f"scripts/{s}", os.path.join(SC, "scripts"))
    cp("requirements.txt", SC)
    open(os.path.join(SC, "README.md"), "w").write(
        "# Shared code\n\nEvery model goes through the same pipeline:\n\n"
        "1. `pathsnap/candidates.py`: the nearby road arcs for each GPS point.\n"
        "2. `pathsnap/features.py`: 14 features per (GPS point, candidate arc).\n"
        "3. `pathsnap/labels.py`: which candidate is the true arc, from the `.route` file.\n"
        "4. `pathsnap/models.py`: the 9 model definitions (`make_model`).\n"
        "5. `scripts/02_train_models.py`: 5-fold GroupKFold by trip on the training trips, then the 10 student-test trips.\n"
        "6. `scripts/02b_train_concurrent.py`: the same protocol with all 9 models in parallel.\n\n"
        "These are copies for reading. Run everything from the real repo (`~/pathsnap`).\n")

    # ---------- data, tuning, HMM ----------
    E = os.path.join(OUT, "00_data_and_features")
    cp("results/eda", E, "eda_plots")
    cp("notebooks/01_EDA_and_preprocessing.ipynb", E)
    cp("results/eda_log.txt", E)

    T = os.path.join(OUT, "10_lightgbm_tuning")
    for p in ["scripts/03_bayes_opt.py", "notebooks/07_Bayesian_Hyperparameter_Optimisation.ipynb",
              "results/tuned_vs_untuned.csv", "results/optuna_trials_lightgbm.csv",
              "results/optuna_history_lightgbm.png", "results/plots/tuned_vs_untuned.png",
              "artifacts/final_model.json", "artifacts/models/final.joblib"]:
        cp(p, T)
    tune_txt = ""
    if tvu is not None:
        u, tu = tvu.set_index("variant").loc["untuned"], tvu.set_index("variant").loc["tuned"]
        trials = final_meta.get("trials", "?") if final_meta else "?"
        tune_txt = (f"Bayesian optimisation (Optuna TPE, {trials} trials) of LightGBM. Test Top-1 went from "
                    f"{f4(u.test_top1)} (untuned) to {f4(tu.test_top1)} (tuned); CV macro F1 from "
                    f"{f4(u.cv_macro_f1)} to {f4(tu.cv_macro_f1)}. No real gain, so the classifier is limited by "
                    "its features, not its settings. The tuned model is deployed as `final.joblib`.")
    open(os.path.join(T, "README.md"), "w").write(f"# LightGBM hyperparameter tuning\n\n{tune_txt}\n")

    H = os.path.join(OUT, "11_hmm_viterbi")
    for p in ["pathsnap/hmm.py", "pathsnap/pipeline.py", "scripts/04_hmm_eval.py", "scripts/09_hmm_top3.py",
              "notebooks/08_HMM_Viterbi_and_Final_Comparison.ipynb", "artifacts/hmm_params.json",
              "results/hmm_summary.csv", "results/hmm_student_test.csv", "results/hmm_top3.csv",
              "results/hmm_top3_per_trip.csv", "results/hmm_comparison.png", "results/plots/hmm_top3.png"]:
        cp(p, H)
    hmm_txt = ""
    if hmm is not None:
        tr = hmm.loc["tuned"]
        hmm_txt = (f"The classifier's probabilities become HMM emissions and Viterbi picks a connected route. "
                   f"With the deployed LightGBM and the tuned decoder (mean of 10 student-test trips): Top-1 "
                   f"{f4(tr.cls_top1)} → **{f4(tr.hmm_top1)}**, route F1 {f4(tr.cls_route_f1)} → "
                   f"**{f4(tr.hmm_route_f1)}**, continuity {f4(tr.cls_continuity)} → {f4(tr.hmm_continuity)}.")
    open(os.path.join(H, "README.md"), "w").write(f"# HMM + Viterbi decoding\n\n{hmm_txt}\n")

    # ---------- top-level README ----------
    tbl = pd.DataFrame({
        "Rank": ranked["rank"],
        "Model": [f"[{NAME[m]}]({FOLDER[m]}/README.md)" for m in ranked.model],
        "Family": ranked.family,
        "CV macro F1": [f"{a:.4f} ± {b:.4f}" for a, b in zip(ranked.cv_macro_f1, ranked.cv_macro_f1_std)],
        "Test Top-1": [f"**{v:.4f}**" if i == 0 else f"{v:.4f}" for i, v in enumerate(ranked.test_top1)],
        "Test macro F1": ranked.test_macro_f1.map("{:.4f}".format),
        "Test log loss": ranked.test_log_loss.map("{:.4f}".format),
        "Fit time (s)": ranked.fit_s.map("{:.1f}".format),
        "ms / point": ranked.infer_ms_per_point.map("{:.4f}".format),
    })
    hmm_line = ""
    if hmm is not None:
        tr = hmm.loc["tuned"]
        hmm_line = (f"With HMM/Viterbi decoding on top of the deployed (tuned) LightGBM, the per-trip mean Top-1 rises "
                    f"from {pct(tr.cls_top1)} to **{pct(tr.hmm_top1)}** and route F1 from {pct(tr.cls_route_f1)} "
                    f"to **{pct(tr.hmm_route_f1)}**.")
    if tvu is not None:
        tu = tvu.set_index("variant").loc["tuned"]
        hmm_line += (f"\nThe deployed model is the tuned LightGBM (`final.joblib`, test Top-1 {tu.test_top1:.4f}); "
                     "tuning made no real difference (see `10_lightgbm_tuning/`).")
    feats = re.search(r"FEATURES\s*=\s*\[(.*?)\]", open(need("pathsnap/features.py")).read(), re.S)
    feats = ", ".join(f"`{f.strip().strip(chr(34))}`" for f in feats.group(1).split(",") if f.strip()) if feats else ""
    readme = f"""# PathSnap: all 9 models

**{NAME[best.model]} is the best classifier: student-test Top-1 {best.test_top1:.4f}, macro F1 {best.test_macro_f1:.4f}.**
{hmm_line}

Each model scores every (GPS point, nearby road arc) pair with P(arc is the true road). All 9 use the same 14
features and the same protocol: 5-fold GroupKFold by trip on the training trips, then the 10 student-test trips.

## Ranking

{md_table(tbl)}

Top-1 here pools all test GPS points. The HMM numbers above are per-trip means, so compare them only with each other.

![all 9 models](summary/plots/all9_student_test.png)

## Folders

| Folder | Contents |
| --- | --- |
| `summary/` | the table above as CSV, all comparison plots, the one-page dashboard |
| `00_data_and_features/` | EDA plots, data-quality log, EDA notebook |
| `01_…` to `09_…` | one folder per model: `README.md`, `code/`, `results/`, `model/` |
| `10_lightgbm_tuning/` | Bayesian optimisation of LightGBM and the deployed `final.joblib` |
| `11_hmm_viterbi/` | HMM/Viterbi decoder, its settings and before/after results |
| `shared_code/` | the pipeline every model uses (candidates, features, labels, training) |

## Features (14)

{feats}

## Source

Built by `build_models_folder.py` from the repo at commit `{commit}`. Every number is copied from
`results/concurrent/model_comparison.csv`, `results/concurrent/model_runs.json`, `results/model_cards.md`,
`results/tuned_vs_untuned.csv` and `results/hmm_*.csv`. Nothing was retrained to build this folder.
"""
    open(os.path.join(OUT, "README.md"), "w").write(readme)
    print("built", OUT)


if __name__ == "__main__":
    main()
