"""Generate the Jupyter notebooks in notebooks/ (run once; then execute them in Jupyter)."""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks")
os.makedirs(OUT, exist_ok=True)

COLAB = '# ── Google Colab setup ── (skipped automatically when running locally)\nimport os, sys, shutil, subprocess, zipfile\nif "google.colab" in sys.modules:\n    REPO = "/content/pathsnap"\n    if not os.path.exists(REPO):\n        subprocess.run(["git", "clone", "-q", "https://github.com/Kabir7code/pathsnap.git", REPO], check=True)\n    os.chdir(REPO)\n    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna", "lightgbm", "folium", "pyarrow"], check=True)\n    if not (os.path.isdir("data/train") and os.path.isdir("data/student_test")):\n        # 1) put Student_folder.zip in your Google Drive (MyDrive) -- or 2) upload it when asked\n        DATA_ZIP = "/content/drive/MyDrive/Student_folder.zip"\n        try:\n            from google.colab import drive\n            drive.mount("/content/drive")\n        except Exception as e:\n            print("Drive not mounted:", e)\n        if not os.path.exists(DATA_ZIP):\n            from google.colab import files\n            print("Student_folder.zip not found in Drive. Please upload it:")\n            DATA_ZIP = "/content/" + list(files.upload())[0]\n        raw = "/content/_raw"\n        shutil.rmtree(raw, ignore_errors=True)\n        zipfile.ZipFile(DATA_ZIP).extractall(raw)\n        src = next(d for d, sub, _ in os.walk(raw) if "train" in sub and "student_test" in sub)\n        os.makedirs("data", exist_ok=True)\n        for part in ("train", "student_test"):\n            shutil.move(os.path.join(src, part), os.path.join("data", part))\n    print("Colab ready:", os.getcwd(), "| train trips:", len(os.listdir("data/train")),\n          "| student_test trips:", len(os.listdir("data/student_test")))'

SETUP = """import os, sys, warnings
# run from notebooks/ or from the repo root
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(ROOT); sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 160)
from pathsnap import nb
from pathsnap.models import make_model
from pathsnap.config import CFG
SEED = CFG["seed"]  # 42: every model and data split uses this seed
print("repo root:", ROOT)"""

LOAD = "tr, tr_pts, te, te_pts = nb.load_data()"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    n = nbf.v4.new_notebook()
    cells[0].source = cells[0].source.replace("{FILE}", name)
    n.cells = [cells[0], md("**On Google Colab:** run the next cell first. It clones the repo, installs packages and loads `Student_folder.zip` from your Google Drive (`MyDrive/Student_folder.zip`) or asks you to upload it. Locally it does nothing."), code(COLAB)] + cells[1:]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, os.path.join(OUT, name))
    print("wrote", name)


PROTOCOL = """
**Protocol (same for every model):**
- Each row is one *(GPS point, candidate arc)* pair. The label is 1 if the arc is the point's reference arc.
- **5-fold GroupKFold by trip** on the 78 usable training trips. Each fold fits on a 300k-row sample of points (KNN uses 100k) and is scored on all candidates of the held-out trips.
- The model is then **refit on the training split** and scored on the **10 student-test trips**.
- **Metrics:**
  - macro F1 / precision / recall: candidate level, threshold 0.5
  - log loss
  - **Top-1 accuracy**: the share of GPS points whose highest-scoring candidate is the reference arc
  - continuity
- The **HMM/Viterbi** decoder (parameters in `models/hmm_params.json`) is then run on this model's probabilities to compare *classifier only* with *classifier + HMM*.
"""


def model_section(key, title, explain, max_rows=300_000):
    v = key
    return [
        md(f"## {title}\n\n{explain}"),
        code(f"""make_{v} = lambda: make_model("{key}", SEED)
print(make_{v}())
cv_{v} = nb.run_cv(make_{v}, tr, tr_pts, max_rows={max_rows})
cv_{v}.round(4)"""),
        code(f"""model_{v}, train_{v}, test_{v}, p_{v} = nb.fit_and_evaluate(make_{v}, tr, tr_pts, te, te_pts, max_rows={max_rows})
print(f"student-test continuity (classifier only): {{test_{v}['continuity']:.4f}} | inference {{test_{v}['infer_ms_per_point']:.4f}} ms/point")
nb.summary_table("{title}", cv_{v}, train_{v}, test_{v})"""),
        code(f"""imp_{v} = nb.plot_importance(model_{v}, te, "{title}")"""),
        code(f"""hmm_{v} = nb.hmm_compare(model_{v}, te, "{title}")
nb.plot_hmm(hmm_{v}, "{title}: classifier only vs classifier + HMM (student-test trips)")
nb.save_results("{key}", cv_{v}, train_{v}, test_{v},
                {{"hmm_mean": hmm_{v}.drop(columns=["trip", "model"]).mean().to_dict()}})"""),
    ]


def comparison(keys, titles):
    d = lambda pat: "{" + ", ".join(pat.format(k=k, t=t) for k, t in zip(keys, titles)) + "}"
    return [
        md("## Comparison of the models in this notebook"),
        code(f"""nb.plot_cv_folds({d('"{t}": cv_{k}')})
nb.plot_overfit({d('"{t}": (train_{k}, cv_{k}, test_{k})')})
nb.plot_confusion({d('"{t}": test_{k}["confusion"]')})
nb.plot_calibration(te.label.to_numpy(), {d('"{t}": p_{k}')})"""),
        code(f"""rows = []
for name, cv, t, h in [{", ".join(f'("{t}", cv_{k}, test_{k}, hmm_{k})' for k, t in zip(keys, titles))}]:
    rows.append({{"model": name, "CV macro F1": cv.macro_f1.mean(), "CV std": cv.macro_f1.std(),
                 "test Top-1": t["top1"], "test macro F1": t["macro_f1"], "test log loss": t["log_loss"],
                 "test continuity": t["continuity"], "Top-1 + HMM (trip mean)": h.hmm_top1.mean(),
                 "route F1 + HMM (trip mean)": h.hmm_route_f1.mean()}})
pd.DataFrame(rows).set_index("model").round(4)"""),
    ]


COLAB_URL = "https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/"


def header(num, title, laptop, runtime, what):
    return md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL}{{FILE}})

# {num}. {title}
**Run on:** {laptop} · **Approx. runtime:** {runtime} on a 4-core laptop

{what}

**Before running:** the dataset must be in `data/train/<id>/` and `data/student_test/<id>/`, and `pip install -r requirements.txt` done. Run all cells top to bottom (Kernel → Restart & Run All). Every number and plot below is computed when you run it.""")


# --------------------------------------------------------------- 01 EDA
write("01_EDA_and_preprocessing.ipynb", [
    header("01", "Dataset exploration and preprocessing", "Laptop A", "~4 min",
           "Inventory of the raw files, data-quality checks, GPS and network plots, an interactive Folium map, "
           "then building the candidate/feature tables used by every model notebook."),
    code(SETUP),
    md("## 1. Exploratory data analysis\nRuns `scripts/00_eda.py`, which reads every trip in **train** and **student_test**, checks missing values, timestamps, coordinates and network connectivity, and saves tables and plots to `results/01_eda_preprocessing/`."),
    code("""sys.path.insert(0, os.path.join(ROOT, "scripts"))
from importlib import import_module
import_module("00_eda").main()"""),
    code("""inv = pd.read_csv("results/01_eda_preprocessing/inventory.csv", dtype={"trip": str})
net = pd.read_csv("results/01_eda_preprocessing/network.csv", dtype={"trip": str})
print("Missing-value / validity checks per partition:")
display(inv.groupby("split")[["track_nan", "non_monotonic_ts", "coords_out_of_range", "gaps_over_10s"]].sum())
display(inv.groupby("split")[["points", "duration_s", "path_km", "dt_max", "frac_dt_1s", "route_arcs"]].describe().round(2))"""),
    code("""display(net.groupby("split")[["nodes", "arcs", "isolated_nodes", "weak_components", "strong_components",
                               "largest_strong_frac", "two_way_frac"]].describe().round(3))"""),
    md("## 2. Plots"),
    code("""from IPython.display import Image, display
for f in ["sampling_and_length", "gps_to_road_distance", "trip_locations", "network_size", "example_trip"]:
    print(f); display(Image(f"results/01_eda_preprocessing/{f}.png"))"""),
    md("## 3. Interactive Folium map (student_test trip 00000000)\nGPS observations, the road network near the track, and the reference route. Use the layer control (top right)."),
    code("""from pathsnap.data import load_trip
from pathsnap.viz import trip_map
trip = load_trip("data/student_test", "00000000")
m = trip_map(trip, [("Reference route", trip.route, "#1a9e3a", 7, None)], "student_test 00000000 (EDA)")
m"""),
    md("## 4. Labels and candidates\n`.route` lists the arcs driven but not which GPS point belongs to which arc. `pathsnap/labels.py` aligns them with a monotonic dynamic program. Then `pathsnap/candidates.py` lists arcs within 50 m (max 8) per point, and `pathsnap/features.py` computes 14 features per (point, candidate)."),
    code("""nb.ensure_features()
for s in ["train", "student_test"]:
    l = pd.read_csv(f"results/01_eda_preprocessing/labelling_{s}.csv", dtype={"trip": str})
    print(f"{s}: labelled {(l.labelled_frac * l.points).sum() / l.points.sum():.4f}, "
          f"mean candidate miss rate {l.candidate_miss_rate.mean():.4f}")
    display(l.sort_values("labelled_frac").head(3).round(4))"""),
    code("""tr, tr_pts, te, te_pts = nb.load_data()
from pathsnap.features import FEATURES
print("features:", FEATURES)
display(tr[FEATURES + ["label"]].describe().T.round(3))
near = te[te.dist_rank == 0][["trip", "point", "arc"]]
m = te_pts[te_pts.true_arc >= 0].merge(near, on=["trip", "point"], how="left")
print(f"Nearest-arc baseline Top-1 on student test: {(m.arc == m.true_arc).mean():.4f}")"""),
    code("""fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, f in zip(ax, ["dist", "heading_diff", "win_dist"]):
    for lab, c in [(0, "#888"), (1, "#d62728")]:
        a.hist(tr.loc[tr.label == lab, f].clip(upper=tr[f].quantile(.99)), bins=50, alpha=.6, density=True,
               color=c, label="reference arc" if lab else "other candidate")
    a.set(title=f, ylabel="density")
ax[0].legend(); fig.suptitle("Feature distributions by class (train)"); fig.tight_layout(); nb.show()"""),
])

# ------------------------------------------------------- 02 linear models
write("02_Logistic_Regression_and_Naive_Bayes.ipynb", [
    header("02", "Linear / probabilistic models: Logistic Regression and Gaussian Naive Bayes", "Laptop A", "~3 min",
           PROTOCOL),
    code(SETUP), code(LOAD),
    *model_section("logreg", "Logistic Regression", """
**What it does:** learns one weight per (standardised) feature, sums them, and passes the total through a sigmoid to get P(reference arc).
- **Why it suits PathSnap:** fast, interpretable (the sign and size of each weight), and gives probabilities the HMM can use.
- **Limitation:** one linear decision boundary, so it can't express interactions like *close AND same direction* unless they're engineered.
- **Hyperparameters:** C = 1.0 (inverse L2 regularisation), lbfgs, max_iter 2000.
- **Loss:** regularised binary cross-entropy (log loss)."""),
    *model_section("gnb", "Gaussian Naive Bayes", """
**What it does:** fits a Gaussian per feature for each class and combines them with Bayes' rule, **assuming features are independent given the class**.
- **Why tried:** a very fast probabilistic baseline.
- **Limitation:** our features are strongly correlated (dist, dist_rank, dist_minus_min, win_dist all measure closeness). It double-counts that evidence and becomes over-confident, which shows up as poor log loss.
- **Hyperparameters:** var_smoothing = 1e-9.
- **Objective:** maximum likelihood of the class-conditional Gaussians."""),
    *comparison(["logreg", "gnb"], ["Logistic Regression", "Gaussian Naive Bayes"]),
])

# ------------------------------------------------------------------ 03 KNN
write("03_KNN.ipynb", [
    header("03", "Instance-based model: K-Nearest Neighbours", "Laptop A", "~6-10 min (prediction is slow)", PROTOCOL),
    code(SETUP), code(LOAD),
    *model_section("knn", "KNN", """
**What it does:** no training. For a new (point, candidate) row it finds the 25 most similar training rows (Euclidean distance on standardised features) and takes a distance-weighted vote.
- **Why tried:** a non-parametric baseline; similar geometric situations should have similar answers.
- **Limitations:**
  - prediction is slow, since it compares against every stored row; we fit on 100k rows to keep it tractable
  - sensitive to feature scaling
  - probabilities are coarse, since they come from votes
- **Hyperparameters:** k = 25, weights = distance, metric = Minkowski (p = 2).
- **Objective:** none (lazy learner).""", max_rows=100_000),
    *comparison(["knn"], ["KNN"]),
])

# ----------------------------------------------------------------- 04 trees
write("04_Decision_Tree_Random_Forest_Extra_Trees.ipynb", [
    header("04", "Tree-based and bagging ensembles: Decision Tree, Random Forest, Extra Trees", "Laptop A", "~8-12 min",
           PROTOCOL),
    code(SETUP), code(LOAD),
    *model_section("dtree", "Decision Tree", """
**What it does:** a flowchart of yes/no splits on features (e.g. *dist_minus_min ≤ 0?*, *heading_diff ≤ 40°?*) ending in leaves with a probability.
- **Why it suits PathSnap:** captures feature interactions and thresholds naturally; no scaling needed.
- **Limitation:** one tree is high-variance, so it is regularised here.
- **Hyperparameters:** max_depth = 12, min_samples_leaf = 50.
- **Objective:** Gini impurity reduction at each split."""),
    *model_section("rf", "Random Forest", """
**What it does:** 200 decision trees, each trained on a bootstrap sample with random feature subsets (sqrt of 14) at each split; probabilities are averaged.
- **Why:** averaging de-correlated trees reduces the variance of a single tree.
- **Limitation:** large model files (about 18 MB) and slower than boosting at similar accuracy.
- **Hyperparameters:** n_estimators = 200, max_depth = 16, min_samples_leaf = 20.
- **Objective:** Gini impurity per tree."""),
    *model_section("extratrees", "Extra Trees", """
**What it does:** like Random Forest, but split thresholds are drawn **at random** and the best random split is kept, which adds even more randomisation.
- **Why tried:** often as accurate as RF and faster to train.
- **Limitation:** random thresholds are less precise on the key distance and heading features; here it is the weakest tree ensemble.
- **Hyperparameters:** n_estimators = 200, max_depth = 18, min_samples_leaf = 20.
- **Objective:** Gini impurity."""),
    *comparison(["dtree", "rf", "extratrees"], ["Decision Tree", "Random Forest", "Extra Trees"]),
])

# --------------------------------------------------------------- 05 boosting
write("05_HistGradientBoosting_and_LightGBM.ipynb", [
    header("05", "Boosting: HistGradientBoosting and LightGBM", "Laptop B", "~4-6 min", PROTOCOL),
    code(SETUP), code(LOAD),
    *model_section("histgb", "HistGradientBoosting", """
**What it does:** gradient boosting. Trees are added **one after another**, each fitted to the errors (gradients of the log loss) of the current ensemble. Features are binned into histograms for speed.
- **Why:** boosting is usually the strongest family on tabular features like ours.
- **Limitation:** sequential training; more hyperparameters to tune.
- **Hyperparameters:** max_iter = 300, learning_rate = 0.1, max_leaf_nodes = 31, automatic early stopping.
- **Loss:** binary log loss."""),
    *model_section("lightgbm", "LightGBM", """
**What it does:** gradient boosting with
- **histogram binning**: fast split search
- **leaf-wise growth**: always splits the leaf that reduces the loss most, so it learns fine interactions like *close AND same direction AND neighbours agree*
- row and feature subsampling plus L2 regularisation

Each new tree corrects the previous ones; the final score is the sum of all trees scaled by the learning rate.
- **Why it suits PathSnap:** handles mixed-unit features without scaling, captures interactions, gives well-calibrated probabilities (important for the HMM), and predicts very fast.
- **Limitation:** can overfit with large leaves; it is a black box apart from importances.
- **Hyperparameters (untuned):** n_estimators = 400, learning_rate = 0.05, num_leaves = 63, min_child_samples = 50, subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1.
- **Loss:** binary log loss."""),
    md("### LightGBM training curve\nLightGBM records the loss after every boosting round. Here it is trained on 80% of the training trips and validated on the other 20% (a group split by trip) to show how train and validation log loss evolve."),
    code("""from sklearn.model_selection import GroupShuffleSplit
from pathsnap.features import FEATURES
from pathsnap.dataset import subsample_points
i_tr, i_va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(tr, groups=tr.trip))
fit = subsample_points(tr.iloc[i_tr], 300_000, SEED); va = tr.iloc[i_va]
curve_model = make_model("lightgbm", SEED)
curve_model.fit(fit[FEATURES].to_numpy(), fit.label.to_numpy(),
                eval_set=[(fit[FEATURES].to_numpy(), fit.label.to_numpy()), (va[FEATURES].to_numpy(), va.label.to_numpy())],
                eval_names=["train", "validation"], eval_metric="binary_logloss")
ev = curve_model.evals_result_
plt.figure(figsize=(7, 4))
plt.plot(ev["train"]["binary_logloss"], label="train"); plt.plot(ev["validation"]["binary_logloss"], label="validation (held-out trips)")
plt.xlabel("boosting round (tree)"); plt.ylabel("log loss"); plt.title("LightGBM training curve"); plt.legend(); nb.show()
best = int(np.argmin(ev["validation"]["binary_logloss"]))
print(f"lowest validation log loss {min(ev['validation']['binary_logloss']):.4f} at round {best + 1} of 400")"""),
    *comparison(["histgb", "lightgbm"], ["HistGradientBoosting", "LightGBM"]),
])

# --------------------------------------------------------------------- 06 MLP
write("06_MLP_Neural_Network.ipynb", [
    header("06", "Neural network: Multi-Layer Perceptron", "Laptop B", "~6-8 min", PROTOCOL),
    code(SETUP), code(LOAD),
    *model_section("mlp", "MLP", """
**What it does:** a feed-forward network, 14 inputs → 64 ReLU → 32 ReLU → 1 sigmoid output, trained with Adam. Each layer learns non-linear combinations of the previous layer's outputs.
- **Why tried:** a different inductive bias from trees: smooth non-linear boundaries.
- **Limitations:** needs feature scaling, is sensitive to initialisation and learning rate, and is less interpretable.
- **Hyperparameters:** hidden layers (64, 32), alpha (L2) = 1e-4, Adam, early stopping on a 10% validation split, max 200 epochs.
- **Loss:** binary cross-entropy."""),
    md("### Training history (epoch-wise)\nThe MLP is the only model here that trains in epochs, so it is the only one with a genuine loss curve."),
    code("""mlp = model_mlp[-1]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(mlp.loss_curve_); ax[0].set(title="MLP training loss", xlabel="epoch", ylabel="log loss")
ax[1].plot(mlp.validation_scores_); ax[1].set(title="MLP validation accuracy (early-stopping split)", xlabel="epoch", ylabel="accuracy")
fig.tight_layout(); nb.show()
print(f"stopped after {mlp.n_iter_} epochs; best validation accuracy {max(mlp.validation_scores_):.4f}")"""),
    *comparison(["mlp"], ["MLP"]),
])

# ---------------------------------------------------------- 07 Bayesian opt
write("07_Bayesian_Hyperparameter_Optimisation.ipynb", [
    header("07", "Bayesian hyperparameter optimisation of LightGBM (Optuna TPE)", "Laptop B", "~10-12 min",
           """LightGBM was the best model in the comparison, so it is tuned here.
- **Search:** Optuna's Tree-structured Parzen Estimator (TPE, a Bayesian method), seed 42, 8 random start-up trials, then model-guided.
- **Objective:** maximise the mean **macro F1** over **3-fold GroupKFold by trip** on the training trips (150k-row samples per fold for speed).
- **Afterwards:** the untuned and tuned models are compared with the **same** 5-fold CV and student-test protocol."""),
    code(SETUP), code(LOAD),
    md("## Search space"),
    code("""import optuna, time
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from importlib import import_module
space = import_module("03_bayes_opt").space   # the same search space as scripts/03_bayes_opt.py
import inspect; print(inspect.getsource(space).split('if name == "rf"')[0])"""),
    code("""N_TRIALS = 20     # increase for a longer search
from pathsnap.experiment import cross_validate

def objective(trial):
    params = space(trial, "lightgbm")
    folds = cross_validate(lambda: make_model("lightgbm", SEED, **params), tr, tr_pts, folds=3,
                           max_rows=150_000, seed=SEED, verbose=False)
    f1 = float(np.mean([f["macro_f1"] for f in folds]))
    print(f"trial {trial.number:2d}: CV macro F1 {f1:.4f}")
    return f1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
t0 = time.time(); study.optimize(objective, n_trials=N_TRIALS); search_s = time.time() - t0
print(f"\\nsearch time {search_s:.0f}s | best CV macro F1 {study.best_value:.4f}")
study.best_params"""),
    md("## Search record and plots"),
    code("""trials = study.trials_dataframe()
os.makedirs("results/02_model_training/notebook_runs", exist_ok=True); trials.to_csv("results/02_model_training/notebook_runs/optuna_trials_lightgbm.csv", index=False)
display(trials[["number", "value"] + [c for c in trials if c.startswith("params_")]].round(4))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(trials.number, trials.value, "o", label="trial"); ax.plot(trials.number, trials.value.cummax(), "r-", label="best so far")
ax.set(xlabel="trial", ylabel="3-fold CV macro F1", title="Optimisation history"); ax.legend(); nb.show()"""),
    code("""imp = optuna.importance.get_param_importances(study)
plt.figure(figsize=(6, 3.5)); plt.barh(list(imp)[::-1], list(imp.values())[::-1])
plt.title("Hyperparameter importance (fANOVA)"); plt.xlabel("importance"); nb.show()
ps = [c for c in trials if c.startswith("params_")]
fig, axs = plt.subplots(2, 4, figsize=(15, 6))
for a, c in zip(axs.ravel(), ps):
    a.scatter(trials[c], trials.value, c=trials.number, cmap="viridis"); a.set(title=c[7:], ylabel="CV macro F1")
    if c[7:] in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda"): a.set_xscale("log")
axs.ravel()[-1].axis("off"); fig.suptitle("Slice plots (colour = trial number)"); fig.tight_layout(); nb.show()"""),
    md("## Untuned vs tuned (same 5-fold CV and student-test protocol)"),
    code("""res = {}
for label, params in [("untuned", {}), ("tuned", study.best_params)]:
    print(f"=== {label}")
    make = lambda p=params: make_model("lightgbm", SEED, **p)
    cv = nb.run_cv(make, tr, tr_pts)
    model, train_m, test_m, p = nb.fit_and_evaluate(make, tr, tr_pts, te, te_pts)
    res[label] = (cv, train_m, test_m, model)
table = pd.DataFrame({k: {"CV macro F1": v[0].macro_f1.mean(), "CV std": v[0].macro_f1.std(), "test Top-1": v[2]["top1"],
                          "test macro F1": v[2]["macro_f1"], "test log loss": v[2]["log_loss"],
                          "test continuity": v[2]["continuity"], "inference ms/point": v[2]["infer_ms_per_point"],
                          "fit seconds": v[2]["fit_s"]} for k, v in res.items()}).round(4)
table"""),
    code("""ms = ["CV macro F1", "test Top-1", "test macro F1", "test log loss"]
x = np.arange(len(ms)); plt.figure(figsize=(8, 3.8))
plt.bar(x - .2, table.loc[ms, "untuned"], .4, label="untuned"); plt.bar(x + .2, table.loc[ms, "tuned"], .4, label="tuned")
plt.xticks(x, ms); plt.title("Untuned vs Bayesian-tuned LightGBM"); plt.legend(); nb.show()
d = table.loc["CV macro F1", "tuned"] - table.loc["CV macro F1", "untuned"]
print(f"CV macro F1 change: {d:+.4f} (fold std about {table.loc['CV std', 'untuned']:.4f})")"""),
    code("""import joblib
joblib.dump(res["tuned"][3], "results/02_model_training/notebook_runs/lightgbm_tuned_notebook.joblib")
print("tuned model saved to results/02_model_training/notebook_runs/ (the deployed model stays models/final.joblib)")"""),
])

# ---------------------------------------------------- 08 HMM + comparison
write("08_HMM_Viterbi_and_Final_Comparison.ipynb", [
    header("08", "HMM / Viterbi route reconstruction and final comparison of all models", "Laptop B", "~5-8 min",
           """Uses the **saved model files** in `models/`, so it runs independently of the other notebooks.

**HMM design (`pathsnap/hmm.py`):**
- **Hidden states:** the candidate arcs of each GPS point.
- **Emission:** `w · log P(classifier says this candidate is the reference arc)`.
- **Transition:** `−|d_route − d_gc| / β`, where `d_route` is the network distance between the two projections (Dijkstra on the directed road graph, with a cut-off) and `d_gc` is the straight-line distance between the GPS points.
- **Edge cases:**
  - unreachable pairs get a floor log-probability, so the chain never breaks
  - gaps over 30 s use uniform transitions (the chain restarts)
  - points without candidates are skipped
- **Viterbi:**
  - **forward pass:** for each point and candidate, keep the best score of any path ending there, `δ_t(j) = max_i [δ_{t−1}(i) + log A(i→j)] + log B_t(j)`, plus a back-pointer
  - **backward pass:** follow the back-pointers from the best final state
- Consecutive decoded arcs that aren't adjacent are joined with shortest paths to form the route.
- Decoder parameters (`models/hmm_params.json`) were tuned with Optuna on **training trips only** (`scripts/04_hmm_eval.py`)."""),
    code(SETUP), code(LOAD),
    md("## 1. All saved models on the student-test trips"),
    code("""import joblib, json
from pathsnap.experiment import evaluate
names = {"logreg": "Logistic Regression", "gnb": "Gaussian NB", "knn": "KNN", "dtree": "Decision Tree",
         "rf": "Random Forest", "extratrees": "Extra Trees", "histgb": "HistGB", "lightgbm": "LightGBM",
         "mlp": "MLP", "final": "LightGBM tuned (deployed)"}
models, rows = {}, []
for k, n in names.items():
    models[k] = joblib.load(f"models/{k}.joblib")
    m, _ = evaluate(models[k], te, te_pts)
    rows.append({"model": n, "test Top-1": m["top1"], "test macro F1": m["macro_f1"], "precision": m["precision"],
                 "recall": m["recall"], "log loss": m["log_loss"], "ms/point": m["infer_ms_per_point"]})
    print("evaluated", n)
cmp = pd.DataFrame(rows).set_index("model")
near = te[te.dist_rank == 0][["trip", "point", "arc"]]
mm = te_pts[te_pts.true_arc >= 0].merge(near, on=["trip", "point"], how="left")
print(f"nearest-arc baseline Top-1: {(mm.arc == mm.true_arc).mean():.4f}")
cmp.sort_values("test Top-1", ascending=False).round(4)"""),
    code("""cv_rec = pd.read_csv("results/03_concurrent_training/model_comparison.csv").set_index("model")   # recorded 5-fold CV (scripts/02b)
c = cmp.drop(index="LightGBM tuned (deployed)").sort_values("test Top-1")
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].barh(c.index, c["test Top-1"]); ax[0].set(title="Student-test Top-1 accuracy", xlim=(0.5, 0.92))
ax[0].axvline((mm.arc == mm.true_arc).mean(), color="k", ls="--", label="nearest-arc baseline"); ax[0].legend()
ax[1].barh(c.index, c["test macro F1"]); ax[1].set(title="Student-test macro F1", xlim=(0.75, 0.95))
ax[2].barh(c.index, c["log loss"], color="#d62728"); ax[2].set(title="Log loss (lower is better)")
fig.tight_layout(); nb.show()
key = {v: k for k, v in names.items()}
cvf = pd.Series({n: cv_rec.loc[key[n], "cv_macro_f1"] for n in c.index}); cvs = pd.Series({n: cv_rec.loc[key[n], "cv_macro_f1_std"] for n in c.index})
plt.figure(figsize=(7, 4)); plt.barh(cvf.index, cvf, xerr=cvs, capsize=3); plt.xlim(0.8, 0.95)
plt.title("Recorded 5-fold CV macro F1 (mean ± std, from scripts/02b)"); plt.tight_layout(); nb.show()"""),
    md("## 2. Classifier only vs classifier + HMM for the top-3 classifiers\nThe same decoder parameters are used for all three; only the emission probabilities change."),
    code("""top3 = {"LightGBM tuned (deployed)": "final", "MLP": "mlp", "HistGB": "histgb"}
hmm_runs = {n: nb.hmm_compare(models[k], te, n) for n, k in top3.items()}
cols = ["cls_top1", "hmm_top1", "cls_route_precision", "hmm_route_precision", "cls_route_f1", "hmm_route_f1",
        "cls_continuity", "hmm_continuity"]
top3_table = pd.DataFrame({n: d[cols].mean() for n, d in hmm_runs.items()}).T.round(4)
top3_table"""),
    code("""fig, ax = plt.subplots(1, 3, figsize=(15, 4)); x = np.arange(3); w = .38
for a, (m, t) in zip(ax, [("top1", "Point Top-1"), ("route_f1", "Route F1"), ("continuity", "Continuity")]):
    a.bar(x - w/2, top3_table[f"cls_{m}"], w, label="classifier only", color="#f28e1c")
    a.bar(x + w/2, top3_table[f"hmm_{m}"], w, label="classifier + HMM", color="#d62728")
    a.set_xticks(x, ["LightGBM", "MLP", "HistGB"]); a.set(title=t, ylim=(0.7, 1.02))
ax[0].legend(loc="lower left"); fig.suptitle("Top-3 classifiers before vs after HMM (mean of 10 student-test trips)")
fig.tight_layout(); nb.show()
nb.plot_hmm(hmm_runs["LightGBM tuned (deployed)"], "Deployed model: per-trip classifier vs classifier + HMM")"""),
    md("## 3. Map: reference vs classifier-only vs HMM route (student_test 00000077, the hardest trip)"),
    code("""from pathsnap.data import load_trip
from pathsnap.pipeline import predict_trip, score_trip
from pathsnap.labels import align_track_to_route
from pathsnap.viz import trip_map
hp = json.load(open("models/hmm_params.json"))
trip = load_trip("data/student_test", "00000077")
out = predict_trip(trip, models["final"], hp)
s = score_trip(trip, out, align_track_to_route(trip, CFG["label_max_dist"])[0])
print({k: round(v, 4) for k, v in s.items() if isinstance(v, float)})
trip_map(trip, [("Reference route", trip.route, "#1a9e3a", 7, None),
                ("Classifier-only route", out["cls_route"], "#f28e1c", 3, "6 6"),
                ("Classifier + HMM route", out["route"], "#d62728", 3, None)],
         f"00000077 | Top-1 cls {s['cls_top1']:.3f} → HMM {s['hmm_top1']:.3f}")"""),
])
print("all notebooks written to", OUT)
