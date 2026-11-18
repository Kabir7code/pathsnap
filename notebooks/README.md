# PathSnap notebooks

There are 8 notebooks, split so that **Laptop A runs 01–04** and **Laptop B runs 05–08**. Each one runs on its own: you don't need to run the others first.

## Before running (both laptops)
1. Get the repo (`git pull`), or unzip `pathsnap_notebooks.zip` into the repo root so the `notebooks/` folder sits next to `pathsnap/`.
2. Put the dataset at `data/train/<id>/` and `data/student_test/<id>/`.
3. Install the requirements, then start Jupyter from the **repo root**:
   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   pip install jupyter
   ```
   ```bash
   jupyter notebook notebooks/
   ```
4. Open a notebook and choose **Kernel → Restart & Run All**.

The first notebook you run on a laptop builds the feature tables in `artifacts/*.parquet` (about 35 s), and later notebooks reuse them.

Every notebook ships **already executed**, so its outputs are visible before you run it. Re-running reproduces the same numbers, because every model, sample and data split uses seed 42.

## What each notebook does

| # | Notebook | Laptop | Runtime* | Models / content | Plots produced |
|---|---|---|---|---|---|
| 01 | `01_EDA_and_preprocessing` | A | ~4 min | No model. EDA on all 90 trips: missing values, timestamps, gaps, coordinate checks, network connectivity, label coverage, candidate miss rate, nearest-arc baseline | Sampling intervals, track lengths, GPS-to-road distance, trip locations, network size, example trip, **interactive Folium map**, feature distributions by class |
| 02 | `02_Logistic_Regression_and_Naive_Bayes` | A | ~1 min | **Logistic Regression**, **Gaussian Naive Bayes** | Per model: feature importance, classifier vs HMM per trip. Comparison: CV per fold, overfitting check, confusion matrices, calibration |
| 03 | `03_KNN` | A | ~6–10 min | **KNN** (k = 25, distance-weighted) | Same set of plots as 02 |
| 04 | `04_Decision_Tree_Random_Forest_Extra_Trees` | A | ~8–12 min | **Decision Tree**, **Random Forest**, **Extra Trees** | Same set as 02, for 3 models |
| 05 | `05_HistGradientBoosting_and_LightGBM` | B | ~4–6 min | **HistGradientBoosting**, **LightGBM** | Same set as 02, plus the **LightGBM training curve** (log loss per boosting round, train vs held-out trips) |
| 06 | `06_MLP_Neural_Network` | B | ~6–8 min | **MLP** (14 → 64 → 32 → 1) | Same set as 02, plus **MLP training loss and validation accuracy per epoch** |
| 07 | `07_Bayesian_Hyperparameter_Optimisation` | B | ~10–12 min | **Bayesian optimisation of LightGBM** (Optuna TPE, 20 trials), then **tuned vs untuned** | Optimisation history, hyperparameter importance, slice plots per parameter, tuned vs untuned bar chart |
| 08 | `08_HMM_Viterbi_and_Final_Comparison` | B | ~5–8 min | **All 10 saved models** on student test, **HMM/Viterbi** before/after for the top 3, route map | Test Top-1 / F1 / log loss for all models, recorded CV F1 ± std, top-3 before/after HMM, per-trip HMM chart, **Folium map** comparing reference, classifier and HMM routes |

\*Approximate, on a 4-core laptop.

## Inside every model notebook (02–06), for each model
1. **Explanation:** what the algorithm does, why it suits PathSnap, its strengths and limitations, its hyperparameters, and its loss or objective. These match the questions the report asks for each model.
2. **5-fold GroupKFold cross-validation** by trip, showing macro F1, precision, recall, log loss and Top-1 for each fold.
3. **Fit on the training split, evaluate on the 10 student-test trips:** a table comparing train, CV and test results, which is the **overfitting check**.
4. **Feature importance:** native importances for trees and boosting, coefficients for logistic regression, permutation importance for the others.
5. **Classifier only vs classifier + HMM/Viterbi** on each test trip.
6. Results saved to `results/notebooks/<model>.json`.

At the end of each notebook, the models in it are compared: CV per fold, overfitting bars, confusion matrices, a calibration curve and a summary table.

## How the metrics are computed (`pathsnap/metrics.py`)
- **Top-1 accuracy:** the share of labelled GPS points whose highest-scoring candidate is the reference arc. If the reference arc isn't among a point's candidates, that point counts as wrong.
- **Macro F1 / precision / recall:** the yes/no decision per candidate ("is this the reference arc?") at threshold 0.5. Macro F1 averages both classes.
- **Log loss:** the quality of the predicted probabilities.
- **Continuity:** the share of consecutive GPS points whose predicted arcs are the same arc or directly connected.
- **Route precision / recall / F1:** the overlap between the predicted route's arcs and the reference route's arcs.
- ⚠ The HMM tables average **per trip**; the model tables **pool all points**. Compare like with like.
