# PathSnap notebooks

There are 9 notebooks, split so that **Laptop A runs 01–04** and **Laptop B runs 05–08**. **09** reads recorded results, takes about 1 minute, and runs anywhere. Each one runs on its own: you don't need to run the others first.

## Run on Google Colab (no install needed)
1. Upload **`Student_folder.zip`** (the dataset) to your Google Drive, in **My Drive**.
2. Click a link below, then in Colab choose **Runtime → Run all**. Allow Drive access when asked, or upload the zip when prompted.
3. The first cell clones the repo, installs `optuna` and the other packages, and unpacks the dataset. Expect about 1–2 minutes.

| Laptop | Notebook |
|---|---|
| A | [01 EDA & preprocessing](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/01_EDA_and_preprocessing.ipynb) |
| A | [02 Logistic Regression & Naive Bayes](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/02_Logistic_Regression_and_Naive_Bayes.ipynb) |
| A | [03 KNN](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/03_KNN.ipynb) |
| A | [04 Decision Tree, Random Forest, Extra Trees](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/04_Decision_Tree_Random_Forest_Extra_Trees.ipynb) |
| B | [05 HistGB & LightGBM](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/05_HistGradientBoosting_and_LightGBM.ipynb) |
| B | [06 MLP](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/06_MLP_Neural_Network.ipynb) |
| B | [07 Bayesian optimisation](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/07_Bayesian_Hyperparameter_Optimisation.ipynb) |
| B | [08 HMM/Viterbi & final comparison](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/08_HMM_Viterbi_and_Final_Comparison.ipynb) |
| any | [09 Final model selection](https://colab.research.google.com/github/Kabir7code/pathsnap/blob/main/notebooks/09_Final_Model_Selection.ipynb) |

Free Colab has only 2 CPU cores, so runtimes are roughly **twice** the figures below. KNN (03) and Bayesian optimisation (07) are the slowest, at about 20–40 min each on Colab.

## Run locally (both laptops)
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

The first notebook you run on a laptop builds the feature tables in `data/features/*.parquet` (about 35 s), and later notebooks reuse them.

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
| 09 | `09_Final_Model_Selection` | any | ~1 min | **LightGBM vs Random Forest vs HistGB** as full systems (classifier + HMM) on 88 development trips; paired bootstrap CIs; pre-declared decision rule; final tuning of LightGBM with the HMM in the loop | Before/after-HMM bars for the 3 models, CI plot of the differences, final-tuning history |

\*Approximate, on a 4-core laptop.

## Inside every model notebook (02–06), for each model
1. **Explanation:** what the algorithm does, why it suits PathSnap, its strengths and limitations, its hyperparameters, and its loss or objective. These match the questions the report asks for each model.
2. **5-fold GroupKFold cross-validation** by trip, showing macro F1, precision, recall, log loss and Top-1 for each fold.
3. **Fit on the training split, evaluate on the 10 student-test trips:** a table comparing train, CV and test results, which is the **overfitting check**.
4. **Feature importance:** native importances for trees and boosting, coefficients for logistic regression, permutation importance for the others.
5. **Classifier only vs classifier + HMM/Viterbi** on each test trip.
6. Results saved to `results/02_model_training/notebook_runs/<model>.json`.

At the end of each notebook, the models in it are compared: CV per fold, overfitting bars, confusion matrices, a calibration curve and a summary table.

## How the metrics are computed (`pathsnap/metrics.py`)
- **Top-1 accuracy:** the share of labelled GPS points whose highest-scoring candidate is the reference arc. If the reference arc isn't among a point's candidates, that point counts as wrong.
- **Macro F1 / precision / recall:** the yes/no decision per candidate ("is this the reference arc?") at threshold 0.5. Macro F1 averages both classes.
- **Log loss:** the quality of the predicted probabilities.
- **Continuity:** the share of consecutive GPS points whose predicted arcs are the same arc or directly connected.
- **Route precision / recall / F1:** the overlap between the predicted route's arcs and the reference route's arcs.
- ⚠ The HMM tables average **per trip**; the model tables **pool all points**. Compare like with like.
