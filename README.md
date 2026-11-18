# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS tracks to a directed road network. It does this in three steps:

1. **Candidate generation.** For each GPS point, find the K nearest road arcs within R metres.
2. **Candidate classification.** A supervised model scores *P(candidate is the true arc)*. Nine models from seven algorithm families are compared, and the chosen one is tuned with Bayesian optimisation.
3. **HMM + Viterbi decoding.** The classifier probabilities are the emissions. The transitions come from the road network's shortest-path distances. Decoding gives a connected route.

A FastAPI backend (laptop 1) runs the full pipeline for a selected evaluation trip and scores it against the reference route, which stays on the server. A Leaflet UI (laptop 2) lists the trips, sends requests and shows the predicted route and its metrics.

## Repository layout

```
pathsnap/            core library (imported by every script, the API and the maps)
  data.py            raw file loading, cleaning, metric projection
  geometry.py        point-to-segment distance, angle difference
  candidates.py      KD-tree candidate generation
  labels.py          per-GPS-point labels from the .route file (monotonic alignment)
  features.py        feature engineering for (point, candidate) pairs
  models.py          model zoo (9 classifiers)
  experiment.py      shared CV / test protocol
  hmm.py             road graph, cached Dijkstra, Viterbi, route reconstruction
  pipeline.py        end-to-end inference for one trip + scoring
  metrics.py         candidate, observation and sequence-level metrics
  viz.py             Folium map building
  config.py          all pipeline settings
scripts/
  00_eda.py              dataset exploration -> results/eda/
  01_build_dataset.py    candidates + features + labels -> artifacts/*.parquet
  02_train_models.py     train & compare all models -> results/model_comparison.csv
  02b_train_concurrent.py  same protocol, all 9 models trained concurrently in parallel worker processes -> results/concurrent/
  03_bayes_opt.py        Optuna TPE search for the chosen model -> artifacts/models/final.joblib
  04_hmm_eval.py         decoder tuning + classifier vs classifier+HMM -> results/hmm_*.csv
  05_folium_maps.py      interactive maps -> maps/*.html
  06_plots.py            comparison plots -> results/plots/
  07_model_cards.py      open every saved model, summarise it -> results/model_cards.md
  08_models_dashboard.py one-page HTML of all models -> results/models_dashboard.html
  09_hmm_top3.py         classifier vs classifier+HMM for the top-3 models -> results/hmm_top3.csv
  make_notebooks.py      regenerates notebooks/*.ipynb
notebooks/           8 executed Jupyter notebooks (EDA, every model family, Bayesian opt., HMM) - Colab-ready
api/server.py        FastAPI backend (laptop 1)
ui/index.html        evaluator UI (laptop 2)
artifacts/           trained models, final config, HMM parameters
results/             experiment logs, CSVs, Optuna studies (.db), plots
maps/                generated Folium HTML maps
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the raw dataset in `data/`, which gives `data/train/<trip_id>/...` and `data/student_test/<trip_id>/...`.

## Reproducing the experiments

Run the scripts in order from the repository root:

```bash
python scripts/00_eda.py
python scripts/01_build_dataset.py
python scripts/02_train_models.py
python scripts/02b_train_concurrent.py --workers 4   # concurrent version (all 9 models in parallel)
python scripts/03_bayes_opt.py --model lightgbm --trials 30
python scripts/04_hmm_eval.py --trials 25
python scripts/05_folium_maps.py --split student_test
python scripts/06_plots.py
python scripts/07_model_cards.py
python scripts/08_models_dashboard.py
python scripts/09_hmm_top3.py
```

### Notebooks
The same experiments, with explanations and plots, are in `notebooks/`. They're saved **with outputs**, so you can read them without running anything.
- **Laptop A** runs 01–04.
- **Laptop B** runs 05–08.

Each notebook runs on its own: `jupyter notebook notebooks/`, then Kernel → Restart & Run All. Each one also opens in **Google Colab** via the badge at its top. See `notebooks/README.md` for the Colab links and what each notebook contains.

Every script fixes its random seed from `pathsnap/config.py` (`seed = 42`). Steps 02–04 are the slow ones, taking minutes to tens of minutes on a 4-core laptop.

## Running the demo (two laptops)

**Step-by-step guide, troubleshooting and the API reference are in [`DEPLOYMENT.md`](DEPLOYMENT.md).** Quick start: on laptop 1 run `start_api.bat` (Windows) or `./start_api.sh`; on laptop 2 run `start_ui.bat <laptop1-ip>` or `./start_ui.sh <laptop1-ip>`. Check laptop 1 first with `python scripts/check_setup.py`.


**Laptop 1: API.** Put the 10 evaluator trips in a folder, one sub-folder per trip in the raw format (including `.route`), then run:

```bash
PATHSNAP_EVAL_DIR=/path/to/eval_trips uvicorn api.server:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell, use `$env:PATHSNAP_EVAL_DIR="C:\path\to\eval_trips"; uvicorn api.server:app --host 0.0.0.0 --port 8000`.

If you don't set `PATHSNAP_EVAL_DIR`, it defaults to `data/student_test`. Every request is logged to the console with the trip, its timing and its metrics. Show this console to the evaluator.

**Laptop 2: UI.**

```bash
cd ui && python -m http.server 8080
```

Open `http://localhost:8080/?api=http://<laptop-1-ip>:8000`, press **Connect**, then pick a trip. Both laptops must be on the same network, and laptop 1's firewall must allow port 8000.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model name, status |
| GET | `/trips` | evaluation trip IDs and GPS point counts |
| POST | `/predict` `{"trip_id": "..."}` | run the pipeline; return GPS points, the predicted route, the classifier-only route, server-side metrics and timings |

### How the reference labels are kept separate

`/predict` loads the trip with `with_route=False`, so the prediction pipeline (`pathsnap.pipeline.predict_trip`) never sees the `.route` file. The reference route is read only inside `_score()` in `api/server.py`. The response contains only aggregate metrics, never the reference geometry or per-point labels. Nothing is hard-coded: every request recomputes candidates, features, classifier scores and Viterbi decoding.

## Key design decisions

- **Projection.** A local equirectangular projection centred on each trip's network, giving coordinates in metres. At city scale (under 30 km) its distortion is far below GPS noise.
- **Per-point labels.** `.route` lists the arcs in order but doesn't say which GPS point belongs to which arc. We align the two with a monotonic dynamic program: route position never goes backwards, and the total point-to-arc distance is minimised. Points more than 60 m from their aligned arc are left unlabelled.
- **Excluded training trips.** Two training trips (`00000054`, `00000033`) have under 60% of points labelled. Their GPS track runs 100–190 m away from the reference route, or the network is missing roads near the track. They are excluded from fitting (`min_trip_label_frac`) because their labels are unreliable.
- **Problem framing.** Each row is one (GPS point, candidate arc) pair with a binary label. The top-scoring candidate is the classifier-only prediction, and the probabilities feed the HMM.
- **Validation.** `GroupKFold` by trip, so points from one trip never appear in both the fitting and the validation part of a fold. The student-test trips are used only for comparison, never for fitting.
- **Decoder tuning.** Decoder parameters are tuned on **out-of-fold** probabilities from the training trips.

## Where each required deliverable is (Deliverables PDF, section 1)

| Required component | Location |
|---|---|
| Data processing | `pathsnap/data.py`, `labels.py`, `candidates.py`, `scripts/00_eda.py`, `scripts/01_build_dataset.py`, `notebooks/01_*` |
| Feature engineering | `pathsnap/features.py` (14 features, documented in the module docstring) |
| Model training: every model | `pathsnap/models.py`, `scripts/02_train_models.py`, `scripts/02b_train_concurrent.py`, `notebooks/02_*` to `06_*` |
| Model evaluation: metrics, plots, confusion matrices | `pathsnap/metrics.py`, `pathsnap/experiment.py`, `scripts/06_plots.py`, `results/plots/`, `results/model_comparison.csv`, `results/concurrent/` |
| Hyperparameter optimisation: code, search config, results | `scripts/03_bayes_opt.py`, `notebooks/07_*`, `results/optuna_lightgbm.db`, `results/optuna_trials_lightgbm.csv`, `results/tuned_vs_untuned.csv`, `results/optuna_hmm.db` |
| HMM and Viterbi | `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/04_hmm_eval.py`, `scripts/09_hmm_top3.py`, `notebooks/08_*`, `results/hmm_*.csv` |
| API | `api/server.py` (endpoints `/health`, `/info`, `/trips`, `/predict`) |
| UI | `ui/index.html` (+ `ui/vendor/leaflet/`) |
| Deployment: config, dependencies, instructions | `requirements.txt`, `pathsnap/config.py`, "Running the demo" above |
| Saved model weights and configuration | `artifacts/models/*.joblib` (9 models + `final.joblib`), `artifacts/final_model.json`, `artifacts/hmm_params.json` |
| Visualisations: Folium and plots | `pathsnap/viz.py`, `scripts/05_folium_maps.py`, `maps/*.html`, `results/eda/`, `results/plots/` |
| Supporting results: CSVs, logs, configs | `results/` (CSVs, JSON, Optuna studies, `*_log.txt`, `api_test_log.txt`, UI screenshots), `results/model_cards.md`, `results/models_dashboard.html` |

**Not in the repo, by design:**
- the raw dataset (`data/`)
- the evaluator trips (`eval_data/`)
- the generated feature tables (`artifacts/*.parquet`; rebuild them with `scripts/01_build_dataset.py`)

The **technical report (PDF)** is submitted separately. Every number in it can be traced to a file in `results/`.

## AI assistance
Parts of this repository's code were written with the help of an AI assistant (Claude). The experiments were run, and the results recorded, by the scripts and notebooks in this repository.
