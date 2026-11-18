# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS tracks to a directed road network. It does this in three steps:

1. **Candidate generation.** For each GPS point, find the K nearest road arcs within R metres.
2. **Candidate classification.** A supervised model scores *P(candidate is the true arc)*. Nine models from seven algorithm families are compared, and the chosen one is tuned with Bayesian optimisation.
3. **HMM + Viterbi decoding.** The classifier probabilities are the emissions. The transitions come from the road network's shortest-path distances. Decoding gives a connected route.

A FastAPI backend (laptop 1) runs the full pipeline for a selected evaluation trip and scores it against the reference route, which stays on the server. A Leaflet UI (laptop 2) lists the trips, sends requests and shows the predicted route and its metrics.

## Repository layout

```
pathsnap/                  core library (imported by every script, the API and the notebooks)
  data.py                  raw file loading, cleaning, metric projection
  geometry.py              point-to-segment distance, angle difference
  candidates.py            KD-tree candidate generation
  labels.py                per-GPS-point labels from the .route file (monotonic alignment)
  features.py              feature engineering for (point, candidate) pairs
  models.py                model zoo (9 classifiers)
  dataset.py               loads the feature tables; ART = data/features, MODELS = models
  experiment.py            shared CV / test protocol
  hmm.py                   road graph, cached Dijkstra, Viterbi, route reconstruction
  pipeline.py              end-to-end inference for one trip + scoring
  metrics.py               candidate, observation and sequence-level metrics
  viz.py                   Folium map building
  nb.py                    helpers used by the notebooks
  config.py                all pipeline settings
scripts/                   the pipeline, run in numeric order (see "Reproducing the experiments")
notebooks/                 8 executed Jupyter notebooks (EDA, every model family, Bayesian opt., HMM), Colab-ready
app/
  api/server.py            FastAPI backend (laptop 1)
  ui/index.html            evaluator UI (laptop 2), Leaflet bundled in ui/vendor/
models/                    trained models: 9 x <name>.joblib, final.joblib (deployed), final_model.json, hmm_params.json
results/
  00_overview/             models_dashboard.html (all models on one page), model_cards.md
  01_eda_preprocessing/    EDA summary, inventory, plots; label coverage (labelling_*.csv); logs
  02_model_training/       model_comparison.csv, model_runs.json, plots/, notebook_runs/ (per-notebook results)
  03_concurrent_training/  all 9 models trained in parallel: comparison, timeline, speed-up
  04_bayesian_optimisation/ Optuna study (.db), trial record, history plot, tuned vs untuned
  05_hmm_viterbi/          decoder tuning, classifier vs classifier+HMM, top-3 comparison
  06_folium_maps/          interactive Folium maps (EDA and student-test predictions)
  07_api_ui/               API end-to-end log, UI screenshots
report/                    technical report (PDF)
deploy/                    start_api / start_ui scripts (macOS/Linux .sh, Windows .bat)
docs/DEPLOYMENT.md         step-by-step two-laptop guide and troubleshooting
data/                      NOT in git: raw dataset (train/, student_test/) and generated features/
```

### Script outputs

| Script | Writes to |
|---|---|
| `00_eda.py` | `results/01_eda_preprocessing/` |
| `01_build_dataset.py` | `data/features/*.parquet`, `results/01_eda_preprocessing/labelling_*.csv` |
| `02_train_models.py` | `models/<name>.joblib`, `results/02_model_training/` |
| `02b_train_concurrent.py` | `results/03_concurrent_training/` |
| `03_bayes_opt.py` | `models/final.joblib`, `models/final_model.json`, `results/04_bayesian_optimisation/` |
| `04_hmm_eval.py` | `models/hmm_params.json`, `results/05_hmm_viterbi/` |
| `05_folium_maps.py` | `results/06_folium_maps/` |
| `06_plots.py` | `results/02_model_training/plots/`, `results/04_bayesian_optimisation/tuned_vs_untuned.png` |
| `07_model_cards.py`, `08_models_dashboard.py` | `results/00_overview/` |
| `09_hmm_top3.py` | `results/05_hmm_viterbi/hmm_top3*` |
| `check_setup.py` | pre-flight check for the demo (prints this laptop's IP) |
| `make_notebooks.py` | regenerates `notebooks/*.ipynb` |

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

**Step-by-step guide, troubleshooting and the API reference are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).**

**Laptop 1: API.** Run the pre-flight check and start the API (it prints the IP for laptop 2):

```bash
./deploy/start_api.sh               # Windows: deploy\start_api.bat
./deploy/start_api.sh eval_data     # demo day: the evaluator trips in eval_data/
```

Without the script: `PATHSNAP_EVAL_DIR=/path/to/eval_trips uvicorn app.api.server:app --host 0.0.0.0 --port 8000` from the repository root. If `PATHSNAP_EVAL_DIR` is not set it defaults to `data/student_test`. Every request is logged to this console with the trip, its timing and its metrics: show it to the evaluator.

**Laptop 2: UI.**

```bash
./deploy/start_ui.sh <laptop-1-ip>  # Windows: deploy\start_ui.bat <laptop-1-ip>
```

Open `http://localhost:8080/?api=http://<laptop-1-ip>:8000`, press **Connect**, then pick a trip. Both laptops must be on the same network, and laptop 1's firewall must allow port 8000.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model name, status |
| GET | `/trips` | evaluation trip IDs and GPS point counts |
| POST | `/predict` `{"trip_id": "..."}` | run the pipeline; return GPS points, the predicted route, the classifier-only route, server-side metrics and timings |

### How the reference labels are kept separate

`/predict` loads the trip with `with_route=False`, so the prediction pipeline (`pathsnap.pipeline.predict_trip`) never sees the `.route` file. The reference route is read only inside `_score()` in `app/api/server.py`. The response contains only aggregate metrics, never the reference geometry or per-point labels. Nothing is hard-coded: every request recomputes candidates, features, classifier scores and Viterbi decoding.

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
| Data processing | `pathsnap/data.py`, `labels.py`, `candidates.py`, `scripts/00_eda.py`, `scripts/01_build_dataset.py`, `notebooks/01_*`, `results/01_eda_preprocessing/` |
| Feature engineering | `pathsnap/features.py` (14 features, documented in the module docstring) |
| Model training: every model | `pathsnap/models.py`, `scripts/02_train_models.py`, `scripts/02b_train_concurrent.py`, `notebooks/02_*` to `06_*` |
| Model evaluation: metrics, plots, confusion matrices | `pathsnap/metrics.py`, `pathsnap/experiment.py`, `scripts/06_plots.py`, `results/02_model_training/`, `results/03_concurrent_training/` |
| Hyperparameter optimisation: code, search config, results | `scripts/03_bayes_opt.py`, `notebooks/07_*`, `results/04_bayesian_optimisation/` |
| HMM and Viterbi | `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/04_hmm_eval.py`, `scripts/09_hmm_top3.py`, `notebooks/08_*`, `results/05_hmm_viterbi/` |
| API | `app/api/server.py` (endpoints `/health`, `/info`, `/trips`, `/predict`) |
| UI | `app/ui/index.html` (+ `app/ui/vendor/leaflet/`) |
| Deployment: config, dependencies, instructions | `requirements.txt`, `pathsnap/config.py`, `deploy/`, `docs/DEPLOYMENT.md` |
| Saved model weights and configuration | `models/*.joblib` (9 models + `final.joblib`), `models/final_model.json`, `models/hmm_params.json` |
| Visualisations: Folium and plots | `pathsnap/viz.py`, `scripts/05_folium_maps.py`, `results/06_folium_maps/`, the `plots/` and `.png` files in each `results/` stage |
| Supporting results: CSVs, logs, configs | every `results/` stage folder (CSVs, JSON, Optuna studies, `*_log.txt`), `results/07_api_ui/` (API log, UI screenshots), `results/00_overview/` |
| Technical report | `report/` |

**Not in the repo, by design:**
- the raw dataset (`data/`)
- the evaluator trips (`eval_data/`)
- the generated feature tables (`data/features/*.parquet`; rebuild them with `scripts/01_build_dataset.py`)

Every number in the technical report can be traced to a file in `results/`.

## AI assistance
Parts of this repository's code were written with the help of an AI assistant (Claude). The experiments were run, and the results recorded, by the scripts and notebooks in this repository.
