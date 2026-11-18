"""PathSnap FastAPI backend (runs on Laptop 1).

Run:
  PATHSNAP_EVAL_DIR=/path/to/evaluator_trips uvicorn app.api.server:app --host 0.0.0.0 --port 8000

Models: every saved model in models/ can be run.  "final" (the tuned LightGBM)
is the deployed default; the other nine are the trained comparison models.
All of them share the same candidate generation, features and HMM decoder
parameters (models/hmm_params.json, tuned on LightGBM probabilities).

PATHSNAP_EVAL_DIR is a folder of trip sub-folders in the raw format
(<id>/<id>.nodes/.arcs/.track and, server-side only, <id>.route).
Defaults to data/student_test for development.

Ground-truth isolation: the prediction pipeline loads each trip WITHOUT its
.route file.  The reference route is read separately, only inside
_score(), and only aggregate metrics are returned -- never the reference
geometry or per-point labels.
"""
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pathsnap.config import CFG
from pathsnap.data import list_trips, load_trip
from pathsnap.labels import align_track_to_route
from pathsnap.models import MODEL_INFO
from pathsnap.pipeline import predict_trip, score_trip
from pathsnap.viz import polylines_from_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(CFG["data_dir"], "student_test"))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "final.joblib")
HMM_PATH = os.path.join(MODELS_DIR, "hmm_params.json")
DEFAULT_MODEL = "final"

hmm_params = json.load(open(HMM_PATH))
model_meta = json.load(open(os.path.join(MODELS_DIR, "final_model.json")))

# Model registry: the deployed model first, then the nine comparison models in
# the order of their recorded student-test Top-1 (results/02_model_training).
_cmp_path = os.path.join(ROOT, "results", "02_model_training", "model_comparison.csv")
_tuned_path = os.path.join(ROOT, "results", "04_bayesian_optimisation", "tuned_vs_untuned.csv")
_conc_path = os.path.join(ROOT, "results", "03_concurrent_training", "model_comparison.csv")
_cmp = pd.read_csv(_cmp_path).set_index("model") if os.path.exists(_cmp_path) else pd.DataFrame()
if os.path.exists(_conc_path):   # KNN was only recorded in the concurrent run (same protocol)
    _conc = pd.read_csv(_conc_path).set_index("model")
    _cmp = pd.concat([_cmp, _conc.loc[[m for m in _conc.index if m not in _cmp.index]]])
_tuned = (pd.read_csv(_tuned_path).set_index("variant").loc["tuned"] if os.path.exists(_tuned_path) else None)
_REC = ["cv_macro_f1", "cv_macro_f1_std", "test_top1", "test_macro_f1", "test_log_loss", "test_continuity"]


def _recorded(row):
    return {k: (None if row is None or k not in row or pd.isna(row[k]) else float(row[k])) for k in _REC}


REGISTRY = {DEFAULT_MODEL: {
    "id": DEFAULT_MODEL, "name": "LightGBM (tuned, deployed)", "family": "Boosting",
    "config": "LightGBM after Bayesian optimisation (Optuna TPE, %s trials)" % model_meta.get("trials", "?"),
    "file": "models/final.joblib", "default": True, "recorded": _recorded(_tuned)}}
_order = list(_cmp.sort_values("test_top1", ascending=False).index) if len(_cmp) else list(MODEL_INFO)
for name in _order + [m for m in MODEL_INFO if m not in _order]:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if name in MODEL_INFO and os.path.exists(path):
        REGISTRY[name] = {"id": name, "name": {"logreg": "Logistic regression", "gnb": "Gaussian Naive Bayes",
                                               "knn": "KNN (k=25)", "dtree": "Decision tree", "rf": "Random forest",
                                               "extratrees": "Extra Trees", "histgb": "HistGradientBoosting",
                                               "lightgbm": "LightGBM (untuned)", "mlp": "MLP (64, 32)"}[name],
                          "family": MODEL_INFO[name][0], "config": MODEL_INFO[name][1],
                          "file": f"models/{name}.joblib", "default": False,
                          "recorded": _recorded(_cmp.loc[name] if name in _cmp.index else None)}

_loaded = {}


def get_model(name):
    """Load a model on first use and keep it in memory."""
    if name not in _loaded:
        t0 = time.time()
        _loaded[name] = joblib.load(os.path.join(ROOT, REGISTRY[name]["file"]))
        log.info("loaded model %s (%s) in %.1fs", name, REGISTRY[name]["name"], time.time() - t0)
    return _loaded[name]


model = get_model(DEFAULT_MODEL)
log.info("loaded model %s from %s", model_meta["model"], MODEL_PATH)
log.info("models available: %s", ", ".join(REGISTRY))
log.info("HMM params %s", hmm_params)
log.info("evaluation trips dir: %s (%d trips)", EVAL_DIR, len(list_trips(EVAL_DIR)))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str
    model: str = DEFAULT_MODEL


@app.get("/health")
def health():
    return {"status": "ok", "model": model_meta["model"], "eval_dir": os.path.basename(EVAL_DIR)}


@app.get("/info")
def info(model: str = DEFAULT_MODEL):
    """Model card for the UI: model config, decoder params and pipeline settings (no labels)."""
    if model not in REGISTRY:
        raise HTTPException(404, f"unknown model {model}")
    log.info("GET /info model=%s", model)
    if model == DEFAULT_MODEL:
        algo, params = model_meta["model"], model_meta["params"]
    else:
        est = get_model(model)
        est = est.steps[-1][1] if hasattr(est, "steps") else est      # the classifier inside a scaler pipeline
        algo = type(est).__name__
        params = {k: v for k, v in sorted(est.get_params(deep=False).items())
                  if isinstance(v, (int, float, str, bool, tuple)) and not isinstance(v, bool) or isinstance(v, bool)}
        params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items() if v is not None}
    return {"model": model, "model_name": REGISTRY[model]["name"], "algorithm": algo, "family": REGISTRY[model]["family"],
            "config": REGISTRY[model]["config"], "model_params": params, "recorded": REGISTRY[model]["recorded"],
            "bayes_trials": model_meta.get("trials") if model == DEFAULT_MODEL else None, "hmm_params": hmm_params,
            "candidate_radius_m": CFG["radius"], "max_candidates": CFG["k"],
            "eval_dir": os.path.basename(os.path.normpath(EVAL_DIR))}


@app.get("/models")
def models():
    """Every runnable model with its family, configuration and recorded results (no labels)."""
    log.info("GET /models -> %d models", len(REGISTRY))
    return {"default": DEFAULT_MODEL, "models": list(REGISTRY.values()),
            "note": "All models share the same candidates, features and HMM decoder parameters "
                    "(tuned on LightGBM probabilities). 'recorded' values come from results/ "
                    "(5-fold GroupKFold CV on train, and the 10 student-test trips)."}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(EVAL_DIR):
        with open(os.path.join(EVAL_DIR, tid, f"{tid}.track")) as f:
            n = sum(1 for line in f if line.strip())
        out.append({"trip_id": tid, "gps_points": n})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


def _score(tid, out):
    """Server-side only: compare predictions with the hidden reference route."""
    ref = load_trip(EVAL_DIR, tid, with_route=True)
    if ref.route is None:
        return None
    true_arc, _, _ = align_track_to_route(ref, CFG["label_max_dist"])
    return score_trip(ref, out, true_arc)


@app.post("/predict")
def predict(req: PredictRequest):
    tid, mname = req.trip_id, req.model
    if tid not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {tid}")
    if mname not in REGISTRY:
        raise HTTPException(404, f"unknown model {mname}; choose one of {list(REGISTRY)}")
    log.info("POST /predict trip=%s model=%s : running pipeline", tid, mname)
    clf = get_model(mname)
    t0 = time.time()
    trip = load_trip(EVAL_DIR, tid, with_route=False)      # <- no labels in the pipeline
    out = predict_trip(trip, clf, hmm_params)
    pred_s = time.time() - t0
    metrics = _score(tid, out)
    log.info("trip=%s model=%s points=%d candidates=%d predicted_route_arcs=%d time=%.2fs",
             tid, mname, len(trip.times), sum(len(c) for c in out["cands"]), len(out["route"]), pred_s)
    if metrics:
        log.info("trip=%s model=%s METRICS top1 cls=%.4f hmm=%.4f | route F1 cls=%.4f hmm=%.4f | continuity hmm=%.4f",
                 tid, mname, metrics["cls_top1"], metrics["hmm_top1"], metrics["cls_route_f1"],
                 metrics["hmm_route_f1"], metrics["hmm_continuity"])
    return {
        "trip_id": tid,
        "model": mname,
        "model_name": REGISTRY[mname]["name"],
        "gps": [[float(lat), float(lon)] for lon, lat in trip.track_ll],
        "predicted_route": polylines_from_route(trip, out["route"]),
        "classifier_only_route": polylines_from_route(trip, out["cls_route"]),
        "predicted_route_arc_ids": [int(a) for a in out["route"]],
        "metrics": metrics,
        "timing": {k: round(v, 3) for k, v in out["timing"].items()},
        "hmm_stats": out["hmm_stats"],
    }
