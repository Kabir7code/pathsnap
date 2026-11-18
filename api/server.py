"""PathSnap FastAPI backend (runs on Laptop 1).

Run:
  PATHSNAP_EVAL_DIR=/path/to/evaluator_trips uvicorn api.server:app --host 0.0.0.0 --port 8000

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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.config import CFG
from pathsnap.data import list_trips, load_trip
from pathsnap.labels import align_track_to_route
from pathsnap.pipeline import predict_trip, score_trip
from pathsnap.viz import polylines_from_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(CFG["data_dir"], "student_test"))
MODEL_PATH = os.path.join(ROOT, "artifacts", "models", "final.joblib")
HMM_PATH = os.path.join(ROOT, "artifacts", "hmm_params.json")

model = joblib.load(MODEL_PATH)
hmm_params = json.load(open(HMM_PATH))
model_meta = json.load(open(os.path.join(ROOT, "artifacts", "final_model.json")))
log.info("loaded model %s from %s", model_meta["model"], MODEL_PATH)
log.info("HMM params %s", hmm_params)
log.info("evaluation trips dir: %s (%d trips)", EVAL_DIR, len(list_trips(EVAL_DIR)))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": model_meta["model"], "eval_dir": os.path.basename(EVAL_DIR)}


@app.get("/info")
def info():
    """Model card for the UI: final model config, decoder params and pipeline settings (no labels)."""
    log.info("GET /info")
    return {"model": model_meta["model"], "model_params": model_meta["params"],
            "bayes_trials": model_meta.get("trials"), "hmm_params": hmm_params,
            "candidate_radius_m": CFG["radius"], "max_candidates": CFG["k"],
            "eval_dir": os.path.basename(os.path.normpath(EVAL_DIR))}


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
    tid = req.trip_id
    if tid not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("POST /predict trip=%s : running pipeline", tid)
    t0 = time.time()
    trip = load_trip(EVAL_DIR, tid, with_route=False)      # <- no labels in the pipeline
    out = predict_trip(trip, model, hmm_params)
    pred_s = time.time() - t0
    metrics = _score(tid, out)
    log.info("trip=%s points=%d candidates=%d predicted_route_arcs=%d time=%.2fs",
             tid, len(trip.times), sum(len(c) for c in out["cands"]), len(out["route"]), pred_s)
    if metrics:
        log.info("trip=%s METRICS top1 cls=%.4f hmm=%.4f | route F1 cls=%.4f hmm=%.4f | continuity hmm=%.4f",
                 tid, metrics["cls_top1"], metrics["hmm_top1"], metrics["cls_route_f1"],
                 metrics["hmm_route_f1"], metrics["hmm_continuity"])
    return {
        "trip_id": tid,
        "model": model_meta["model"],
        "gps": [[float(lat), float(lon)] for lon, lat in trip.track_ll],
        "predicted_route": polylines_from_route(trip, out["route"]),
        "classifier_only_route": polylines_from_route(trip, out["cls_route"]),
        "predicted_route_arc_ids": [int(a) for a in out["route"]],
        "metrics": metrics,
        "timing": {k: round(v, 3) for k, v in out["timing"].items()},
        "hmm_stats": out["hmm_stats"],
    }
