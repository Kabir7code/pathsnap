"""Central configuration (pipeline settings shared by every script)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = {
    "data_dir": os.path.join(ROOT, "data"),
    "seed": 42,
    # candidate generation
    "radius": 50.0,         # metres: max GPS-to-arc distance considered
    "k": 8,                 # max candidates per GPS point
    "window": 2,            # +/- points used for the windowed distance feature
    # labelling
    "label_max_dist": 60.0, # points farther than this from their aligned route arc are unlabelled
    "min_trip_label_frac": 0.6,  # trips below this labelled fraction are excluded from training (data anomalies)
    # HMM defaults (tuned later by scripts/05_hmm_eval.py)
    "hmm": {"beta": 10.0, "emission_weight": 1.0, "dijkstra_cutoff": 1500.0,
            "unreachable_logp": -30.0, "backward_tolerance": 15.0},
}
