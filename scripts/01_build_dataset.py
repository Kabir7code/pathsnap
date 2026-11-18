"""Build the candidate feature tables for the train and student_test splits.

Outputs:
  data/features/features_<split>.parquet   one row per (GPS point, candidate arc)
  results/01_eda_preprocessing/labelling_<split>.csv        per-trip label coverage & candidate miss rate
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import CFG
from pathsnap.data import list_trips, load_trip
from pathsnap.features import build_trip_frame
from pathsnap.labels import align_track_to_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    os.makedirs("data/features", exist_ok=True); os.makedirs("results/01_eda_preprocessing", exist_ok=True)
    for split in args.splits:
        frames, stats, pts = [], [], []
        for tid in list_trips(os.path.join(CFG["data_dir"], split)):
            t0 = time.time()
            trip = load_trip(os.path.join(CFG["data_dir"], split), tid)
            true_arc, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
            df, cands, _, _ = build_trip_frame(trip, CFG["radius"], CFG["k"], CFG["window"], true_arc)
            labelled = true_arc >= 0
            in_set = np.array([true_arc[i] in cands[i] for i in range(len(cands))])
            stats.append(dict(trip=tid, points=len(true_arc), labelled_frac=labelled.mean(),
                              no_candidates_frac=np.mean([len(c) == 0 for c in cands]),
                              candidate_miss_rate=1 - in_set[labelled].mean(),
                              rows=len(df), seconds=round(time.time() - t0, 2)))
            frames.append(df)
            pts.append(pd.DataFrame({"trip": tid, "point": np.arange(len(true_arc)), "true_arc": true_arc,
                                     "n_cands": [len(c) for c in cands]}))
            print(split, stats[-1])
        pd.concat(frames, ignore_index=True).to_parquet(f"data/features/features_{split}.parquet")
        pd.concat(pts, ignore_index=True).to_parquet(f"data/features/points_{split}.parquet")
        pd.DataFrame(stats).to_csv(f"results/01_eda_preprocessing/labelling_{split}.csv", index=False)


if __name__ == "__main__":
    main()
