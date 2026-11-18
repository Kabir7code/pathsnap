"""Helpers to load the feature tables produced by scripts/01_build_dataset.py."""
import os

import numpy as np
import pandas as pd

from .config import CFG, ROOT

ART = os.path.join(ROOT, "artifacts")


def load_split(split, exclude_bad=True):
    df = pd.read_parquet(os.path.join(ART, f"features_{split}.parquet"))
    pts = pd.read_parquet(os.path.join(ART, f"points_{split}.parquet"))
    if exclude_bad:
        lab = pd.read_csv(os.path.join(ROOT, "results", f"labelling_{split}.csv"), dtype={"trip": str})
        good = set(lab.loc[lab.labelled_frac >= CFG["min_trip_label_frac"], "trip"])
        df, pts = df[df.trip.isin(good)], pts[pts.trip.isin(good)]
    # only labelled points are used for supervised training / scoring
    df = df[df.true_arc >= 0].reset_index(drop=True)
    return df, pts.reset_index(drop=True)


def subsample_points(df, max_rows, seed):
    """Sample whole GPS points (all their candidates) up to ~max_rows rows."""
    if len(df) <= max_rows:
        return df
    keys = df[["trip", "point"]].drop_duplicates()
    frac = max_rows / len(df)
    keep = keys.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["trip", "point"])
