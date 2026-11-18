"""Concurrent model training: all classifiers trained in parallel worker processes.

Each model runs in its own process (ProcessPoolExecutor) with one thread, so
up to --workers models train at the same time on separate CPU cores.  The
protocol is identical to scripts/02_train_models.py (5-fold GroupKFold CV
on train trips + refit and evaluation on the student-test trips), so the
metrics are directly comparable.

Outputs (results/03_concurrent_training/):
  model_runs.json        per-model CV folds + student-test metrics + timings
  model_comparison.csv   summary table
  timeline.csv / timeline.png   start/end of every model (shows the overlap)
  run_summary.json       wall-clock time vs summed per-model time (speed-up)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "03_concurrent_training")
ALL = ["logreg", "gnb", "knn", "dtree", "rf", "extratrees", "histgb", "lightgbm", "mlp"]
MAX_ROWS = {"knn": 100_000}


def run_one(name, max_rows, t_origin):
    os.environ["OMP_NUM_THREADS"] = "1"
    from pathsnap.config import CFG
    from pathsnap.dataset import load_split
    from pathsnap.experiment import cross_validate, fit_and_test
    from pathsnap.models import MODEL_INFO, make_model

    start = time.time()
    pid = os.getpid()
    seed = CFG["seed"]
    tr, tr_pts = load_split("train")
    te, te_pts = load_split("student_test", exclude_bad=False)
    mr = MAX_ROWS.get(name, max_rows)
    make = lambda: make_model(name, seed, n_jobs=1)
    folds = cross_validate(make, tr, tr_pts, folds=5, max_rows=mr, seed=seed, verbose=False)
    model, test = fit_and_test(make, tr, te, te_pts, max_rows=mr, seed=seed)
    end = time.time()
    return name, {"family": MODEL_INFO[name][0], "config": MODEL_INFO[name][1], "seed": seed,
                  "train_rows": test.pop("train_rows"), "cv_folds": folds, "student_test": test,
                  "pid": pid, "start_s": start - t_origin, "end_s": end - t_origin, "duration_s": end - start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max_rows", type=int, default=300_000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    runs = {}
    print(f"training {len(args.models)} models concurrently on {args.workers} worker processes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, m, args.max_rows, t0): m for m in args.models}
        for f in as_completed(futs):
            name, r = f.result()
            runs[name] = r
            t = r["student_test"]
            print(f"[{time.time() - t0:7.1f}s] done {name:10s} pid={r['pid']} took {r['duration_s']:.1f}s | "
                  f"test top1={t['top1']:.4f} F1={t['macro_f1']:.4f} ll={t['log_loss']:.4f}", flush=True)
            json.dump(runs, open(f"{OUT}/model_runs.json", "w"), indent=1)
    wall = time.time() - t0
    summed = sum(r["duration_s"] for r in runs.values())

    rows = []
    for name, r in runs.items():
        cv = pd.DataFrame([{k: v for k, v in f.items() if k != "confusion"} for f in r["cv_folds"]])
        t = r["student_test"]
        rows.append({"model": name, "family": r["family"], "config": r["config"],
                     "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
                     "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
                     "test_top1": t["top1"], "test_macro_f1": t["macro_f1"], "test_precision": t["precision"],
                     "test_recall": t["recall"], "test_log_loss": t["log_loss"], "test_continuity": t["continuity"],
                     "fit_s": t["fit_s"], "infer_ms_per_point": t["infer_ms_per_point"],
                     "worker_pid": r["pid"], "duration_s": r["duration_s"]})
    out = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    out.to_csv(f"{OUT}/model_comparison.csv", index=False)
    tl = pd.DataFrame([{"model": n, "pid": r["pid"], "start_s": r["start_s"], "end_s": r["end_s"]} for n, r in runs.items()])
    tl.sort_values("start_s").to_csv(f"{OUT}/timeline.csv", index=False)
    summary = {"models": len(runs), "workers": args.workers, "wall_clock_s": wall,
               "sum_of_model_times_s": summed, "speedup": summed / wall}
    json.dump(summary, open(f"{OUT}/run_summary.json", "w"), indent=1)

    tl = tl.sort_values("start_s").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    pids = {p: i for i, p in enumerate(sorted(tl.pid.unique()))}
    for i, r in tl.iterrows():
        ax.barh(i, r.end_s - r.start_s, left=r.start_s, color=f"C{pids[r.pid]}")
        ax.text(r.end_s + 3, i, f"worker {pids[r.pid]}", va="center", fontsize=8)
    ax.set_yticks(range(len(tl)), tl.model)
    ax.invert_yaxis()
    ax.set(xlabel="seconds since start", title=f"Concurrent training: {len(tl)} models on {args.workers} workers "
                                                f"(wall {wall:.0f}s vs {summed:.0f}s summed, {summed / wall:.1f}x)")
    fig.tight_layout(); fig.savefig(f"{OUT}/timeline.png", dpi=130)
    print(out.drop(columns=["config"]).round(4).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
