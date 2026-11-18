"""Exploratory data analysis of the raw dataset (train + student_test).

Outputs (results/eda/):
  inventory.csv          per-trip file sizes, row counts, time span, extent
  network.csv            per-trip graph size, connectivity, isolated nodes
  summary.txt            partition-level summary + data-quality notes
  *.png                  plots (sampling intervals, track lengths, coordinates,
                         GPS-to-road distance, network sizes, example trip)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.candidates import ArcIndex
from pathsnap.config import CFG
from pathsnap.data import list_trips, load_trip

OUT = "results/eda"
os.makedirs(OUT, exist_ok=True)


def raw_checks(split_dir, tid):
    """Checks on the *raw* track file before load_trip's cleaning."""
    raw = pd.read_csv(os.path.join(split_dir, tid, f"{tid}.track"), sep=r"\s+", header=None)
    t = raw[2].to_numpy()
    dt = np.diff(t)
    lon, lat = raw[0], raw[1]
    return {
        "track_rows_raw": len(raw),
        "track_nan": int(raw.isna().sum().sum()),
        "non_monotonic_ts": int((dt <= 0).sum()),
        "coords_out_of_range": int(((lon.abs() > 180) | (lat.abs() > 90)).sum()),
    }


def main():
    inv, net, all_dt, dist_samples = [], [], {}, {}
    for split in ["train", "student_test"]:
        sd = os.path.join(CFG["data_dir"], split)
        all_dt[split] = []
        dist_samples[split] = []
        for tid in list_trips(sd):
            files = {e: os.path.join(sd, tid, f"{tid}.{e}") for e in ["nodes", "arcs", "track", "route"]}
            row = {"split": split, "trip": tid}
            for e, p in files.items():
                row[f"has_{e}"] = os.path.exists(p)
                row[f"{e}_kb"] = round(os.path.getsize(p) / 1024, 1) if os.path.exists(p) else np.nan
            row.update(raw_checks(sd, tid))
            trip = load_trip(sd, tid)
            dt = np.diff(trip.times)
            all_dt[split].append(dt)
            step = np.linalg.norm(np.diff(trip.track_xy, axis=0), axis=1)
            ext = trip.track_xy.max(0) - trip.track_xy.min(0)
            row.update({
                "points": len(trip.times), "duration_s": trip.times[-1] - trip.times[0],
                "dt_median": np.median(dt), "dt_max": dt.max(), "frac_dt_1s": np.mean(dt == 1),
                "gaps_over_10s": int((dt > 10).sum()), "path_km": step.sum() / 1000,
                "max_speed_ms": (step / np.maximum(dt, 1e-6)).max(),
                "extent_x_km": ext[0] / 1000, "extent_y_km": ext[1] / 1000,
                "lon_min": trip.track_ll[:, 0].min(), "lon_max": trip.track_ll[:, 0].max(),
                "lat_min": trip.track_ll[:, 1].min(), "lat_max": trip.track_ll[:, 1].max(),
                "route_arcs": len(trip.route), "route_km": trip.arc_len[trip.route].sum() / 1000,
                "route_disconnected_steps": int((trip.arcs[trip.route[:-1], 1] != trip.arcs[trip.route[1:], 0]).sum()),
                "route_repeated_arcs": len(trip.route) - len(np.unique(trip.route)),
            })
            inv.append(row)

            N, A = len(trip.nodes_xy), trip.arcs
            g = coo_matrix((np.ones(len(A)), (A[:, 0], A[:, 1])), shape=(N, N))
            nw, lw = connected_components(g, directed=True, connection="weak")
            ns, ls = connected_components(g, directed=True, connection="strong")
            deg = np.bincount(A.ravel(), minlength=N)
            key = set(map(tuple, A))
            net.append({"split": split, "trip": tid, "nodes": N, "arcs": len(A),
                        "isolated_nodes": int((deg == 0).sum()),
                        "weak_components": nw, "largest_weak_frac": np.bincount(lw).max() / N,
                        "strong_components": ns, "largest_strong_frac": np.bincount(ls).max() / N,
                        "two_way_frac": np.mean([(b, a) in key for a, b in A]),
                        "self_loops": int((A[:, 0] == A[:, 1]).sum()),
                        "zero_len_arcs": int((trip.arc_len < 0.01).sum()),
                        "mean_arc_len_m": trip.arc_len.mean()})
            # GPS -> nearest arc distance (sample of points)
            idx = ArcIndex(trip)
            c, d, _ = idx.query(trip.track_xy[::10], radius=300, k=1)
            dist_samples[split].extend([x[0] if len(x) else 300 for x in d])
            print(split, tid, flush=True)

    inv, net = pd.DataFrame(inv), pd.DataFrame(net)
    inv.to_csv(f"{OUT}/inventory.csv", index=False)
    net.to_csv(f"{OUT}/network.csv", index=False)

    with open(f"{OUT}/summary.txt", "w") as f:
        for split, g in inv.groupby("split"):
            n = net[net.split == split]
            f.write(f"== {split}: {len(g)} trips\n")
            f.write(g[["points", "duration_s", "path_km", "dt_max", "frac_dt_1s", "gaps_over_10s", "route_arcs", "route_km"]]
                    .describe().round(2).to_string() + "\n")
            f.write(n[["nodes", "arcs", "isolated_nodes", "weak_components", "strong_components", "largest_strong_frac", "two_way_frac"]]
                    .describe().round(3).to_string() + "\n")
            f.write(f"missing files: {int((~g[[c for c in g if c.startswith('has_')]]).sum().sum())}\n")
            f.write(f"NaN values in tracks: {g.track_nan.sum()}, non-monotonic timestamps: {g.non_monotonic_ts.sum()}, "
                    f"out-of-range coords: {g.coords_out_of_range.sum()}\n")
            f.write(f"trips with disconnected reference-route steps: {(g.route_disconnected_steps > 0).sum()}\n")
            ds = np.array(dist_samples[split])
            f.write(f"GPS->nearest arc distance (m) percentiles 50/90/99: {np.percentile(ds, [50, 90, 99]).round(1)}; "
                    f"share > 50 m: {np.mean(ds > 50):.4f}\n\n")
    print(open(f"{OUT}/summary.txt").read())

    # ---- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for split in all_dt:
        dt = np.concatenate(all_dt[split])
        ax[0].hist(np.clip(dt, 0, 20), bins=40, alpha=.6, label=split, density=True)
    ax[0].set(yscale="log", xlabel="sampling interval (s, clipped at 20)", ylabel="density", title="GPS sampling intervals")
    ax[0].legend()
    for split, g in inv.groupby("split"):
        ax[1].hist(g.points, bins=20, alpha=.6, label=split)
    ax[1].set(xlabel="GPS points per trip", ylabel="trips", title="Track length")
    ax[1].legend(); fig.tight_layout(); fig.savefig(f"{OUT}/sampling_and_length.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for split in dist_samples:
        ax.hist(np.clip(dist_samples[split], 0, 100), bins=50, alpha=.6, label=split, density=True)
    ax.set(xlabel="distance GPS point -> nearest arc (m, clipped 100)", ylabel="density", title="GPS noise vs road network")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/gps_to_road_distance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for split, g in inv.groupby("split"):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=g.points / 100 + 5, alpha=.6, label=split)
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip centres (size ~ #points)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/trip_locations.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(net.nodes, net.arcs, c=(net.split == "train").map({True: "C0", False: "C1"}), s=15)
    ax.set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    fig.tight_layout(); fig.savefig(f"{OUT}/network_size.png", dpi=130); plt.close(fig)

    # example trip: network + GPS + route
    trip = load_trip(os.path.join(CFG["data_dir"], "student_test"), "00000000")
    fig, ax = plt.subplots(figsize=(7, 7))
    a, b = trip.arc_a, trip.arc_b
    ax.plot(np.stack([a[:, 0], b[:, 0]]), np.stack([a[:, 1], b[:, 1]]), color="#bbb", lw=.4)
    r = trip.route
    ax.plot(np.stack([a[r, 0], b[r, 0]]), np.stack([a[r, 1], b[r, 1]]), color="green", lw=2.5)
    ax.scatter(trip.track_xy[:, 0], trip.track_xy[:, 1], s=2, color="blue", zorder=3)
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="student_test 00000000: network (grey), route (green), GPS (blue)")
    fig.tight_layout(); fig.savefig(f"{OUT}/example_trip.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
