"""Interactive Folium maps: GPS, road network, reference route, predicted routes.

  python scripts/05_folium_maps.py --split student_test --trips 00000000 00000056
  python scripts/05_folium_maps.py --split train --trips 00000010 --eda   # EDA map, no model
Maps are written to results/06_folium_maps/<split>_<trip>.html
"""
import argparse
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.config import CFG
from pathsnap.data import list_trips, load_trip
from pathsnap.dataset import ART, MODELS
from pathsnap.labels import align_track_to_route
from pathsnap.pipeline import predict_trip, score_trip
from pathsnap.viz import trip_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--eda", action="store_true", help="only GPS + network + reference route")
    args = ap.parse_args()
    sd = os.path.join(CFG["data_dir"], args.split)
    trips = args.trips or list_trips(sd)
    os.makedirs("results/06_folium_maps", exist_ok=True)
    if not args.eda:
        model = joblib.load(os.path.join(MODELS, "final.joblib"))
        hp = json.load(open(os.path.join(MODELS, "hmm_params.json")))
    for tid in trips:
        trip = load_trip(sd, tid)
        layers = [("Reference route", trip.route, "#1a9e3a", 7, None)]
        title = f"{args.split} {tid}"
        if not args.eda:
            out = predict_trip(trip, model, hp)
            ta, _, _ = align_track_to_route(trip, CFG["label_max_dist"])
            s = score_trip(trip, out, ta)
            layers += [("Classifier-only route", out["cls_route"], "#f28e1c", 3, "6 6"),
                       ("Classifier + HMM route", out["route"], "#d62728", 3, None)]
            title += (f" | top-1 cls {s['cls_top1']:.3f} / HMM {s['hmm_top1']:.3f}"
                      f" | route F1 cls {s['cls_route_f1']:.3f} / HMM {s['hmm_route_f1']:.3f}")
        path = f"results/06_folium_maps/{args.split}_{tid}{'_eda' if args.eda else ''}.html"
        trip_map(trip, layers, title).save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
