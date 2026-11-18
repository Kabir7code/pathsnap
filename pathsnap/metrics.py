"""Evaluation metrics.

Candidate level (binary: is this candidate the reference arc?):
  macro F1 (threshold 0.5), precision, recall, log loss, confusion matrix.
Observation level:
  Top-1 accuracy = share of labelled GPS points whose highest-scoring
  candidate equals the reference arc.  Points whose reference arc is not
  among the candidates (candidate miss) count as wrong.
Sequence level:
  route continuity = share of consecutive point pairs whose predicted arcs
  are identical or directly connected (end node of first == start node of
  second).  1.0 means the predicted path never 'jumps'.
  route precision/recall/F1 = overlap of the predicted set of traversed arcs
  with the reference route's arcs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def candidate_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yh, average="macro"),
        "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yh, labels=[0, 1]).tolist(),
    }


def top1_predictions(df, score_col="score"):
    """Per (trip, point): arc with highest score."""
    i = df.groupby(["trip", "point"])[score_col].idxmax()
    return df.loc[i, ["trip", "point", "arc"]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(points, pred):
    """points: trip, point, true_arc (-1 unlabelled). pred: trip, point, pred_arc."""
    m = points.merge(pred, on=["trip", "point"], how="left")
    m = m[m.true_arc >= 0]
    return float((m.pred_arc.fillna(-2).to_numpy() == m.true_arc.to_numpy()).mean())


def continuity(arcs_seq, arcs_table):
    s = np.asarray([a for a in arcs_seq if a >= 0])
    if len(s) < 2:
        return 1.0
    a, b = s[:-1], s[1:]
    same = a == b
    conn = arcs_table[a, 1] == arcs_table[b, 0]
    return float((same | conn).mean())


def route_overlap(pred_arcs, true_route):
    P, T = set(int(x) for x in pred_arcs if x >= 0), set(int(x) for x in true_route)
    inter = len(P & T)
    prec = inter / len(P) if P else 0.0
    rec = inter / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
