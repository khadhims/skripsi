from __future__ import annotations

from pathlib import Path
from typing import Any

import motmetrics as mm
import numpy as np
import pandas as pd

# motmetrics 1.4.0 (latest on PyPI) still calls np.asfarray, removed in NumPy 2.0.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)


def frame_range_check(gt_path: Path, pred_path: Path) -> tuple[int, int, int, int]:
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
    gt_df = pd.read_csv(gt_path, header=None, names=cols)
    pred_df = pd.read_csv(pred_path, header=None, names=cols)

    gt_min = int(gt_df["frame"].min())
    gt_max = int(gt_df["frame"].max())
    pred_min = int(pred_df["frame"].min())
    pred_max = int(pred_df["frame"].max())
    return gt_min, gt_max, pred_min, pred_max


def evaluate_mot(gt_path: Path, pred_path: Path, iou_dist_threshold: float, name: str = "Agnostic", class_agnostic: bool = True) -> dict[str, Any]:
    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D")
    pred = mm.io.loadtxt(str(pred_path), fmt="mot15-2D")

    # For class-agnostic evaluation, set all ClassId to 1 (same class for all objects)
    if class_agnostic:
        gt = gt.copy()
        gt['ClassId'] = 1
        pred = pred.copy()
        pred['ClassId'] = 1

    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=iou_dist_threshold)
    mh = mm.metrics.create()

    # Compute both standard MOT Challenge metrics and additional detailed metrics
    metrics_to_compute = list(mm.metrics.motchallenge_metrics) + ['idtp', 'idfp', 'idfn', 'num_matches', 'num_objects', 'num_detections']
    summary = mh.compute(acc, metrics=metrics_to_compute, name=name)

    row = summary.loc[name]
    return {
        "mota": float(row["mota"]),
        "idf1": float(row["idf1"]),
        "idp": float(row["idp"]),
        "idr": float(row["idr"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "num_switches": int(row["num_switches"]),
        "num_false_positives": int(row["num_false_positives"]),
        "num_misses": int(row["num_misses"]),
        "num_fragmentations": int(row["num_fragmentations"]),
        "num_matches": int(row["num_matches"]),
        "num_objects": int(row["num_objects"]),
        "num_detections": int(row["num_detections"]),
        "idtp": int(row["idtp"]),
        "idfp": int(row["idfp"]),
        "idfn": int(row["idfn"]),
    }


def render_mot_summary(gt_path: Path, pred_path: Path, iou_dist_threshold: float, name: str = "Agnostic", class_agnostic: bool = True) -> str:
    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D")
    pred = mm.io.loadtxt(str(pred_path), fmt="mot15-2D")

    # For class-agnostic evaluation, set all ClassId to 1 (same class for all objects)
    if class_agnostic:
        gt = gt.copy()
        gt['ClassId'] = 1
        pred = pred.copy()
        pred['ClassId'] = 1

    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=iou_dist_threshold)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=name)

    return mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
