from __future__ import annotations

from pathlib import Path
from typing import Any

import motmetrics as mm
import pandas as pd


def frame_range_check(gt_path: Path, pred_path: Path) -> tuple[int, int, int, int]:
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
    gt_df = pd.read_csv(gt_path, header=None, names=cols)
    pred_df = pd.read_csv(pred_path, header=None, names=cols)

    gt_min = int(gt_df["frame"].min())
    gt_max = int(gt_df["frame"].max())
    pred_min = int(pred_df["frame"].min())
    pred_max = int(pred_df["frame"].max())
    return gt_min, gt_max, pred_min, pred_max


def evaluate_mot(gt_path: Path, pred_path: Path, iou_threshold: float, name: str = "Agnostic") -> dict[str, Any]:
    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D")
    pred = mm.io.loadtxt(str(pred_path), fmt="mot15-2D")

    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=iou_threshold)
    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=name)

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
    }


def render_mot_summary(gt_path: Path, pred_path: Path, iou_threshold: float, name: str = "Agnostic") -> str:
    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D")
    pred = mm.io.loadtxt(str(pred_path), fmt="mot15-2D")

    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=iou_threshold)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=name)

    return mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
