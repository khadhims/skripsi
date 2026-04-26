from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mot_io import load_mot_file, xywh_to_xyxy


@dataclass
class EvalResult:
    precision: float
    recall: float
    avg_frame_count_error: float
    total_unique_count_error: int
    total_unique_gt: int
    total_unique_pred: int
    total_tp: int
    total_fp: int
    total_fn: int


def compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def match_frame_boxes(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    while True:
        max_iou = float(iou_matrix.max())
        if max_iou < iou_threshold:
            break
        gt_idx, pred_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matched_gt.add(int(gt_idx))
        matched_pred.add(int(pred_idx))
        iou_matrix[gt_idx, :] = 0.0
        iou_matrix[:, pred_idx] = 0.0

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def evaluate_prediction_file(gt_path: Path, pred_path: Path, iou_threshold: float) -> EvalResult:
    gt_df = load_mot_file(gt_path)
    pred_df = load_mot_file(pred_path)

    all_frames = sorted(gt_df["frame"].unique())

    total_tp = 0
    total_fp = 0
    total_fn = 0
    frame_count_errors: list[int] = []

    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        gt_boxes = [
            xywh_to_xyxy(float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            for _, row in gt_frame.iterrows()
        ]
        pred_boxes = [
            xywh_to_xyxy(float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            for _, row in pred_frame.iterrows()
        ]

        tp, fp, fn = match_frame_boxes(gt_boxes, pred_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        frame_count_errors.append(abs(len(pred_boxes) - len(gt_boxes)))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_count_error = float(np.mean(frame_count_errors)) if frame_count_errors else 0.0

    total_unique_gt = int(gt_df["id"].nunique())
    total_unique_pred = int(pred_df["id"].nunique())

    return EvalResult(
        precision=precision,
        recall=recall,
        avg_frame_count_error=avg_count_error,
        total_unique_count_error=abs(total_unique_pred - total_unique_gt),
        total_unique_gt=total_unique_gt,
        total_unique_pred=total_unique_pred,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
    )
