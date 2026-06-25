from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.mot.evaluator import evaluate_mot, frame_range_check, render_mot_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MOT metrics for one prediction file")
    parser.add_argument("--gt", required=True, help="Path to GT MOT file")
    parser.add_argument("--pred", required=True, help="Path to prediction MOT file")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU dist threshold")
    parser.add_argument("--name", default="Agnostic", help="Summary name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)

    gt_min, gt_max, pred_min, pred_max = frame_range_check(gt_path, pred_path)
    print("===== FRAME RANGE CHECK =====")
    print(f"GT   : {gt_min} -> {gt_max}")
    print(f"Pred : {pred_min} -> {pred_max}")

    if pred_max != gt_max:
        raise ValueError("Frame range mismatch: evaluation stopped.")

    print("\n===== MOT Evaluation =====\n")
    print(render_mot_summary(gt_path, pred_path, args.iou, name=args.name))

    key_metrics = evaluate_mot(gt_path, pred_path, args.iou, name=args.name)
    print("\n===== Key Metrics =====")
    print(f"MOTA : {key_metrics['mota']:.4f}")
    print(f"IDF1 : {key_metrics['idf1']:.4f}")


if __name__ == "__main__":
    main()
