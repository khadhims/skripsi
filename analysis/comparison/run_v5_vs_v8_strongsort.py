from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.comparison.config import load_experiment_config
from analysis.comparison.metrics import evaluate_prediction_file
from analysis.comparison.report import eval_result_to_dict
from evaluation.mot.evaluator import evaluate_mot, frame_range_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLOv5-StrongSORT vs YOLOv8-StrongSORT using tracked predictions"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/v5_vs_v8_strongsort.json",
        help="JSON configuration path",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/v5_vs_v8_strongsort_metrics.csv",
        help="Output CSV detail path",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/reports/v5_vs_v8_strongsort_summary.csv",
        help="Output CSV summary path",
    )
    parser.add_argument(
        "--strict-frame-range",
        action="store_true",
        help="Fail when pred max frame differs from GT max frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()

    thresholds, experiments = load_experiment_config(config_path)
    rows: list[dict[str, object]] = []

    print("=" * 72)
    print("YOLOV5-STRONGSORT VS YOLOV8-STRONGSORT")
    print(f"Config : {config_path}")
    print(f"IoU    : {thresholds.iou_threshold}")
    print("=" * 72)

    for exp in experiments:
        if not exp.gt.exists() or not exp.tracked_pred.exists():
            print(f"SKIP: missing file for {exp.name}")
            continue

        gt_min, gt_max, pred_min, pred_max = frame_range_check(exp.gt, exp.tracked_pred)
        if args.strict_frame_range and pred_max != gt_max:
            print(f"SKIP: frame range mismatch for {exp.name} ({pred_max} != {gt_max})")
            continue

        basic_eval = eval_result_to_dict(
            evaluate_prediction_file(exp.gt, exp.tracked_pred, thresholds.iou_threshold)
        )
        mot_eval = evaluate_mot(exp.gt, exp.tracked_pred, thresholds.iou_threshold, name=exp.name)
        basic_eval_prefixed = {f"basic_{key}": value for key, value in basic_eval.items()}

        row = {
            "experiment": exp.name,
            "model_family": exp.model_family,
            "gt_path": str(exp.gt),
            "pred_path": str(exp.tracked_pred),
            "gt_min_frame": gt_min,
            "gt_max_frame": gt_max,
            "pred_min_frame": pred_min,
            "pred_max_frame": pred_max,
            **basic_eval_prefixed,
            **mot_eval,
        }
        rows.append(row)

        print(
            f"{exp.name}: "
            f"MOTA={mot_eval['mota']:.4f}, IDF1={mot_eval['idf1']:.4f}, "
            f"Precision={basic_eval['precision']:.4f}, Recall={basic_eval['recall']:.4f}"
        )

    if not rows:
        print("No comparable rows were produced.")
        return

    details_df = pd.DataFrame(rows)

    metric_cols = [
        "mota",
        "idf1",
        "basic_precision",
        "basic_recall",
        "basic_avg_frame_count_error",
        "basic_total_tp",
        "basic_total_fp",
        "basic_total_fn",
        "precision",
        "recall",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_fragmentations",
    ]

    summary_df = (
        details_df.groupby("model_family", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("model_family")
    )

    detail_path = (PROJECT_ROOT / args.output).resolve()
    summary_path = (PROJECT_ROOT / args.summary_output).resolve()

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved detail report : {detail_path}")
    print(f"Saved summary report: {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
