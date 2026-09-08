from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.comparison.config import load_experiment_config
from analysis.comparison.metrics import evaluate_prediction_file
from analysis.comparison.mot_io import infer_yolo_only_path
from analysis.comparison.report import (
    eval_result_to_dict,
    print_pairwise_table,
    summarize_by_model_family,
    write_csv,
)
from analysis.comparison.yolo_only import generate_yolo_only_prediction

try:
    from benchmarks.run_pipeline_benchmark import run_benchmark as _run_benchmark
except ImportError:
    _run_benchmark = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tracking vs YOLO-only per experiment")
    parser.add_argument(
        "--config",
        default="configs/experiments/v5_vs_v8_strongsort.json",
        help="JSON configuration path",
    )
    parser.add_argument(
        "--skip-generate-yolo-only",
        action="store_true",
        help="Skip generating YOLO-only file when missing",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/tracking_vs_yolo_metrics.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/reports/tracking_vs_yolo_summary.csv",
        help="Output CSV for averaged summary",
    )
    parser.add_argument(
        "--speed-frames",
        type=int,
        default=0,
        help="Measure tracking FPS/inference speed using N frames (0 to skip)",
    )
    parser.add_argument(
        "--speed-all",
        action="store_true",
        help="Measure tracking FPS/inference speed using the entire video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()

    thresholds, experiments = load_experiment_config(config_path)
    rows: list[dict[str, object]] = []

    print("=" * 72)
    print("TRACKING VS YOLO-ONLY")
    print(f"Config   : {config_path}")
    print(f"IoU      : {thresholds.iou_threshold}")
    print(f"Conf     : {thresholds.conf_threshold}")
    print(f"Image sz : {thresholds.imgsz}")
    print("=" * 72)

    for exp in experiments:
        print(f"\n>>> Processing: {exp.name} ({exp.model_family})")

        if not exp.gt.exists():
            print(f"SKIP: GT not found: {exp.gt}")
            continue
        if not exp.tracked_pred.exists():
            print(f"SKIP: tracked prediction not found: {exp.tracked_pred}")
            continue

        yolo_only_path = exp.yolo_only_pred or infer_yolo_only_path(exp.tracked_pred)

        if not yolo_only_path.exists():
            if args.skip_generate_yolo_only:
                print(f"SKIP: missing YOLO-only prediction and generation disabled: {yolo_only_path}")
                continue
            if not exp.video.exists():
                print(f"SKIP: video not found: {exp.video}")
                continue
            if not exp.yolo_model.exists():
                print(f"SKIP: YOLO model not found: {exp.yolo_model}")
                continue

            print(f"Generating YOLO-only prediction: {yolo_only_path}")
            generate_yolo_only_prediction(
                video_path=exp.video,
                model_path=exp.yolo_model,
                output_path=yolo_only_path,
                conf_threshold=thresholds.conf_threshold,
                imgsz=thresholds.imgsz,
                iou_threshold=thresholds.iou_threshold,
            )

        speed_metrics: dict[str, float] | None = None
        do_speed = args.speed_frames > 0 or args.speed_all
        if do_speed:
            if _run_benchmark is None:
                print("SKIP speed: benchmarks.run_pipeline_benchmark not available")
            elif not exp.video.exists():
                print(f"SKIP speed: video not found for {exp.name}")
            elif not exp.yolo_model.exists():
                print(f"SKIP speed: YOLO model not found for {exp.name}")
            else:
                frames = args.speed_frames if args.speed_frames > 0 else 0
                speed_metrics = _run_benchmark(
                    str(exp.video),
                    frames,
                    model_path=str(exp.yolo_model),
                    conf_threshold=thresholds.conf_threshold,
                    imgsz=thresholds.imgsz,
                    iou_threshold=thresholds.iou_threshold,
                    print_summary=False,
                )

        avg_fps = None
        avg_inference_ms = None
        if speed_metrics and speed_metrics.get("total_frames", 0.0) > 0:
            avg_fps = speed_metrics.get("avg_fps")
            avg_inference_ms = speed_metrics.get("avg_frame_time_ms")

        tracked_eval = eval_result_to_dict(
            evaluate_prediction_file(exp.gt, exp.tracked_pred, thresholds.iou_threshold)
        )
        yolo_eval = eval_result_to_dict(
            evaluate_prediction_file(exp.gt, yolo_only_path, thresholds.iou_threshold)
        )

        print_pairwise_table(exp.name, tracked_eval, yolo_eval)

        rows.append(
            {
                "experiment": exp.name,
                "model_family": exp.model_family,
                "method": "tracking",
                "avg_fps": avg_fps,
                "avg_inference_ms": avg_inference_ms,
                **tracked_eval,
                "gt_path": str(exp.gt),
                "pred_path": str(exp.tracked_pred),
            }
        )
        rows.append(
            {
                "experiment": exp.name,
                "model_family": exp.model_family,
                "method": "yolo_only",
                "avg_fps": None,
                "avg_inference_ms": None,
                **yolo_eval,
                "gt_path": str(exp.gt),
                "pred_path": str(yolo_only_path),
            }
        )

    if not rows:
        print("No rows produced.")
        return

    output_path = (PROJECT_ROOT / args.output).resolve()
    summary_output_path = (PROJECT_ROOT / args.summary_output).resolve()

    write_csv(rows, output_path)
    print(f"\nSaved detail report: {output_path}")

    summary_df = summarize_by_model_family(rows)
    if not summary_df.empty:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_output_path, index=False)
        print(f"Saved summary report: {summary_output_path}")
        print("\nSummary by model family and method:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
