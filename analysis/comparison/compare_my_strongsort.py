"""Bandingkan metrik my_strongsort vs boxmot StrongSort (baseline_metrics.csv / tuned_metrics.csv).

Menjalankan evaluasi (basic + motmetrics) atas config my_strongsort_*.json lewat
pipeline yang sama dengan run_v5_vs_v8_strongsort.py, lalu join per `experiment`
dengan outputs/reports/{baseline,tuned}_metrics.csv yang sudah ada.

Usage:
  venv/bin/python -m analysis.comparison.compare_my_strongsort
"""
from __future__ import annotations

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

REPORTS = PROJECT_ROOT / "outputs" / "reports"
KEY_METRICS = ["mota", "idf1", "idp", "idr", "precision", "recall",
               "num_switches", "num_false_positives", "num_misses", "num_fragmentations"]

VARIANTS = [
    # (label, boxmot_metrics_csv, my_strongsort_config)
    ("baseline", REPORTS / "baseline_metrics.csv", PROJECT_ROOT / "configs/experiments/my_strongsort_baseline.json"),
    ("tuned", REPORTS / "tuned_metrics.csv", PROJECT_ROOT / "configs/experiments/my_strongsort_tuned.json"),
]


def evaluate_config(config_path: Path) -> pd.DataFrame:
    thresholds, experiments = load_experiment_config(config_path)
    rows = []
    for exp in experiments:
        if not exp.gt.exists() or not exp.tracked_pred.exists():
            print(f"SKIP: missing gt/pred for {exp.name} ({exp.tracked_pred})")
            continue
        gt_min, gt_max, pred_min, pred_max = frame_range_check(exp.gt, exp.tracked_pred)
        basic_eval = eval_result_to_dict(
            evaluate_prediction_file(exp.gt, exp.tracked_pred, thresholds.iou_threshold)
        )
        mot_eval = evaluate_mot(exp.gt, exp.tracked_pred, thresholds.iou_threshold, name=exp.name)
        rows.append({
            "experiment": exp.name,
            "model_family": exp.model_family,
            "pred_path": str(exp.tracked_pred),
            "gt_max_frame": gt_max,
            "pred_max_frame": pred_max,
            **{f"basic_{k}": v for k, v in basic_eval.items()},
            **mot_eval,
        })
    return pd.DataFrame(rows)


def main() -> None:
    for label, boxmot_csv, my_config in VARIANTS:
        if not boxmot_csv.exists():
            print(f"SKIP {label}: {boxmot_csv} not found")
            continue
        if not my_config.exists():
            print(f"SKIP {label}: {my_config} not found")
            continue

        print("=" * 72)
        print(f"MY_STRONGSORT vs BOXMOT StrongSort — {label}")
        print("=" * 72)

        boxmot_df = pd.read_csv(boxmot_csv)[["experiment", "model_family", *KEY_METRICS]]
        mine_df = evaluate_config(my_config)
        if mine_df.empty:
            print(f"SKIP {label}: no my_strongsort predictions found yet")
            continue
        mine_df = mine_df[["experiment", "model_family", *KEY_METRICS]]

        merged = boxmot_df.merge(mine_df, on="experiment", suffixes=("_boxmot", "_my_strongsort"))
        for metric in KEY_METRICS:
            merged[f"delta_{metric}"] = merged[f"{metric}_my_strongsort"] - merged[f"{metric}_boxmot"]

        out_path = REPORTS / f"my_strongsort_vs_{label}_comparison.csv"
        merged.to_csv(out_path, index=False)
        print(f"Saved: {out_path}\n")
        print(merged[["experiment", "mota_boxmot", "mota_my_strongsort", "delta_mota",
                      "idf1_boxmot", "idf1_my_strongsort", "delta_idf1"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
