"""Generate pred.txt for every experiment in a my_strongsort_*.json config.

Reuses ExperimentConfig (gt/tracked_pred/video/yolo_model) + demo.generate_pred
(MyStrongSort.run_video) — no new tracking logic here, just orchestration.

Usage:
  venv/bin/python -m analysis.comparison.generate_my_strongsort_preds \
      --config configs/experiments/my_strongsort_tuned.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.comparison.config import load_experiment_config
from my_strongsort.demo import generate_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="my_strongsort_*.json path")
    ap.add_argument("--force", action="store_true", help="Regenerate even if tracked_pred already exists")
    args = ap.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    thresholds, experiments = load_experiment_config(config_path)

    for exp in experiments:
        if exp.tracked_pred.exists() and not args.force:
            print(f"SKIP (exists): {exp.name} -> {exp.tracked_pred}")
            continue
        if not exp.video.exists() or not exp.yolo_model.exists():
            print(f"SKIP (missing video/model): {exp.name}")
            continue
        exp.tracked_pred.parent.mkdir(parents=True, exist_ok=True)
        print(f">>> {exp.name} ({exp.model_family}): {exp.video.name} -> {exp.tracked_pred}")
        generate_pred(
            str(exp.video), str(exp.yolo_model), str(exp.tracked_pred),
            imgsz=thresholds.imgsz, conf=thresholds.conf_threshold, iou=thresholds.iou_threshold,
        )


if __name__ == "__main__":
    main()
