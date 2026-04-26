from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class ThresholdConfig:
    iou_threshold: float = 0.7
    conf_threshold: float = 0.25
    imgsz: int = 1088


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_family: str
    gt: Path
    tracked_pred: Path
    video: Path
    yolo_model: Path
    yolo_only_pred: Path | None = None


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_experiment_config(config_path: Path) -> tuple[ThresholdConfig, list[ExperimentConfig]]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    settings = raw.get("settings", {})
    thresholds = ThresholdConfig(
        iou_threshold=float(settings.get("iou_threshold", 0.7)),
        conf_threshold=float(settings.get("conf_threshold", 0.25)),
        imgsz=int(settings.get("imgsz", 1088)),
    )

    base_dir = config_path.parent
    experiments: list[ExperimentConfig] = []

    for exp in raw.get("experiments", []):
        yolo_only_raw = exp.get("yolo_only_pred")
        yolo_only_pred = _resolve(base_dir, yolo_only_raw) if yolo_only_raw else None

        experiments.append(
            ExperimentConfig(
                name=str(exp["name"]),
                model_family=str(exp.get("model_family", "unknown")),
                gt=_resolve(base_dir, str(exp["gt"])),
                tracked_pred=_resolve(base_dir, str(exp["tracked_pred"])),
                video=_resolve(base_dir, str(exp["video"])),
                yolo_model=_resolve(base_dir, str(exp["yolo_model"])),
                yolo_only_pred=yolo_only_pred,
            )
        )

    return thresholds, experiments
