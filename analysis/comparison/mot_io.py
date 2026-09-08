from __future__ import annotations

from pathlib import Path

import pandas as pd

MOT_COLUMNS = [
    "frame",
    "id",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "class",
    "visibility",
]


def load_mot_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = MOT_COLUMNS
    return df


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x, y, x + w, y + h


def infer_yolo_only_path(tracked_pred_path: Path) -> Path:
    stem = tracked_pred_path.stem
    new_stem = stem.replace("-pred", "-yoloonly") if "-pred" in stem else f"{stem}-yoloonly"
    return tracked_pred_path.with_name(f"{new_stem}.txt")
