from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


def print_pairwise_table(name: str, tracked: dict[str, Any], yolo_only: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"PERBANDINGAN: {name}")
    print("=" * 72)
    print(f"{'Metric':<36}{'Tracking':>16}{'YOLO Only':>16}")
    print("-" * 72)

    rows = [
        ("Precision", tracked["precision"], yolo_only["precision"]),
        ("Recall", tracked["recall"], yolo_only["recall"]),
        (
            "Avg Frame Count Error",
            tracked["avg_frame_count_error"],
            yolo_only["avg_frame_count_error"],
        ),
        ("Total TP", tracked["total_tp"], yolo_only["total_tp"]),
        ("Total FP", tracked["total_fp"], yolo_only["total_fp"]),
        ("Total FN", tracked["total_fn"], yolo_only["total_fn"]),
    ]

    for label, left, right in rows:
        if isinstance(left, float) or isinstance(right, float):
            print(f"{label:<36}{left:>16.4f}{right:>16.4f}")
        else:
            print(f"{label:<36}{left:>16}{right:>16}")


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def summarize_by_model_family(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    metric_cols = [
        "avg_fps",
        "avg_inference_ms",
        "precision",
        "recall",
        "avg_frame_count_error",
        "total_tp",
        "total_fp",
        "total_fn",
    ]

    summary = (
        df.groupby(["model_family", "method"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["model_family", "method"])
    )
    return summary


def eval_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return dict(result)
