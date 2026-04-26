from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze inference log and summarize latency per model")
    parser.add_argument(
        "--log",
        default="inference_log.txt",
        help="Path to inference log file",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/inference_latency_summary.csv",
        help="CSV output path",
    )
    return parser.parse_args()


def calculate_average_inference(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    model_inference_times: dict[str, list[float]] = defaultdict(list)
    current_model = "unknown-model"

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            model_match = re.search(r"Loading model from: (.+)", line)
            if model_match:
                current_model = model_match.group(1).strip()
                continue

            time_match = re.search(r"Inference Time = (\d+\.\d+) ms", line)
            if time_match:
                model_inference_times[current_model].append(float(time_match.group(1)))

    rows: list[dict[str, object]] = []
    for model_name, values in model_inference_times.items():
        if not values:
            continue
        rows.append(
            {
                "model_name": model_name,
                "avg_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "frames": len(values),
            }
        )

    return pd.DataFrame(rows).sort_values("avg_ms") if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    output_path = Path(args.output)

    df = calculate_average_inference(log_path)

    if df.empty:
        print("No inference entries found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 100)
    print(f"{'Model Name':<60} | {'Avg Time (ms)':<15} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Frames':<6}")
    print("=" * 100)

    for _, row in df.iterrows():
        model_name = str(row["model_name"])
        display = model_name if len(model_name) <= 58 else "..." + model_name[-55:]
        print(
            f"{display:<60} | "
            f"{float(row['avg_ms']):<15.2f} | "
            f"{float(row['min_ms']):<10.2f} | "
            f"{float(row['max_ms']):<10.2f} | "
            f"{int(row['frames']):<6}"
        )

    print("=" * 100)
    print(f"Saved summary CSV: {output_path}")


if __name__ == "__main__":
    main()
