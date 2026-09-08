from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert annotation CSV to MOT ground-truth format")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output gt.txt path")
    parser.add_argument("--video-width", type=int, required=True, help="Video width in pixels")
    parser.add_argument("--video-height", type=int, required=True, help="Video height in pixels")
    parser.add_argument(
        "--track-id-col",
        default="object_id",
        help="Column name for object identifier",
    )
    return parser.parse_args()


def convert_csv_to_mot(
    csv_path: Path,
    output_path: Path,
    video_width: int,
    video_height: int,
    track_id_col: str,
) -> None:
    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "frame": "frame",
            "x": "x",
            "y": "y",
            "width": "w",
            "height": "h",
            track_id_col: "track_id",
        }
    )

    df["track_id"] = df["track_id"].astype("category").cat.codes + 1

    df["x"] = (df["x"] / 100 * video_width).astype(int)
    df["y"] = (df["y"] / 100 * video_height).astype(int)
    df["w"] = (df["w"] / 100 * video_width).astype(int)
    df["h"] = (df["h"] / 100 * video_height).astype(int)

    df["conf"] = 1
    df["class"] = 1
    df["visibility"] = 1

    df = df[["frame", "track_id", "x", "y", "w", "h", "conf", "class", "visibility"]]
    df = df.sort_values(["frame", "track_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=False)


def main() -> None:
    args = parse_args()
    convert_csv_to_mot(
        csv_path=Path(args.csv),
        output_path=Path(args.output),
        video_width=args.video_width,
        video_height=args.video_height,
        track_id_col=args.track_id_col,
    )
    print("MOT GT file generated successfully")


if __name__ == "__main__":
    main()
