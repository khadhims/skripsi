from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from video by interval")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Output directory for PNG frames")
    parser.add_argument("--interval-sec", type=float, default=None, help="Extract every N seconds")
    parser.add_argument("--interval-frames", type=int, default=None, help="Extract every N frames")
    return parser.parse_args()


def extract_frames(
    video_path: Path,
    output_folder: Path,
    interval_sec: float | None,
    interval_frames: int | None,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frame_jump = 1
    if interval_frames is not None and interval_frames > 0:
        frame_jump = int(interval_frames)
    elif interval_sec is not None and interval_sec > 0 and fps > 0:
        frame_jump = max(1, int(interval_sec * fps))

    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    print(f"Extracting every {frame_jump} frame(s)")

    current_frame = 0
    saved_count = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        output_name = f"frame_{saved_count}.png"
        output_path = output_folder / output_name
        cv2.imwrite(str(output_path), frame)

        saved_count += 1
        current_frame += frame_jump

    cap.release()
    print(f"Done. Extracted {saved_count} frame(s) to {output_folder}")


def main() -> None:
    args = parse_args()
    extract_frames(
        video_path=Path(args.video),
        output_folder=Path(args.output_dir),
        interval_sec=args.interval_sec,
        interval_frames=args.interval_frames,
    )


if __name__ == "__main__":
    main()
