from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import ObjectDetection, build_boxmot_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO + StrongSORT pipeline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default=None, help="Path to YOLO weights (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    return parser.parse_args()


def run_benchmark(
    video_path: str,
    max_frames: int,
    model_path: str | None = None,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    iou_threshold: float = 0.3,
    print_summary: bool = True,
) -> dict[str, float]:
    print(f"Starting benchmark on {video_path}")
    print(f"Frames to process: {max_frames}")

    detector = ObjectDetection(
        capture_index=video_path,
        model_path=model_path,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        iou_threshold=iou_threshold,
    )
    cap = cv2.VideoCapture(video_path)
    # Jika max_frames <= 0, gunakan seluruh frame video jika metadata tersedia.
    # Jika metadata tidak tersedia, fallback ke loop hingga EOF (max_frames = inf).
    if max_frames <= 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            max_frames = total_frames
        else:
            max_frames = float("inf")
    tracker = detector.tracker

    cpu_usages: list[float] = []
    mem_usages: list[float] = []
    frame_times: list[float] = []

    if hasattr(tracker, "model") and hasattr(tracker.model, "warmup"):
        tracker.model.warmup()

    frame_count = 0
    process = psutil.Process()
    outputs = [None]

    start_total = time.perf_counter()

    while cap.isOpened() and frame_count < max_frames:
        start_frame = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        results = detector.predict(frame)
        for result in results:
            dets = build_boxmot_detections(result)
            outputs[0] = tracker.update(dets, frame)

        frame_time = time.perf_counter() - start_frame
        frame_times.append(frame_time)
        cpu_usages.append(process.cpu_percent())
        mem_usages.append(process.memory_info().rss / 1024 / 1024)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()

    end_total = time.perf_counter()
    if frame_count == 0:
        if print_summary:
            print("No frame was processed.")
        return {
            "avg_fps": 0.0,
            "avg_frame_time_ms": 0.0,
            "total_frames": 0.0,
            "total_time_s": 0.0,
            "avg_cpu": 0.0,
            "max_mem_mb": 0.0,
            "avg_mem_mb": 0.0,
        }

    avg_fps = frame_count / (end_total - start_total)
    avg_frame_time = float(np.mean(frame_times) * 1000.0)
    avg_cpu = float(np.mean(cpu_usages)) if cpu_usages else 0.0
    max_mem = float(np.max(mem_usages)) if mem_usages else 0.0
    avg_mem = float(np.mean(mem_usages)) if mem_usages else 0.0

    if print_summary:
        print("\n" + "=" * 44)
        print("PIPELINE BENCHMARK RESULTS")
        print("=" * 44)
        print(f"Total Frames   : {frame_count}")
        print(f"Total Time     : {end_total - start_total:.2f} s")
        print(f"Average FPS    : {avg_fps:.2f}")
        print(f"Avg Frame Time : {avg_frame_time:.2f} ms")
        print("-" * 44)
        print(f"Avg CPU Usage  : {avg_cpu:.2f}%")
        print(f"Max RAM Usage  : {max_mem:.2f} MB")
        print(f"Avg RAM Usage  : {avg_mem:.2f} MB")
        print("=" * 44)

    return {
        "avg_fps": float(avg_fps),
        "avg_frame_time_ms": float(avg_frame_time),
        "total_frames": float(frame_count),
        "total_time_s": float(end_total - start_total),
        "avg_cpu": float(avg_cpu),
        "max_mem_mb": float(max_mem),
        "avg_mem_mb": float(avg_mem),
    }


def main() -> None:
    args = parse_args()
    run_benchmark(
        args.video,
        args.max_frames,
        model_path=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        iou_threshold=args.iou,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
