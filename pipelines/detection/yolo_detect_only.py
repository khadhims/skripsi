import argparse
import logging
import os
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO detection-only inference and annotated video export"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLO model weights (.pt) or model name (e.g. yolov8n.pt).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video source path (file). Use 0 for webcam if needed.",
    )
    parser.add_argument(
        "--output",
        default="video-results/detection-only/output.mp4",
        help="Output video path for annotated results.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Optional class filter, e.g. --classes 0 2 3.",
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if args.model != "yolov8n.pt" and not os.path.exists(args.model):
        logging.error("Model file not found: %s", args.model)
        return

    logging.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    logging.info("Opening source: %s", args.source)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Could not open video source: %s", args.source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_parent_dir(args.output)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_id = 0
    logging.info("Starting detection-only inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        start_time = time.time()

        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            verbose=False,
        )

        annotated_frame = frame
        for result in results:
            annotated_frame = result.plot()

        inference_ms = (time.time() - start_time) * 1000.0
        cv2.putText(
            annotated_frame,
            f"Inference: {inference_ms:.2f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated_frame)

        if frame_id % 50 == 0:
            logging.info("Processed %d frames", frame_id)

    cap.release()
    writer.release()
    logging.info("Done. Saved annotated video to: %s", args.output)


if __name__ == "__main__":
    main()
