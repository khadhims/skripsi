from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick detection test and latency logging")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1088, help="Inference image size")
    parser.add_argument("--classes", nargs="*", type=int, default=None, help="Optional class list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    if not Path(args.model).exists() and args.model != "yolov8n.pt":
        logging.error("Model file not found: %s", args.model)
        return

    logging.info("Loading model from: %s", args.model)
    model = YOLO(args.model)

    logging.info("Opening video source: %s", args.video)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logging.error("Could not open video source")
        return

    frame_count = 0
    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of stream")
            break

        frame_count += 1
        start_time = time.time()

        results = model.predict(
            frame,
            verbose=False,
            conf=args.conf,
            stream=True,
            imgsz=args.imgsz,
            iou=args.iou,
            classes=args.classes,
        )

        annotated_frame = frame
        for result in results:
            annotated_frame = result.plot()

        inference_ms = (time.time() - start_time) * 1000.0
        logging.info("Frame %s: Inference Time = %.2f ms", frame_count, inference_ms)

        cv2.putText(
            annotated_frame,
            f"Inference Time: {inference_ms:.2f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        display_frame = cv2.resize(annotated_frame, (0, 0), fx=5 / 6, fy=5 / 6)
        cv2.imshow("YOLO Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("User interrupted")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Processing complete")


if __name__ == "__main__":
    main()
