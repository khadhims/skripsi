from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


def generate_yolo_only_prediction(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    conf_threshold: float,
    imgsz: int,
    iou_threshold: float,
) -> Path:
    model = YOLO(str(model_path))
    model.fuse()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    det_id_counter = 0

    with open(output_path, "w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            results = model.predict(
                frame,
                conf=conf_threshold,
                imgsz=imgsz,
                iou=iou_threshold,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    det_id_counter += 1
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())

                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    f.write(
                        f"{frame_id},{det_id_counter},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},1,1\n"
                    )

    cap.release()
    return output_path
