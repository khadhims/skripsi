"""
Membandingkan performa sistem:
  1. Dengan Tracking   (YOLO + StrongSORT)
  2. Tanpa Tracking    (YOLO saja)

Metrik:
  - Precision  (IoU-based per frame)
  - Recall     (IoU-based per frame)
  - Object Counting Error (per frame & total unique objects)

Cara pakai:
  python compare_tracking.py

Pastikan:
  - File GT sudah ada di dataset/tracker-val/
  - File pred tracking sudah ada di dataset/tracker-results/
  - Video tersedia di dataset/videos/
  - Model YOLO tersedia di folder models/
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO


# ==============================
# CONFIG
# ==============================
CONFIGS = [
    {
        "name": "v8-medium-it2-vid1",
        "gt": "./dataset/tracker-val/vid1-gt.txt",
        "tracked_pred": "./dataset/tracker-results/iterasi-1/v8-medium-it2-vid1-pred.txt",
        "video": "./dataset/videos/vid_1.mp4",
        "model": "./models/yolov8/iterasi-2/v8-medium.pt",
    },
    {
        "name": "v8-medium-it2-vid2",
        "gt": "./dataset/tracker-val/vid2-gt.txt",
        "tracked_pred": "./dataset/tracker-results/iterasi-1/v8-medium-it2-vid2-pred.txt",
        "video": "./dataset/videos/vid_2.mp4",
        "model": "./models/yolov8/iterasi-2/v8-medium.pt",
    },
    {
        "name": "v8-medium-it2-vid3",
        "gt": "./dataset/tracker-val/vid3-gt.txt",
        "tracked_pred": "./dataset/tracker-results/iterasi-1/v8-medium-it2-vid3-pred.txt",
        "video": "./dataset/videos/vid_3.mp4",
        "model": "./models/yolov8/iterasi-2/v8-medium.pt",
    },
]

IOU_THRESHOLD = 0.7
CONF_THRESHOLD = 0.25
IMGSZ = 1088


# ==============================
# UTILS
# ==============================

def load_mot_file(path):
    """Load file format MOT: frame, id, x, y, w, h, conf, class, visibility"""
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
    df = pd.read_csv(path, header=None)
    df.columns = cols
    return df


def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def compute_iou(box1, box2):
    """Hitung IoU antara dua box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union == 0:
        return 0.0
    return inter / union


def match_frame_boxes(gt_boxes, pred_boxes, iou_threshold):
    """
    Greedy matching antara GT dan prediksi berdasarkan IoU.
    Returns: (tp, fp, fn)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    # Hitung IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gb, pb)

    matched_gt = set()
    matched_pred = set()

    # Greedy matching: ambil pasangan dengan IoU tertinggi
    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        gi, pi = idx[0], idx[1]
        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_matrix[gi, :] = 0
        iou_matrix[:, pi] = 0

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


# ==============================
# GENERATE YOLO-ONLY PREDICTIONS
# ==============================

def generate_yolo_only_pred(video_path, model_path, output_path):
    """
    Jalankan YOLO inference TANPA tracking.
    Simpan hasil dalam format MOT: frame, id, x, y, w, h, conf, class, visibility
    ID di-assign sequential per-frame (tidak persistent).
    """
    print(f"  Generating YOLO-only predictions: {output_path}")

    model = YOLO(model_path)
    model.fuse()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    frame_id = 0
    det_id_counter = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False, iou=0.7)

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    det_id_counter += 1
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())
                    x1, y1, x2, y2 = xyxy
                    x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                    # Agnostic: class = 1
                    f.write(f"{frame_id},{det_id_counter},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},1,1\n")

            if frame_id % 100 == 0:
                print(f"    Frame {frame_id} selesai...")

    cap.release()
    print(f"  Selesai. Total frames: {frame_id}, Total deteksi: {det_id_counter}")
    return output_path


# ==============================
# EVALUASI
# ==============================

def evaluate(gt_path, pred_path, iou_threshold):
    """
    Hitung Precision, Recall, dan Object Counting Error
    berdasarkan GT dan pred file.

    Returns dict: precision, recall, avg_count_error, total_unique_gt, total_unique_pred
    """
    gt_df = load_mot_file(gt_path)
    pred_df = load_mot_file(pred_path)

    all_frames = sorted(gt_df["frame"].unique())

    total_tp = 0
    total_fp = 0
    total_fn = 0
    count_errors = []

    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        # Konversi ke list of [x1,y1,x2,y2]
        gt_boxes = [
            xywh_to_xyxy(r["x"], r["y"], r["w"], r["h"])
            for _, r in gt_frame.iterrows()
        ]
        pred_boxes = [
            xywh_to_xyxy(r["x"], r["y"], r["w"], r["h"])
            for _, r in pred_frame.iterrows()
        ]

        tp, fp, fn = match_frame_boxes(gt_boxes, pred_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Object Counting Error per frame
        count_error = abs(len(pred_boxes) - len(gt_boxes))
        count_errors.append(count_error)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    avg_count_error = np.mean(count_errors) if count_errors else 0.0
    total_count_error = abs(pred_df["id"].nunique() - gt_df["id"].nunique())

    return {
        "precision": precision,
        "recall": recall,
        "avg_frame_count_error": avg_count_error,
        "total_unique_count_error": total_count_error,
        "total_unique_gt": gt_df["id"].nunique(),
        "total_unique_pred": pred_df["id"].nunique(),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


# ==============================
# MAIN
# ==============================

def print_comparison(name, tracked_result, yolo_result):
    print(f"\n{'=' * 60}")
    print(f"  PERBANDINGAN: {name}")
    print(f"{'=' * 60}")

    header = f"{'Metrik':<35} {'Tracking':>10} {'YOLO Only':>10}"
    print(header)
    print("-" * 60)

    rows = [
        ("Precision", f"{tracked_result['precision']:.4f}", f"{yolo_result['precision']:.4f}"),
        ("Recall", f"{tracked_result['recall']:.4f}", f"{yolo_result['recall']:.4f}"),
        ("Avg Frame Count Error", f"{tracked_result['avg_frame_count_error']:.2f}", f"{yolo_result['avg_frame_count_error']:.2f}"),
        ("Total Unique Objects (GT)", f"{tracked_result['total_unique_gt']}", f"{yolo_result['total_unique_gt']}"),
        ("Total Unique Objects (Pred)", f"{tracked_result['total_unique_pred']}", f"{yolo_result['total_unique_pred']}"),
        ("Total Unique Count Error", f"{tracked_result['total_unique_count_error']}", f"{yolo_result['total_unique_count_error']}"),
        ("TP", f"{tracked_result['total_tp']}", f"{yolo_result['total_tp']}"),
        ("FP", f"{tracked_result['total_fp']}", f"{yolo_result['total_fp']}"),
        ("FN", f"{tracked_result['total_fn']}", f"{yolo_result['total_fn']}"),
    ]

    for label, val_t, val_y in rows:
        print(f"{label:<35} {val_t:>10} {val_y:>10}")

    print(f"{'=' * 60}")


def main():
    print("=" * 60)
    print("  PERBANDINGAN PERFORMA: TRACKING vs YOLO-ONLY")
    print(f"  IoU Threshold: {IOU_THRESHOLD}")
    print(f"  Confidence   : {CONF_THRESHOLD}")
    print("=" * 60)

    all_tracked = []
    all_yolo = []

    for cfg in CONFIGS:
        name = cfg["name"]
        gt_path = cfg["gt"]
        tracked_pred_path = cfg["tracked_pred"]
        video_path = cfg["video"]
        model_path = cfg["model"]

        print(f"\n>>> Evaluasi: {name}")

        # Cek file GT dan tracked pred
        if not Path(gt_path).exists():
            print(f"  SKIP: GT tidak ditemukan → {gt_path}")
            continue
        if not Path(tracked_pred_path).exists():
            print(f"  SKIP: Tracked pred tidak ditemukan → {tracked_pred_path}")
            continue

        # Generate YOLO-only pred jika belum ada
        yolo_pred_path = Path(tracked_pred_path).parent / (
            Path(tracked_pred_path).stem.replace("-pred", "-yoloonly") + ".txt"
        )

        if not yolo_pred_path.exists():
            if not Path(video_path).exists():
                print(f"  SKIP: Video tidak ditemukan → {video_path}")
                continue
            if not Path(model_path).exists():
                print(f"  SKIP: Model tidak ditemukan → {model_path}")
                continue
            generate_yolo_only_pred(video_path, model_path, yolo_pred_path)
        else:
            print(f"  YOLO-only pred sudah ada: {yolo_pred_path}")

        # Evaluasi kedua metode
        print(f"  Evaluasi TRACKING...")
        tracked_result = evaluate(gt_path, tracked_pred_path, IOU_THRESHOLD)

        print(f"  Evaluasi YOLO-ONLY...")
        yolo_result = evaluate(gt_path, str(yolo_pred_path), IOU_THRESHOLD)

        all_tracked.append(tracked_result)
        all_yolo.append(yolo_result)

        print_comparison(name, tracked_result, yolo_result)

    # Rata-rata keseluruhan
    if len(all_tracked) > 1:
        print(f"\n{'=' * 60}")
        print(f"  RATA-RATA KESELURUHAN")
        print(f"{'=' * 60}")

        header = f"{'Metrik':<35} {'Tracking':>10} {'YOLO Only':>10}"
        print(header)
        print("-" * 60)

        avg_metrics = ["precision", "recall", "avg_frame_count_error"]
        labels = ["Precision", "Recall", "Avg Frame Count Error"]

        for label, key in zip(labels, avg_metrics):
            val_t = np.mean([r[key] for r in all_tracked])
            val_y = np.mean([r[key] for r in all_yolo])
            print(f"{label:<35} {val_t:>10.4f} {val_y:>10.4f}")

        sum_metrics = ["total_tp", "total_fp", "total_fn"]
        sum_labels = ["Total TP", "Total FP", "Total FN"]
        for label, key in zip(sum_labels, sum_metrics):
            val_t = sum(r[key] for r in all_tracked)
            val_y = sum(r[key] for r in all_yolo)
            print(f"{label:<35} {val_t:>10} {val_y:>10}")

        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
