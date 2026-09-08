"""verify_vs_boxmot.py — cek kesetaraan MyStrongSort vs boxmot StrongSort.

Deteksi YOLO dihitung SEKALI per frame lalu diumpankan ke KEDUA tracker, jadi
perbedaan output murni berasal dari implementasi tracker (bukan detektor).
Melaporkan kecocokan ID dan selisih koordinat box. Karena math di-mirror & bobot
ReID sama, output seharusnya (nyaris) identik.

  venv/bin/python -m my_strongsort.verify_vs_boxmot --frames 40
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from boxmot.trackers.strongsort.strongsort import StrongSort
from my_strongsort.pipeline import MyStrongSort, build_detections_array

REID = "strongsort/ReID/osnet_x0_25_msmt17.pt"
CONFIG = "strongsort/configs/default_config.yaml"


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _compare(out_mine, out_box):
    """Cocokkan baris berdasarkan GEOMETRI (IoU), lalu cek kesamaan id & koordinat.
    Track yang sama tapi berbeda label id (relabel) dilaporkan terpisah, bukan gagal.
    """
    mine = [(int(r[4]), np.asarray(r[:4], float)) for r in out_mine]
    box = [(int(r[4]), np.asarray(r[:4], float)) for r in out_box]
    used = set()
    matched, box_diffs, relabel = 0, [], 0
    for idm, bm in mine:
        best_j, best_iou = -1, 0.0
        for j, (idb, bb) in enumerate(box):
            if j in used:
                continue
            i = _iou(bm, bb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou > 0.5:
            used.add(best_j)
            matched += 1
            box_diffs.append(float(np.abs(bm - box[best_j][1]).max()))
            if idm != box[best_j][0]:
                relabel += 1
    return dict(matched=matched, relabel=relabel,
                only_mine=len(mine) - matched, only_box=len(box) - len(used),
                max_box_diff=max(box_diffs) if box_diffs else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dataset/videos/ramai.mp4")
    ap.add_argument("--yolo", default="models/7 - tuned/yolov5.pt")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    with open(CONFIG) as f:
        cfg = (yaml.safe_load(f) or {}).get("STRONGSORT", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    box = StrongSort(
        reid_weights=Path(REID), device=torch.device(device), half=False, min_conf=0.1,
        max_cos_dist=cfg["MAX_DIST"], max_iou_dist=cfg["MAX_IOU_DISTANCE"],
        max_age=cfg["MAX_AGE"], n_init=cfg["N_INIT"], nn_budget=cfg["NN_BUDGET"],
        mc_lambda=cfg["MC_LAMBDA"], ema_alpha=cfg["EMA_ALPHA"])
    mine = MyStrongSort.from_default_config(device=device)

    from ultralytics import YOLO
    model = YOLO(args.yolo)
    cap = cv2.VideoCapture(args.video)

    totals = dict(matched=0, relabel=0, only_mine=0, only_box=0)
    worst = 0.0
    fid = 0
    while fid < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, iou=args.iou, verbose=False)[0]
        dets = build_detections_array(res)
        o_mine = mine.update(dets.copy(), frame.copy())
        o_box = box.update(dets.copy(), frame.copy())
        c = _compare(o_mine, o_box)
        for k in totals:
            totals[k] += c[k]
        worst = max(worst, c["max_box_diff"])
    cap.release()

    print(f"\nframes={fid}  boxes_matched={totals['matched']}  "
          f"max_box_diff={worst:.4f}px  id_relabel={totals['relabel']}  "
          f"unmatched(mine/box)={totals['only_mine']}/{totals['only_box']}")
    # ponytail: relabel diizinkan — id-counter bisa geser 1 akibat tie-break pada track
    # Tentative yg tak pernah di-emit; yg penting box & asosiasi identik (max_box_diff~0).
    ok = totals["only_mine"] == 0 and totals["only_box"] == 0 and worst < 1.0
    print("VERDICT:", "SETARA ✓ (box identik; id relabel=%d)" % totals["relabel"]
          if ok else "ADA SELISIH GEOMETRI — periksa di atas")


if __name__ == "__main__":
    main()
