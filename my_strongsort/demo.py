"""demo.py — jalankan MyStrongSort pada video sampel dan dump visualisasi tiap komponen.

Hasil ke outputs/my_strongsort/<komponen>/ (gambar PNG + matrix .npy/.csv):
  01 reid, 02 ecc, 03 ema, 04 kalman, 05 gate, 06 cost_matrix,
  07 assignment, 08 matched, 09 new_track, 10 lifecycle, 11 output

Contoh:
  venv/bin/python -m my_strongsort.demo --frames 30
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np

from my_strongsort import (assignment, cost_matrix, ecc, ema, gate, kalman,
                           matched, new_track, output, reid)
from my_strongsort.pipeline import MyStrongSort, build_detections_array

OUT = "outputs/my_strongsort"


def _mean_to_tlbr(mean):
    ret = np.asarray(mean[:4], dtype=float).copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    ret[2:] = ret[:2] + ret[2:]
    return ret


def run_and_collect(video="dataset/videos/ramai.mp4", yolo="models/7 - tuned/yolov8.pt",
                    frames=30, imgsz=1088, conf=0.25, iou=0.7):
    """Jalankan pipeline beberapa frame; kumpulkan snapshot frame 'kaya' + histori.

    Return (ss, rich, ema_hist, life_hist) di mana rich=(fid, frame, prev, last).
    Dipakai bersama oleh demo CLI dan notebook.
    """
    from ultralytics import YOLO
    model = YOLO(yolo)
    ss = MyStrongSort.from_default_config()
    cap = cv2.VideoCapture(video)

    ema_hist = defaultdict(list)          # id -> [fitur mentah per frame]
    life_hist = defaultdict(list)         # id -> [dict(frame, tsu, age, hits, state)]
    prev_frame, rich = None, None

    fid = 0
    while fid < frames:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        res = model.predict(frame, conf=conf, imgsz=imgsz, iou=iou, verbose=False)[0] # type: ignore
        dets = build_detections_array(res)
        ss.update(dets, frame)
        last = ss.last

        for tid, feat in last.get("matched_feats", []):
            ema_hist[tid].append(np.asarray(feat, dtype=float))
        for t in ss.tracks:
            life_hist[t.id].append(dict(frame=fid, time_since_update=t.time_since_update,
                                        age=t.age, hits=t.hits, state=t.state))

        ap_cost = last.get("appearance_cost")
        if ap_cost is not None and np.size(ap_cost) > 0:
            rich = (fid, frame.copy(), prev_frame, dict(last))
        prev_frame = frame.copy()

    cap.release()
    if rich is None:
        raise SystemExit("Tak ada frame dengan data matching appearance; naikkan frames.")
    return ss, rich, ema_hist, life_hist


def generate_pred(video, yolo, out_txt, imgsz=1088, conf=0.25, iou=0.7, max_frames=None):
    """Jalankan MyStrongSort atas video penuh, tulis pred.txt format MOT (spt main.py).

    Format baris identik dgn main.py: frame,id,x,y,w,h,conf,cls(=1),1
    (lihat MyStrongSort.run_video / output.to_mot_lines).
    """
    ss = MyStrongSort.from_default_config()
    ss.run_video(video, yolo, out_txt, conf=conf, iou=iou, imgsz=imgsz, max_frames=max_frames)
    print(f"[demo] pred.txt tersimpan di {out_txt}")
    return out_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dataset/videos/ramai.mp4")
    ap.add_argument("--yolo", default="models/7 - tuned/yolov8.pt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1088)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out-txt", default=None,
                     help="Jika diisi: lewati dump visualisasi, hanya jalankan seluruh video "
                          "dan tulis pred.txt MOT ke path ini (spt main.py).")
    ap.add_argument("--max-frames", type=int, default=None,
                     help="Batasi jumlah frame saat --out-txt dipakai (default: seluruh video).")
    args = ap.parse_args()

    if args.out_txt:
        generate_pred(args.video, args.yolo, args.out_txt,
                      imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                      max_frames=args.max_frames)
        return

    ss, rich, ema_hist, life_hist = run_and_collect(
        args.video, args.yolo, args.frames, args.imgsz, args.conf, args.iou)
    fid, frame, prev, last = rich
    print(f"[demo] frame terpilih untuk matriks: {fid}")

    # 1 — OSNet ReID
    reid.visualize(last["xyxy"], frame, last["features"], f"{OUT}/01_reid", tag=f"f{fid}")

    # 2 — ECC
    if "warp_matrix" in last:
        ecc.visualize(last["warp_matrix"], f"{OUT}/02_ecc", tag=f"f{fid}",
                      prev_img=prev, curr_img=frame)

    # 3 — EMA (track paling panjang riwayatnya)
    if ema_hist:
        tid = max(ema_hist, key=lambda k: len(ema_hist[k]))
        if len(ema_hist[tid]) >= 2:
            ema.visualize(ema_hist[tid], f"{OUT}/03_ema", tag=f"id{tid}", alpha=ss.ema_alpha)

    # 4 — NSA Kalman (satu track matched: predicted vs corrected + gain/innovation)
    if last["corrected_state"]:
        kid = next(iter(last["corrected_state"]))
        cmean, ccov = last["corrected_state"][kid]
        pred = last["predicted_state"].get(kid)
        kalman.visualize(cmean, ccov, f"{OUT}/04_kalman", tag=f"id{kid}",
                         extras=last["kalman_extras"].get(kid), predicted=pred)

    # 5 — Gate (Mahalanobis)
    if last.get("maha") is not None and np.size(last["maha"]) > 0:
        gate.visualize(last["maha"], f"{OUT}/05_gate", tag=f"f{fid}",
                       track_ids=last["appearance_track_ids"], det_ids=last["appearance_det_ids"])

    # 6 — Cost Matrix (appearance / fused / IoU)
    cost_matrix.visualize(
        {"appearance": last.get("appearance_cost"), "fused": last.get("fused_cost"),
         "iou": last.get("iou_cost")},
        f"{OUT}/06_cost_matrix", tag=f"f{fid}",
        track_ids=last.get("appearance_track_ids"), det_ids=last.get("appearance_det_ids"))

    # 7 — Hungarian Assignment (fused cost + pasangan lokal stage appearance)
    if last.get("fused_cost") is not None and np.size(last["fused_cost"]) > 0:
        assignment.visualize(last["fused_cost"], last.get("appearance_matches_local", []),
                             f"{OUT}/07_assignment", tag=f"f{fid}",
                             track_ids=last["appearance_track_ids"],
                             det_ids=last["appearance_det_ids"])

    # 8 — Matched update (predicted biru vs corrected hijau)
    if last["matched_before"]:
        ids = list(last["matched_before"].keys())
        before_boxes = [last["matched_before"][i] for i in ids]
        after_boxes = [_mean_to_tlbr(last["corrected_state"][i][0]) for i in ids]
        matched.visualize(frame, before_boxes, after_boxes, ids, f"{OUT}/08_matched", tag=f"f{fid}")

    # 9 — New Track (jika ada track lahir di frame ini)
    if last["new_tracks"]:
        new_track.visualize(frame, last["new_tracks"], f"{OUT}/09_new_track", tag=f"f{fid}")

    # 10 — Lifecycle timeline
    lifecycle_hist = {k: v for k, v in life_hist.items() if len(v) >= 2}
    if lifecycle_hist:
        from my_strongsort import lifecycle as lc
        lc.visualize(lifecycle_hist, f"{OUT}/10_lifecycle", tag="all_tracks")

    # 11 — Track Output
    if np.size(last["output"]) > 0:
        output.visualize(frame, last["output"], f"{OUT}/11_output", tag=f"f{fid}")

    print(f"[demo] visualisasi tersimpan di {OUT}/")


if __name__ == "__main__":
    main()
