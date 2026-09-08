"""Replikasi generator log I/O per-komponen (dipakai skripsi) — dump nilai
numerik pipeline StrongSORT pada satu frame "hero" ke CSV/JSON di
outputs/my_strongsort/thesis_math/io_logs/.

Bukan re-implementasi algoritma (itu sudah ada di modul komponennya masing-
masing) — script ini cuma menjalankan MyStrongSort beberapa frame lalu
mem-broadcast isi `ss.last` (dan riwayat EMA) apa adanya ke file.

Catatan: file `log_XX_*.md` yang narasi/pedagogis (formula, diagram ASCII,
penjelasan arsitektur OSNet/ECC) TIDAK di-generate ulang di sini — itu teks
tulisan tangan, bukan hasil komputasi. Script ini hanya mereproduksi bagian
yang benar-benar data: 00_summary.json dan seluruh *.csv.

Contoh:
  venv/bin/python -m my_strongsort.thesis_io_logs --frames 30
"""
import argparse
import json
import os

import numpy as np

from my_strongsort.demo import run_and_collect
from my_strongsort.gate import GATING_THRESHOLD

OUT = "outputs/my_strongsort/thesis_math/io_logs"
STATE_LABELS = ["x", "y", "a", "h", "vx", "vy", "va", "vh"]
Z_LABELS = ["x", "y", "a", "h"]


def _csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def dump(video="dataset/videos/ramai.mp4", yolo="models/7 - tuned/yolov5.pt",
         frames=30, imgsz=1088, conf=0.25, iou=0.7):
    os.makedirs(OUT, exist_ok=True)
    ss, rich, ema_hist, life_hist = run_and_collect(video, yolo, frames, imgsz, conf, iou)
    fid, frame, prev_frame, last = rich

    xyxy = last["xyxy"]
    feats = last["features"]
    dets = last["detections"]

    # "Track #1" = track tertua yang masih ter-match di frame ini (hero skenario)
    hero_id = min(last["corrected_state"]) if last["corrected_state"] else None
    if hero_id is None:
        raise SystemExit("Tak ada track matched di frame terpilih; naikkan --frames.")
    row = last["appearance_track_ids"].index(hero_id)
    maha = last["maha"][row]
    appearance = last["appearance_cost"][row]
    fused = last["fused_cost"][row]
    hero_idx = int(np.argmin(fused))  # det ter-match = cost fusi terkecil

    # ---- 01: semua deteksi YOLO ----
    _csv(f"{OUT}/01_yolo_detections_frame{fid}.csv",
         "det_idx,x1,y1,x2,y2,conf,class",
         [[d.det_ind, *[f"{v:.6f}" for v in xyxy[i]], f"{d.conf:.6f}", float(d.cls)]
          for i, d in enumerate(dets)])

    # ---- 02a: crop pixel hero ----
    x1, y1, x2, y2 = [int(v) for v in xyxy[hero_idx]]
    crop = frame[y1:y2, x1:x2]
    rows = [(r, c, *crop[r, c]) for r in range(crop.shape[0]) for c in range(crop.shape[1])]
    _csv(f"{OUT}/02a_reid_input_crop_pixels.csv", "row_y,col_x,B,G,R", rows)

    # ---- 02b: fitur hero 512D ----
    _csv(f"{OUT}/02b_reid_output_feat512d.csv", "dim_idx,value",
         [(i, f"{v:.10f}") for i, v in enumerate(feats[hero_idx])])

    # ---- 02c: semua fitur deteksi ----
    header = "det_idx," + ",".join(f"d{i}" for i in range(feats.shape[1]))
    _csv(f"{OUT}/02c_reid_all_dets_features.csv", header,
         [[i, *[f"{v:.8f}" for v in feats[i]]] for i in range(len(feats))])

    # ---- 03: warp matrix ECC ----
    warp = last["warp_matrix"]
    _csv(f"{OUT}/03_ecc_warp_matrix.csv", "row,col0,col1,col2 (translation)",
         [[i, *[f"{v:.8f}" for v in row]] for i, row in enumerate(warp)])

    # ---- 04: prediksi Kalman hero ----
    pred_mean, pred_cov = last["predicted_state"][hero_id]
    _csv(f"{OUT}/04_kalman_predict_mean.csv", "dim,symbol,value",
         [[i, s, f"{v:.8f}"] for i, (s, v) in enumerate(zip(STATE_LABELS, pred_mean))])
    _csv(f"{OUT}/04_kalman_predict_covariance_8x8.csv", "," + ",".join(STATE_LABELS),
         [[STATE_LABELS[i], *[f"{v:.8f}" for v in pred_cov[i]]] for i in range(8)])

    # ---- 05: update Kalman hero ----
    extras = last["kalman_extras"][hero_id]
    corr_mean, corr_cov = last["corrected_state"][hero_id]
    K, S = extras["kalman_gain"], extras["projected_cov"]
    innov, proj_mean = extras["innovation"], extras["projected_mean"]
    meas = dets[hero_idx].to_xyah()

    _csv(f"{OUT}/05a_kalman_update_S_4x4.csv", "," + ",".join(Z_LABELS),
         [[Z_LABELS[i], *[f"{v:.8f}" for v in S[i]]] for i in range(4)])
    _csv(f"{OUT}/05b_kalman_update_K_8x4.csv", "state," + ",".join(f"z_{z}" for z in Z_LABELS),
         [[STATE_LABELS[i], *[f"{v:.8f}" for v in K[i]]] for i in range(8)])
    _csv(f"{OUT}/05c_kalman_update_innovation.csv",
         "dim,symbol,z (measurement),H·x_pred (projected),y = z - H·x_pred",
         [[i, Z_LABELS[i], f"{meas[i]:.8f}", f"{proj_mean[i]:.8f}", f"{innov[i]:.8f}"]
          for i in range(4)])
    correction = K @ innov
    _csv(f"{OUT}/05d_kalman_corrected_mean.csv", "dim,symbol,x_pred,K·y (correction),x_corr",
         [[i, STATE_LABELS[i], f"{pred_mean[i]:.8f}", f"{correction[i]:.8f}", f"{corr_mean[i]:.8f}"]
          for i in range(8)])
    _csv(f"{OUT}/05e_kalman_corrected_covariance_8x8.csv", "," + ",".join(STATE_LABELS),
         [[STATE_LABELS[i], *[f"{v:.8f}" for v in corr_cov[i]]] for i in range(8)])

    # ---- 06: gating mahalanobis ----
    measurements = np.array([d.to_xyah() for d in dets])
    _csv(f"{OUT}/06_gating_mahalanobis.csv",
         "det_idx,cx,cy,a,h,d2_mahalanobis,threshold,pass_gate",
         [[i, *[f"{v:.4f}" for v in measurements[i]], f"{maha[i]:.6f}",
           GATING_THRESHOLD, "YES" if maha[i] <= GATING_THRESHOLD else "NO"]
          for i in range(len(maha))])

    # ---- 07: cost matrix ----
    _csv(f"{OUT}/07_cost_matrix.csv",
         "det_idx,cos_sim,appearance_cost (1-cos),maha_norm,fused_cost,selected",
         [[i, f"{1 - appearance[i]:.8f}", f"{appearance[i]:.8f}", f"{maha[i]:.8f}",
           f"{fused[i]:.8f}", "MATCH" if i == hero_idx else ""]
          for i in range(len(appearance))])

    # ---- 09: EMA update hero ----
    hist = ema_hist.get(hero_id, [])
    if len(hist) >= 2:
        f_prev_u = hist[-2] / np.linalg.norm(hist[-2])
        f_new_u = hist[-1] / np.linalg.norm(hist[-1])
        blend = 0.9 * f_prev_u + 0.1 * f_new_u
        smooth = blend / np.linalg.norm(blend)
        _csv(f"{OUT}/09_ema_update_full512d.csv",
             "dim_idx,f_prev,f_new,0.9*f_prev + 0.1*f_new (pre-norm),f_smooth (post L2-norm)",
             [[i, f"{f_prev_u[i]:.10f}", f"{f_new_u[i]:.10f}", f"{blend[i]:.10f}", f"{smooth[i]:.10f}"]
              for i in range(len(f_prev_u))])

    # ---- 00: ringkasan ----
    summary = dict(
        frame=fid, hero_det_idx=hero_idx,
        hero_detection_6d=[*[float(v) for v in xyxy[hero_idx]],
                            float(dets[hero_idx].conf), float(dets[hero_idx].cls)],
        hero_measurement_xyah=[float(v) for v in meas],
        hero_conf=float(dets[hero_idx].conf),
        hero_crop_shape=list(crop.shape),
        hero_crop_bbox=dict(x1=x1, y1=y1, x2=x2, y2=y2),
        ecc_warp=dict(dx=float(warp[0][2]), dy=float(warp[1][2])),
        pred_mean=[float(v) for v in pred_mean],
        corr_mean=[float(v) for v in corr_mean],
        innovation=[float(v) for v in innov],
        hero_d2=float(maha[hero_idx]), gate_threshold=GATING_THRESHOLD,
        cos_sim=float(1 - appearance[hero_idx]), appearance_cost=float(appearance[hero_idx]),
        fused_cost=float(fused[hero_idx]),
        K_diagonal=[float(K[i, i]) for i in range(4)],
        S_diagonal=[float(S[i, i]) for i in range(4)],
        total_dets_frame30=len(dets),
        feat_norm_new=float(np.linalg.norm(feats[hero_idx])), feat_norm_smooth=1.0,
    )
    with open(f"{OUT}/00_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[thesis_io_logs] frame terpilih: {fid} | hero track id={hero_id} det_idx={hero_idx}")
    print(f"[thesis_io_logs] CSV/JSON tersimpan di {OUT}/")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dataset/videos/ramai.mp4")
    ap.add_argument("--yolo", default="models/7 - tuned/yolov5.pt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1088)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()
    dump(args.video, args.yolo, args.frames, args.imgsz, args.conf, args.iou)


if __name__ == "__main__":
    main()
