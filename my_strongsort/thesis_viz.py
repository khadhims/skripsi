"""thesis_viz.py — Satu objek (hero), 11 komponen StrongSORT, thesis-friendly.

Pilih satu track yang paling panjang riwayatnya, buat visualisasi bertahap
dari output YOLO hingga track output. Semua gambar di-save ke:
    outputs/my_strongsort/thesis/

    venv/bin/python -m my_strongsort.thesis_viz --frames 30
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from my_strongsort.output import to_mot_lines

OUT = "outputs/my_strongsort/thesis"
THRESHOLD_MAHA = 9.4877   # chi2inv95[4]
FS = 11                    # base font size


def _savefig(fig, name):
    path = f"{OUT}/{name}"
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _tlbr(mean):
    cx, cy, a, h = float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3])
    w = a * h
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def _box(ax, xyxy, color, lw=2, ls="-", label=None, alpha=1.0):
    x1, y1, x2, y2 = xyxy
    kw = dict(linewidth=lw, edgecolor=color, facecolor="none", linestyle=ls, alpha=alpha)
    if label:
        kw["label"] = label
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, **kw))


# ── 01 — YOLO Detection ──────────────────────────────────────────
def viz_01(frame, all_xyxy, hero_xyxy, hero_conf):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(_rgb(frame))
    for box in all_xyxy:
        _box(ax, box, "gray", lw=1, ls="--", alpha=0.4)
    _box(ax, hero_xyxy, "gold", lw=3, label="Objek Hero")
    x1, y1 = hero_xyxy[0], hero_xyxy[1]
    ax.text(x1, y1 - 6, f"conf = {hero_conf:.2f}",
            color="gold", fontsize=FS, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=2))
    ax.legend(loc="upper right", fontsize=9, facecolor="black", labelcolor="white")
    ax.set_title(f"01 — Output Deteksi YOLO  |  {len(all_xyxy)} objek terdeteksi\n"
                 "Abu-abu = semua deteksi  ·  Kuning = objek hero yang akan dilacak",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _savefig(fig, "01_detection.png")


# ── 02 — ECC Camera Compensation ─────────────────────────────────
def viz_02(curr_frame, prev_frame, warp_matrix, pred_xyxy):
    dx = float(warp_matrix[0, 2])
    dy = float(warp_matrix[1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if prev_frame is not None:
        axes[0].imshow(_rgb(prev_frame))
        x1p = pred_xyxy[0] - dx
        y1p = pred_xyxy[1] - dy
        _box(axes[0], [x1p, y1p, x1p + (pred_xyxy[2]-pred_xyxy[0]),
                       y1p + (pred_xyxy[3]-pred_xyxy[1])], "cyan", lw=2,
             label="Posisi track di t-1")
        axes[0].legend(fontsize=8, facecolor="black", labelcolor="white")
    axes[0].set_title("Frame sebelumnya (t-1)", fontsize=FS)
    axes[0].axis("off")

    axes[1].imshow(_rgb(curr_frame))
    _box(axes[1], pred_xyxy, "cyan", lw=2, label="Prediksi setelah ECC")

    H, W = curr_frame.shape[:2]
    cx, cy = W // 2, H // 2
    scale = max(10, min(W, H) // 40)
    axes[1].annotate("", xy=(cx + dx * scale, cy + dy * scale), xytext=(cx, cy),
                     arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    axes[1].text(cx + dx * scale + 8, cy + dy * scale,
                 f"Gerak kamera:\ndx = {dx:+.1f} px\ndy = {dy:+.1f} px",
                 color="red", fontsize=10, bbox=dict(facecolor="white", alpha=0.75, pad=3))
    axes[1].legend(fontsize=8, facecolor="black", labelcolor="white")
    axes[1].set_title("Frame sekarang (t)  |  Prediksi track digeser kompensasi kamera",
                      fontsize=FS)
    axes[1].axis("off")

    fig.suptitle(f"02 — ECC: Kompensasi Gerak Kamera  "
                 f"(dx = {dx:+.1f} px,  dy = {dy:+.1f} px)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "02_ecc.png")


# ── 03 — OSNet ReID ───────────────────────────────────────────────
def viz_03(frame, hero_xyxy, hero_feat):
    x1, y1, x2, y2 = [int(v) for v in hero_xyxy]
    crop = frame[max(0, y1):max(y2, y1+1), max(0, x1):max(x2, x1+1)]
    if crop.size == 0:
        crop = np.zeros((256, 128, 3), dtype=np.uint8)
    crop_rgb = _rgb(cv2.resize(crop, (128, 256)))

    feat = np.asarray(hero_feat, dtype=float)
    feat_2d = feat.reshape(16, 32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1, 3]})
    axes[0].imshow(crop_rgb)
    axes[0].set_title("Crop Objek Hero\n(input ke OSNet)", fontsize=FS)
    axes[0].axis("off")

    im = axes[1].imshow(feat_2d, aspect="auto", cmap="viridis")
    axes[1].set_title("Fitur Appearance 512-D  (L2-ternormalisasi)\n"
                      "Disusun ulang menjadi 16 baris × 32 kolom untuk visualisasi",
                      fontsize=FS)
    axes[1].set_xlabel("Kolom dimensi (0–31)", fontsize=9)
    axes[1].set_ylabel("Baris dimensi (0–15)", fontsize=9)
    cb = plt.colorbar(im, ax=axes[1], fraction=0.03)
    cb.set_label("Nilai aktivasi", fontsize=9)
    axes[1].text(0.01, 0.99, f"‖fitur‖₂ = {np.linalg.norm(feat):.4f}  (≈ 1.0 = L2-norm)",
                 transform=axes[1].transAxes, fontsize=9, va="top",
                 bbox=dict(facecolor="white", alpha=0.8, pad=2))

    fig.suptitle("03 — OSNet ReID: Ekstraksi Fitur Appearance 512-D dari Objek Hero",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "03_reid.png")


# ── 04 — NSA Kalman Filter (Predict) ─────────────────────────────
def viz_04(frame, pred_mean, pred_cov, corr_mean, hero_det_xyxy):
    labels = ["x", "y", "a", "h", "vx", "vy", "va", "vh"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Kiri: frame dengan tiga kotak + ellipse
    axes[0].imshow(_rgb(frame))
    _box(axes[0], _tlbr(pred_mean), "royalblue", lw=2, ls="--",
         label="Prediksi KF  (sebelum koreksi)")
    _box(axes[0], hero_det_xyxy, "gold", lw=2, ls=":",
         label="Deteksi YOLO  (pengukuran)")
    _box(axes[0], _tlbr(corr_mean), "limegreen", lw=2.5,
         label="Terkoreksi KF  (sesudah koreksi)")

    cov2 = pred_cov[:2, :2]
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0)
    el_w = 2 * np.sqrt(vals[1]) * 3
    el_h = 2 * np.sqrt(vals[0]) * 3
    angle = np.degrees(np.arctan2(float(vecs[1, 1]), float(vecs[0, 1])))
    cx_e, cy_e = float(pred_mean[0]), float(pred_mean[1])
    axes[0].add_patch(Ellipse((cx_e, cy_e), el_w, el_h, angle=angle,
                              edgecolor="royalblue", facecolor="royalblue",
                              alpha=0.12, lw=1.5, ls="--"))
    axes[0].legend(loc="upper left", fontsize=8, facecolor="black", labelcolor="white")
    axes[0].set_title("Biru = prediksi KF  ·  Kuning = deteksi  ·  Hijau = terkoreksi\n"
                      "Ellipse biru = ketidakpastian posisi (3σ)", fontsize=FS)
    axes[0].axis("off")

    # Kanan: state vector bar chart
    colors_bar = ["royalblue"] * 4 + ["darkorange"] * 4
    y = [float(v) for v in pred_mean]
    bars = axes[1].bar(labels, y, color=colors_bar, alpha=0.82,
                       edgecolor="black", linewidth=0.5)
    ymax = max(abs(v) for v in y)
    for bar, v in zip(bars, y):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     v + (ymax * 0.025 if v >= 0 else -ymax * 0.025),
                     f"{v:.1f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title("State Vector Prediksi NSA Kalman\n"
                      "Biru = posisi/ukuran [x, y, a, h]  ·  Oranye = kecepatan [vx, vy, va, vh]",
                      fontsize=FS)
    axes[1].set_ylabel("Nilai", fontsize=FS)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("04 — NSA Kalman Filter: Prediksi Posisi → Koreksi dengan Deteksi",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "04_kalman.png")


# ── 05 — EMA ─────────────────────────────────────────────────────
def viz_05(raw_feats, smooth_feat, alpha=0.9):
    if len(raw_feats) < 2:
        print("  05_ema: histori terlalu pendek, skip")
        return

    from my_strongsort.ema import smooth as ema_smooth
    drifts = []
    s = raw_feats[0] / (np.linalg.norm(raw_feats[0]) + 1e-12)
    for raw in raw_feats[1:]:
        r = raw / (np.linalg.norm(raw) + 1e-12)
        drifts.append(1.0 - float(np.dot(s, r)))
        s = ema_smooth(s, raw, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    frames_x = list(range(2, len(raw_feats) + 1))
    axes[0].plot(frames_x, drifts, marker="o", color="steelblue", lw=2, ms=4)
    axes[0].fill_between(frames_x, drifts, alpha=0.18, color="steelblue")
    axes[0].axhline(float(np.mean(drifts)), color="darkorange", ls="--",
                    label=f"Rata-rata drift = {float(np.mean(drifts)):.3f}")
    axes[0].set_xlabel("Nomor Frame", fontsize=FS)
    axes[0].set_ylabel("Cosine Distance  (fitur mentah vs fitur smooth)", fontsize=FS)
    axes[0].set_title(f"Drift Fitur per Frame  (α = {alpha})\n"
                      "Nilai kecil = fitur stabil antar frame",
                      fontsize=FS)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    # Bar: 32 dimensi pertama
    raw_last = raw_feats[-1][:32]
    x = np.arange(32)
    w = 0.38
    axes[1].bar(x - w/2, raw_last, w, label="Fitur mentah (raw, frame ini)",
                alpha=0.78, color="tomato", zorder=3)
    if smooth_feat is not None:
        axes[1].bar(x + w/2, smooth_feat[:32], w,
                    label=f"Fitur EMA  (α·prev + (1−α)·new,  α={alpha})",
                    alpha=0.78, color="steelblue", zorder=3)
    axes[1].set_xlabel("Indeks dimensi ke-0 s/d 31  (dari 512 total)", fontsize=9)
    axes[1].set_ylabel("Nilai fitur", fontsize=9)
    axes[1].set_title("Perbandingan 32 Dimensi Pertama:\nFitur Mentah vs Sesudah EMA",
                      fontsize=FS)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"05 — EMA: Penghalusan Fitur Appearance  "
                 f"(histori {len(raw_feats)} frame,  α = {alpha})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "05_ema.png")


# ── 06 — Gate Mahalanobis ─────────────────────────────────────────
def viz_06(maha, hero_row, adids):
    d = maha[hero_row].copy()
    n = len(d)
    colors = ["limegreen" if v <= THRESHOLD_MAHA else "tomato" for v in d]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))
    bars = ax.bar(range(n), d, color=colors, edgecolor="black", lw=0.8, alpha=0.85, zorder=3)
    ax.axhline(THRESHOLD_MAHA, color="red", lw=2, ls="--",
               label=f"Threshold  χ²(df=4, p=0.95) = {THRESHOLD_MAHA}",
               zorder=4)

    for i, (bar, v) in enumerate(zip(bars, d)):
        label_s = "✓ lolos" if v <= THRESHOLD_MAHA else "✗ dibuang"
        ax.text(i, v + THRESHOLD_MAHA * 0.05,
                f"d² = {v:.1f}\n{label_s}",
                ha="center", va="bottom", fontsize=8,
                color="darkgreen" if v <= THRESHOLD_MAHA else "darkred")

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"Det {adids[i]}" for i in range(n)], fontsize=FS)
    ax.set_ylabel("Jarak Mahalanobis²  (ruang state Kalman 4D)", fontsize=FS)
    ax.set_title("06 — Gate Mahalanobis: Track Hero vs Semua Deteksi\n"
                 "Hijau = geometri konsisten (lolos)  ·  Merah = mustahil secara gerak (dibuang)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0, top=max(float(d.max()), THRESHOLD_MAHA) * 1.38)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "06_gate.png")


# ── 07 — Cost Matrix (baris hero) ────────────────────────────────
def viz_07(ap_cost, maha, fused, hero_row, matches_local, adids, mc_lambda=0.995):
    ap = ap_cost[hero_row].copy()
    ma = maha[hero_row].copy()
    fu = np.minimum(fused[hero_row].copy(), 1.0)   # cap INFTY → 1.0 untuk display
    n = len(ap)

    # Normalisasi maha ke 0-1 (threshold = 1.0)
    ma_norm = np.minimum(ma / THRESHOLD_MAHA, 2.0) / 2.0

    x = np.arange(n)
    w = 0.25
    matched_col = next((c for r, c in matches_local if r == hero_row), None)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.7), 6))
    ax.bar(x - w, ap, w, label="Appearance cost  (1 − cosine_similarity)",
           color="steelblue", alpha=0.82)
    ax.bar(x, ma_norm, w,
           label="Mahalanobis²  (dinormalisasi: 0=dekat, 1=2×threshold)",
           color="darkorange", alpha=0.82)
    ax.bar(x + w, fu, w,
           label=f"Fused  = {mc_lambda}·appearance + {1-mc_lambda:.3f}·mahalanobis",
           color="mediumpurple", alpha=0.82)

    if matched_col is not None and matched_col < n:
        ax.axvspan(matched_col - 0.52, matched_col + 0.52,
                   color="yellow", alpha=0.15, zorder=0)
        ymax_val = max(float(ap.max()), float(ma_norm.max()), float(fu.max()))
        ax.text(matched_col, ymax_val * 1.02,
                "MATCH\n(biaya fused minimum)",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="darkgreen",
                bbox=dict(facecolor="white", alpha=0.75, pad=2))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Det {adids[i]}" for i in range(n)], fontsize=FS)
    ax.set_ylabel("Biaya  (lebih kecil = lebih cocok)", fontsize=FS)
    ax.set_title("07 — Cost Matrix: Biaya Track Hero ke Setiap Deteksi\n"
                 "Kolom kuning = deteksi yang terpilih (biaya fused minimum)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "07_cost.png")


# ── 08 — Hungarian Assignment ─────────────────────────────────────
def viz_08(fused_cost, matches_local, track_ids, adids, hero_id):
    n_t, n_d = fused_cost.shape
    hero_row = track_ids.index(hero_id)

    disp = fused_cost.copy().astype(float)
    disp[disp >= 1e4] = np.nan

    vmax = 1.0
    if not np.all(np.isnan(disp)):
        vmax = min(1.0, float(np.nanmax(disp)))

    fig, ax = plt.subplots(figsize=(max(7, n_d * 1.3 + 1), max(4, n_t * 0.75 + 1)))
    im = ax.imshow(disp, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax,
                 label="Biaya Fused  (hijau = cocok · merah = tidak cocok · putih = di-gate)")

    for r, c in matches_local:
        ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                         lw=3, edgecolor="white", facecolor="none"))

    ax.add_patch(mpatches.Rectangle((-0.5, hero_row - 0.5), n_d, 1,
                                     lw=2, edgecolor="gold", facecolor="gold", alpha=0.18))

    ax.set_xticks(range(n_d))
    ax.set_xticklabels([f"Det {d}" for d in adids], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([f"Track {t}  ← hero" if t == hero_id else f"Track {t}"
                        for t in track_ids], fontsize=9)
    ax.set_xlabel("Deteksi", fontsize=FS)
    ax.set_ylabel("Track", fontsize=FS)
    ax.set_title("08 — Hungarian Assignment: Pemilihan Pasangan Optimal\n"
                 "Kotak putih = pasangan terpilih  ·  Baris kuning = track hero",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "08_hungarian.png")


# ── 09 — Matched Update ───────────────────────────────────────────
def viz_09(frame, pred_mean, corr_mean, hero_id, hero_det_xyxy):
    labels = ["x", "y", "a", "h", "vx", "vy", "va", "vh"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(_rgb(frame))
    _box(axes[0], _tlbr(pred_mean), "royalblue", lw=2, ls="--",
         label="Prediksi KF  (sebelum update)")
    _box(axes[0], hero_det_xyxy, "gold", lw=2, ls=":",
         label="Deteksi YOLO  (input koreksi)")
    _box(axes[0], _tlbr(corr_mean), "limegreen", lw=2.5,
         label="Terkoreksi  (sesudah KF update + EMA)")
    axes[0].legend(loc="upper left", fontsize=8, facecolor="black", labelcolor="white")
    axes[0].set_title(f"Track #{hero_id}: Biru = prediksi  ·  Kuning = deteksi  ·  Hijau = hasil",
                      fontsize=FS)
    axes[0].axis("off")

    x = np.arange(8)
    w = 0.38
    axes[1].bar(x - w/2, pred_mean, w, label="State sebelum update (prediksi)",
                color="royalblue", alpha=0.78)
    axes[1].bar(x + w/2, corr_mean, w, label="State sesudah update (terkoreksi)",
                color="limegreen", alpha=0.78)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=FS)
    axes[1].set_ylabel("Nilai State", fontsize=FS)
    axes[1].set_title("State Vector: Sebelum vs Sesudah KF Update + EMA", fontsize=FS)
    axes[1].legend(fontsize=9)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("09 — Matched Update: KF Koreksi Posisi + EMA Perbarui Fitur",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "09_matched.png")


# ── 10 — Lifecycle ────────────────────────────────────────────────
def viz_10(hero_id, life_hist):
    hist = life_hist.get(hero_id, [])
    if len(hist) < 2:
        print(f"  10_lifecycle: Track #{hero_id} histori terlalu pendek, skip")
        return

    from my_strongsort.track import TrackState
    frames = [h["frame"] for h in hist]
    tsu = [h["time_since_update"] for h in hist]
    hits = [h["hits"] for h in hist]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].step(frames, hits, color="steelblue", lw=2, where="post",
                 label="Hits (jumlah total frame berhasil di-match)")
    axes[0].axhline(3, color="darkorange", ls="--", lw=1.5,
                    label="n_init = 3  (Tentative → Confirmed)")
    conf_frame = next((h["frame"] for h in hist
                       if h["state"] == TrackState.Confirmed), None)
    if conf_frame:
        axes[0].axvline(conf_frame, color="limegreen", ls=":", lw=2,
                        label=f"Confirmed mulai frame {conf_frame}")
    axes[0].set_ylabel("Hits", fontsize=FS)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].step(frames, tsu, color="tomato", lw=2, where="post",
                 label="time_since_update  (0 = di-match frame ini)")
    axes[1].axhline(30, color="darkred", ls="--", lw=1.5,
                    label="max_age = 30  (→ Deleted jika terlampaui)")
    axes[1].set_xlabel("Nomor Frame", fontsize=FS)
    axes[1].set_ylabel("Frame sejak\nterakhir di-match", fontsize=FS)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.suptitle(f"10 — Lifecycle Track #{hero_id}  "
                 f"(diamati selama {len(frames)} frame)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "10_lifecycle.png")


# ── 11 — Track Output ─────────────────────────────────────────────
def viz_11(frame, output_arr, hero_id, fid):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(_rgb(frame))

    hero_mot = None
    for row in output_arr:
        tid = int(row[4])
        if tid == hero_id:
            _box(ax, row[:4], "limegreen", lw=3)
            ax.text(row[0], row[1] - 8, f"ID: {tid}",
                    color="limegreen", fontsize=12, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, pad=3))
            hero_mot = row
        else:
            _box(ax, row[:4], "gray", lw=1, alpha=0.3)

    if hero_mot is not None:
        hero_arr = np.array([hero_mot])
        lines = to_mot_lines(fid, hero_arr)
        if lines:
            ax.text(0.01, 0.02,
                    f"Format MOTChallenge:\n{lines[0]}",
                    transform=ax.transAxes, fontsize=8.5,
                    color="white", va="bottom",
                    bbox=dict(facecolor="black", alpha=0.75, pad=4))

    ax.set_title(f"11 — Track Output  |  Hijau = Track Hero #{hero_id}  "
                 f"(frame {fid})",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _savefig(fig, "11_output.png")


# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dataset/videos/ramai.mp4")
    ap.add_argument("--yolo", default="models/7 - tuned/yolov5.pt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    print("[thesis_viz] Menjalankan pipeline...")
    from my_strongsort.demo import run_and_collect
    ss, rich, ema_hist, life_hist = run_and_collect(
        args.video, args.yolo, args.frames, args.imgsz, args.conf, args.iou)
    fid, frame, prev, last = rich

    # ── Pilih hero: matched + dalam appearance stage + histori terpanjang ──
    atids = last.get("appearance_track_ids", [])
    corrected = last.get("corrected_state", {})
    candidates = [tid for tid in atids if tid in corrected]
    if not candidates:
        candidates = list(corrected.keys())
    if not candidates:
        raise SystemExit("[thesis_viz] Tidak ada track matched. Naikkan --frames.")
    hero_id = max(candidates, key=lambda k: len(ema_hist.get(k, [])))
    print(f"[thesis_viz] Hero = Track #{hero_id}  |  "
          f"Histori EMA = {len(ema_hist.get(hero_id, []))} frame  |  "
          f"Frame kaya dipilih = {fid}")

    matches_local = last.get("appearance_matches_local", [])
    adids = last["appearance_det_ids"]
    hero_row = atids.index(hero_id) if hero_id in atids else 0
    hero_col = next((c for r, c in matches_local if r == hero_row), None)

    hero_det_orig = adids[hero_col] if hero_col is not None else None
    hero_det_obj = (next((d for d in last["detections"] if d.det_ind == hero_det_orig), None)
                    if hero_det_orig is not None else None)

    if hero_det_obj is not None:
        tl = hero_det_obj.tlwh
        hero_xyxy = [float(tl[0]), float(tl[1]),
                     float(tl[0] + tl[2]), float(tl[1] + tl[3])]
        hero_feat = hero_det_obj.feat
        hero_conf = float(hero_det_obj.conf)
    else:
        cm = last["corrected_state"][hero_id][0]
        hero_xyxy = _tlbr(cm)
        hero_feat = (last["features"][0] if len(last.get("features", [])) > 0
                     else np.zeros(512))
        hero_conf = 0.0

    pred_mean, pred_cov = last["predicted_state"][hero_id]
    corr_mean, _ = last["corrected_state"][hero_id]

    hero_feat_smooth = None
    for t in ss.tracks:
        if t.id == hero_id and t.features:
            hero_feat_smooth = np.asarray(t.features[-1], dtype=float)
            break

    os.makedirs(OUT, exist_ok=True)
    print(f"[thesis_viz] Menyimpan ke {OUT}/\n")

    viz_01(frame, last["xyxy"], hero_xyxy, hero_conf)

    if "warp_matrix" in last and prev is not None:
        viz_02(frame, prev, last["warp_matrix"], _tlbr(pred_mean))
    else:
        print("  02_ecc.png: skip (ECC belum aktif / frame pertama / tidak ada prev)")

    viz_03(frame, hero_xyxy, hero_feat)
    viz_04(frame, pred_mean, pred_cov, corr_mean, hero_xyxy)

    if hero_id in ema_hist and len(ema_hist[hero_id]) >= 2:
        viz_05(ema_hist[hero_id], hero_feat_smooth, ss.ema_alpha)
    else:
        print("  05_ema.png: skip (histori EMA terlalu pendek, naikkan --frames)")

    if last.get("maha") is not None and np.size(last["maha"]) > 0:
        viz_06(last["maha"], hero_row, adids)
        viz_07(last["appearance_cost"], last["maha"], last["fused_cost"],
               hero_row, matches_local, adids, ss.mc_lambda)
        viz_08(last["fused_cost"], matches_local, atids, adids, hero_id)
    else:
        print("  06/07/08: skip (tidak ada appearance matching di frame ini)")

    viz_09(frame, pred_mean, corr_mean, hero_id, hero_xyxy)
    viz_10(hero_id, life_hist)

    if np.size(last.get("output", [])) > 0:
        viz_11(frame, last["output"], hero_id, fid)
    else:
        print("  11_output.png: skip (tidak ada output confirmed track)")

    print(f"\n[thesis_viz] ✓ Selesai. Output di {OUT}/")
    print("  File: 01_detection.png  02_ecc.png  03_reid.png  04_kalman.png  05_ema.png")
    print("        06_gate.png  07_cost.png  08_hungarian.png  09_matched.png")
    print("        10_lifecycle.png  11_output.png")


if __name__ == "__main__":
    main()
