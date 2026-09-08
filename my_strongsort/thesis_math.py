"""thesis_math.py — Proses matematika tiap komponen StrongSORT dengan angka nyata.

Menghasilkan gambar PNG bergaya "halaman perhitungan skripsi":
rumus → matriks/vektor input dengan angka aktual → langkah operasi → output.

    venv/bin/python -m my_strongsort.thesis_math --frames 30

Output: outputs/my_strongsort/thesis_math/math_0X_*.png
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "outputs/my_strongsort/thesis_math"
KF_LBL = ["x", "y", "a", "h", "vx", "vy", "va", "vh"]
MEAS_LBL = ["x", "y", "a", "h"]
MAHA_THR = 9.4877


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def _savefig(fig, name):
    path = f"{OUT}/{name}"
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _noff(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def _arrow(ax, text="→", fs=18, color="#2c3e50"):
    _noff(ax)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=fs, color=color, transform=ax.transAxes, fontweight="bold")


def _formula(ax, formula, subtitle=None):
    """Kotak formula bergaris oranye."""
    _noff(ax)
    ax.text(0.5, 0.62, formula, ha="center", va="center",
            fontsize=11, color="#c0392b", fontweight="bold",
            fontfamily="monospace", transform=ax.transAxes,
            bbox=dict(facecolor="#fff9e6", boxstyle="round,pad=0.5",
                      edgecolor="#e67e22", linewidth=1.5))
    if subtitle:
        ax.text(0.5, 0.18, subtitle, ha="center", va="center",
                fontsize=9, color="#555", style="italic", transform=ax.transAxes)


def _vec_table(ax, vec, labels, title, highlight=None, dec=3, fs=9):
    """Vektor sebagai 1-baris tabel dengan header label."""
    ax.axis("off")
    v = np.asarray(vec, dtype=float)
    n = len(v)
    cells = [[f"{x:+.{dec}f}" for x in v]]

    tbl = ax.table(cellText=cells, colLabels=labels[:n],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="w", fontweight="bold")
        else:
            if highlight and c in highlight:
                cell.set_facecolor("#f39c12"); cell.set_text_props(fontweight="bold")
            else:
                val = float(cell.get_text().get_text().replace("+", ""))
                cell.set_facecolor("#d5f5e3" if val >= 0 else "#fde8e8")

    ax.set_title(title, fontsize=fs + 0.5, fontweight="bold", pad=3)


def _mat_table(ax, mat, rlbls, clbls, title, hi_cells=None, dec=3, fs=8):
    """Matriks sebagai tabel warna (biru=besar, putih=kecil)."""
    ax.axis("off")
    M = np.asarray(mat, dtype=float)
    rows, cols = M.shape
    cells = [[f"{M[r,c]:+.{dec}f}" for c in range(cols)] for r in range(rows)]
    mx = np.abs(M).max() + 1e-12

    tbl = ax.table(cellText=cells, rowLabels=rlbls,
                   colLabels=clbls, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    tbl.scale(1, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor("#1a252f"); cell.set_text_props(color="w", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="w")
        else:
            if hi_cells and (r - 1, c) in hi_cells:
                cell.set_facecolor("#e74c3c"); cell.set_text_props(color="w", fontweight="bold")
            else:
                inten = abs(M[r - 1, c]) / mx
                if inten > 0.7:
                    cell.set_facecolor("#2980b9"); cell.set_text_props(color="w")
                elif inten > 0.35:
                    cell.set_facecolor("#aed6f1")
                else:
                    cell.set_facecolor("#eaf4fb")

    ax.set_title(title, fontsize=fs + 0.5, fontweight="bold", pad=2)


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=13, fontweight="bold", y=0.97,
                 bbox=dict(facecolor="#1a252f", alpha=1, pad=6), color="white")


# ════════════════════════════════════════════════════════════════
#  01 — OSNet ReID
# ════════════════════════════════════════════════════════════════

def math_01_reid(frame, hero_xyxy, hero_feat):
    """Input: crop tensor H×W×3  →  Output: vektor fitur 512-D (L2-norm)."""
    x1, y1, x2, y2 = [int(v) for v in hero_xyxy]
    crop = frame[max(0, y1):max(y2, y1+2), max(0, x1):max(x2, x1+2)]
    crop_rgb = _rgb(cv2.resize(crop, (128, 256)) if crop.size > 0
                    else np.zeros((256, 128, 3), np.uint8))
    feat = np.asarray(hero_feat, dtype=float)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7),
                             gridspec_kw={"height_ratios": [0.28, 1], "wspace": 0.5, "hspace": 0.45})
    _suptitle(fig, "01 — OSNet ReID: Proses Ekstraksi Fitur Appearance")

    # Baris atas: alur proses
    info = [
        ("Input", f"Crop objek\n{(y2-y1)}×{(x2-x1)}×3\n(H×W×channel RGB)"),
        ("", "→"),
        ("Proses", "OSNet CNN\nforward pass\n(tensor 3D → 512D)"),
        ("", "→"),
        ("Output", f"Vektor fitur\n512 dimensi\n(L2-ternormalisasi)"),
    ]
    for ax, (lbl, txt) in zip(axes[0], info):
        _noff(ax)
        color = "#2980b9" if lbl in ("Input", "Output") else "#e67e22" if lbl == "Proses" else "black"
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=10,
                color="white" if lbl else color, transform=ax.transAxes,
                bbox=(dict(facecolor=color, pad=5, boxstyle="round") if lbl else None))

    # Baris bawah: konten detail
    # 1. Crop image
    axes[1, 0].imshow(crop_rgb)
    axes[1, 0].set_title(f"Crop objek hero\n({y2-y1}×{x2-x1} piksel)", fontsize=9)
    axes[1, 0].axis("off")

    # 2. Konsep tensor
    _noff(axes[1, 1])
    axes[1, 1].text(0.5, 0.55,
                    f"Tensor Input:\nshape = ({y2-y1}, {x2-x1}, 3)\n\n"
                    "Setiap piksel = 3 nilai\n(R, G, B : 0–255)",
                    ha="center", va="center", fontsize=9, transform=axes[1, 1].transAxes,
                    bbox=dict(facecolor="#eaf4fb", pad=5, boxstyle="round"))

    # 3. Arrow
    _arrow(axes[1, 2])

    # 4. Bar chart 16 dim pertama
    n_show = 16
    ax_f = axes[1, 3]
    colors_bar = ["#2980b9" if v >= 0 else "#e74c3c" for v in feat[:n_show]]
    bars = ax_f.bar(np.arange(n_show), feat[:n_show], color=colors_bar,
                    edgecolor="black", lw=0.3, alpha=0.88)
    for bar, v in zip(bars, feat[:n_show]):
        ax_f.text(bar.get_x() + bar.get_width() / 2,
                  v + (0.008 if v >= 0 else -0.025),
                  f"{v:.3f}", ha="center",
                  va="bottom" if v >= 0 else "top", fontsize=6, rotation=90)
    ax_f.axhline(0, color="black", lw=0.5)
    ax_f.set_xlabel("Indeks dimensi (0–15 dari 512)", fontsize=8)
    ax_f.set_ylabel("Nilai aktivasi", fontsize=8)
    ax_f.set_title(f"16 Dimensi Pertama Fitur 512-D", fontsize=9, fontweight="bold")
    ax_f.grid(True, axis="y", alpha=0.3)

    # 5. Normalisasi L2
    n2 = float(np.linalg.norm(feat))
    _noff(axes[1, 4])
    axes[1, 4].text(0.5, 0.65,
                    f"L2 Normalisasi:\n\nf  ←  f / ‖f‖₂\n\n‖f‖₂ = {n2:.6f}\n≈ 1.0000",
                    ha="center", va="center", fontsize=10, transform=axes[1, 4].transAxes,
                    bbox=dict(facecolor="#fff9e6", pad=8, boxstyle="round",
                              edgecolor="#e67e22", linewidth=1.5),
                    color="#c0392b", fontfamily="monospace", fontweight="bold")
    axes[1, 4].text(0.5, 0.15,
                    "Tujuan: semua fitur\ndibandingkan dalam\nskala yang sama",
                    ha="center", va="center", fontsize=8, transform=axes[1, 4].transAxes,
                    color="#555", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_01_reid.png")


# ════════════════════════════════════════════════════════════════
#  02 — ECC Camera Compensation
# ════════════════════════════════════════════════════════════════

def math_02_ecc(pred_mean, warp_matrix):
    """Matriks warp 2×3 → update posisi track (cx, cy)."""
    W = np.asarray(warp_matrix, dtype=float)
    dx, dy = float(W[0, 2]), float(W[1, 2])
    cx_before = float(pred_mean[0]) - dx   # approximate "before" camera_update
    cy_before = float(pred_mean[1]) - dy

    fig, axes = plt.subplots(2, 5, figsize=(16, 7),
                             gridspec_kw={"height_ratios": [0.3, 1], "wspace": 0.5, "hspace": 0.5})
    _suptitle(fig, "02 — ECC: Kompensasi Gerak Kamera  |  Enhanced Correlation Coefficient")

    # Baris atas: alur
    for ax, txt in zip(axes[0], [
        "Frame t-1\n(grayscale, 15% ukuran)",
        "→",
        "cv2.findTransformECC\n(cari transformasi W)",
        "→",
        "Update posisi track\n(geser cx, cy)"
    ]):
        _noff(ax)
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=10,
                transform=ax.transAxes)

    # Baris bawah

    # Matriks W (2×3)
    _mat_table(axes[1, 0], W,
               ["baris 0", "baris 1"],
               ["col 0\n(sx)", "col 1\n(shy)", "col 2\n(dx/dy)"],
               "Matriks Warp W (2×3)",
               hi_cells={(0, 2), (1, 2)},  # highlight dx, dy
               dec=4, fs=9)

    # Arti setiap elemen
    _noff(axes[1, 1])
    axes[1, 1].text(0.5, 0.55,
                    "W[0,0] = skala x  ≈ 1\n"
                    "W[1,1] = skala y  ≈ 1\n"
                    "W[0,1] = shear   ≈ 0\n"
                    "W[1,0] = shear   ≈ 0\n\n"
                    f"W[0,2] = dx = {dx:+.3f} px\n"
                    f"W[1,2] = dy = {dy:+.3f} px",
                    ha="center", va="center", fontsize=9.5,
                    fontfamily="monospace", transform=axes[1, 1].transAxes,
                    bbox=dict(facecolor="#eaf4fb", pad=6, boxstyle="round"))

    # Formula
    _formula(axes[1, 2],
             "cx ← cx + W[0,2]\ncy ← cy + W[1,2]",
             "Hanya translasi (dx, dy)\nyang diaplikasikan ke posisi track")

    # State sebelum
    v_before = np.array([cx_before, cy_before, pred_mean[2], pred_mean[3],
                          pred_mean[4], pred_mean[5], pred_mean[6], pred_mean[7]])
    _vec_table(axes[1, 3], v_before, KF_LBL,
               "State Sebelum (t-1)", highlight=[0, 1], dec=2)

    # State sesudah
    v_after = np.array([float(pred_mean[0]), float(pred_mean[1]),
                         pred_mean[2], pred_mean[3],
                         pred_mean[4], pred_mean[5], pred_mean[6], pred_mean[7]])
    _vec_table(axes[1, 4], v_after, KF_LBL,
               "State Sesudah (t)\nKuning = nilai berubah", highlight=[0, 1], dec=2)

    # Annotation perubahan
    axes[1, 4].text(0.5, 0.02,
                    f"Δcx = {dx:+.3f} px  |  Δcy = {dy:+.3f} px\n"
                    "Kecepatan [vx,vy,...] tidak berubah",
                    ha="center", va="bottom", fontsize=8, transform=axes[1, 4].transAxes,
                    color="#c0392b", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_02_ecc.png")


# ════════════════════════════════════════════════════════════════
#  03 — EMA
# ════════════════════════════════════════════════════════════════

def math_03_ema(raw_feats, alpha=0.9):
    """f_smooth = α·f_prev + (1-α)·f_new → L2 normalize."""
    if len(raw_feats) < 2:
        print("  math_03_ema: skip (histori pendek)"); return

    f_prev = np.asarray(raw_feats[-2], dtype=float)
    f_prev = f_prev / (np.linalg.norm(f_prev) + 1e-12)
    f_new_raw = np.asarray(raw_feats[-1], dtype=float)
    f_new = f_new_raw / (np.linalg.norm(f_new_raw) + 1e-12)

    f_weighted = alpha * f_prev + (1 - alpha) * f_new
    norm_val = float(np.linalg.norm(f_weighted))
    f_smooth = f_weighted / norm_val

    n = 8  # tampilkan 8 dimensi pertama untuk kejelasan

    fig, axes = plt.subplots(2, 6, figsize=(18, 7),
                             gridspec_kw={"height_ratios": [0.28, 1],
                                          "wspace": 0.55, "hspace": 0.45})
    _suptitle(fig, "03 — EMA: Exponential Moving Average Penghalusan Fitur Appearance")

    # Baris atas: alur
    for ax, txt in zip(axes[0], [
        f"f_prev\n(fitur smooth\nframe t-1)",
        f"f_new\n(fitur mentah\nframe t, L2-norm)",
        "→",
        f"Perkalian bobot\nα={alpha} dan {1-alpha}",
        "→",
        "Hasil smooth\n(L2-normalisasi)"
    ]):
        _noff(ax); ax.text(0.5, 0.5, txt, ha="center", va="center",
                           fontsize=9.5, transform=ax.transAxes)

    # Baris bawah
    lbl_n = [str(i) for i in range(n)]

    # f_prev (8 dims)
    _vec_table(axes[1, 0], f_prev[:n], lbl_n, f"f_prev (dim 0–{n-1})", dec=4, fs=7.5)

    # f_new (8 dims)
    _vec_table(axes[1, 1], f_new[:n], lbl_n, f"f_new  (dim 0–{n-1})", dec=4, fs=7.5)

    # Formula
    _formula(axes[1, 2],
             f"f_w = {alpha}·f_prev\n     + {1-alpha}·f_new",
             f"(element-wise, seluruh 512 dimensi)\nα = {alpha}  →  prev lebih dominan")

    # Contoh perhitungan 4 dimensi pertama
    _noff(axes[1, 3])
    lines = ["Contoh (dim 0–3):"]
    for i in range(4):
        pv = f_prev[i]; nv = f_new[i]; wv = f_weighted[i]
        lines.append(f"dim {i}: {alpha}×({pv:+.4f})\n"
                     f"      + {1-alpha}×({nv:+.4f})\n"
                     f"      = {wv:+.4f}")
    axes[1, 3].text(0.5, 0.98, "\n".join(lines),
                    ha="center", va="top", fontsize=8.5,
                    fontfamily="monospace", transform=axes[1, 3].transAxes,
                    bbox=dict(facecolor="#f0f0f0", pad=5, boxstyle="round"))

    # Normalisasi L2
    _noff(axes[1, 4])
    axes[1, 4].text(0.5, 0.55,
                    f"‖f_w‖₂ = {norm_val:.6f}\n\n"
                    "f_smooth = f_w / ‖f_w‖₂\n\n"
                    "→ ‖f_smooth‖₂ = 1.0000",
                    ha="center", va="center", fontsize=10,
                    fontfamily="monospace", transform=axes[1, 4].transAxes,
                    bbox=dict(facecolor="#fff9e6", pad=6, boxstyle="round",
                              edgecolor="#e67e22", linewidth=1.5),
                    color="#c0392b", fontweight="bold")

    # f_smooth (8 dims)
    _vec_table(axes[1, 5], f_smooth[:n], lbl_n, f"f_smooth (dim 0–{n-1})", dec=4, fs=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_03_ema.png")


# ════════════════════════════════════════════════════════════════
#  04 — NSA Kalman Filter: Predict
# ════════════════════════════════════════════════════════════════

def math_04_kalman_predict(pred_mean, pred_mean_before):
    """x_pred = F @ x  (constant-velocity model)."""
    # pred_mean_before = approximate state before KF predict
    # (we reconstruct from corr_mean of previous frame or use pred_mean)
    # For clarity: show the F matrix + formula + before/after
    ndim = 4
    F = np.eye(8)
    for i in range(ndim):
        F[i, ndim + i] = 1.0   # dt = 1

    x_before = np.asarray(pred_mean_before, dtype=float)
    x_after = F @ x_before

    fig, axes = plt.subplots(2, 5, figsize=(18, 8),
                             gridspec_kw={"height_ratios": [0.25, 1],
                                          "wspace": 0.5, "hspace": 0.4})
    _suptitle(fig, "04a — NSA Kalman Filter: Langkah Prediksi  |  x_pred = F · x")

    # Baris atas
    for ax, txt in zip(axes[0], [
        "State x\n(frame t-1)", "×", "Matriks F\n(8×8)", "=", "x_pred\n(frame t)"
    ]):
        _noff(ax); ax.text(0.5, 0.5, txt, ha="center", va="center",
                           fontsize=11, transform=ax.transAxes, fontweight="bold")

    # State before
    _vec_table(axes[1, 0], x_before, KF_LBL, "State x (t-1)\n[x, y, a, h, vx, vy, va, vh]",
               dec=2, fs=8)

    # Arrow
    _arrow(axes[1, 1], "×")

    # F matrix (8×8)
    _mat_table(axes[1, 2], F,
               [f"{l}" for l in KF_LBL],
               KF_LBL,
               "Matriks Transisi F (8×8)\n1 = ada pengaruh, 0 = tidak",
               hi_cells={(i, 4 + i) for i in range(4)},  # highlight dt=1 off-diagonal
               dec=0, fs=8)

    # Arrow
    _arrow(axes[1, 3], "=")

    # State after (with delta annotations)
    ax_r = axes[1, 4]
    _vec_table(ax_r, x_after, KF_LBL, "x_pred (t)\n[Kuning = nilai berubah]",
               highlight=[0, 1, 2, 3], dec=2, fs=8)
    # Annotation
    deltas = x_after - x_before
    note_lines = []
    pairs = [("x", 0, "vx", 4), ("y", 1, "vy", 5), ("a", 2, "va", 6), ("h", 3, "vh", 7)]
    for pos_n, pi, vel_n, vi in pairs:
        note_lines.append(f"{pos_n}: {x_before[pi]:.2f} + {x_before[vi]:+.2f} = {x_after[pi]:.2f}")
    ax_r.text(0.5, 0.02, "\n".join(note_lines),
              ha="center", va="bottom", fontsize=7.5, fontfamily="monospace",
              transform=ax_r.transAxes, color="#2c3e50",
              bbox=dict(facecolor="#f0f0f0", pad=3, boxstyle="round"))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_04a_kalman_predict.png")


# ════════════════════════════════════════════════════════════════
#  05 — NSA Kalman: NSA Noise Scaling + Update
# ════════════════════════════════════════════════════════════════

def math_05_kalman_update(pred_mean, corr_mean, kalman_extras, hero_conf, z_meas):
    """Update: innovation, Kalman gain, koreksi state."""
    K = np.asarray(kalman_extras["kalman_gain"], dtype=float)   # (8,4)
    innov = np.asarray(kalman_extras["innovation"], dtype=float)  # (4,)
    proj_mean = np.asarray(kalman_extras["projected_mean"], dtype=float)  # (4,) = H@x_pred
    proj_cov = np.asarray(kalman_extras["projected_cov"], dtype=float)    # (4,4) = S

    # NSA std computation
    h = float(pred_mean[3])
    sw_pos = 1.0 / 20
    std_base = [sw_pos * h, sw_pos * h, 1e-1, sw_pos * h]
    std_nsa = [(1 - hero_conf) * s for s in std_base]
    R_diag = [s ** 2 for s in std_nsa]

    fig, axes = plt.subplots(3, 5, figsize=(18, 11),
                             gridspec_kw={"height_ratios": [0.15, 1, 1],
                                          "wspace": 0.55, "hspace": 0.5})
    _suptitle(fig, "04b — NSA Kalman: NSA Noise Scaling + Update State")

    # Baris atas: judul bagian
    for ax, txt in zip(axes[0], ["NSA Noise", "Pengukuran z", "Innovation", "Kalman Gain K", "Koreksi"]):
        _noff(ax)
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=11,
                fontweight="bold", transform=ax.transAxes,
                bbox=dict(facecolor="#1a252f", pad=4, boxstyle="round"), color="white")

    # ── Baris tengah: NSA ──

    # 1. NSA noise scaling
    _noff(axes[1, 0])
    lines = [
        "Rk = (1 − conf) · Rk_base",
        "",
        f"conf = {hero_conf:.3f}",
        f"h = {h:.1f}  (tinggi objek)",
        f"sw_pos = 1/20 = {sw_pos:.3f}",
        "",
        "Rk_base std per dim:",
        f"  [x]: {sw_pos:.3f} × {h:.1f} = {std_base[0]:.3f}",
        f"  [y]: {sw_pos:.3f} × {h:.1f} = {std_base[1]:.3f}",
        f"  [a]: {1e-1:.3f}",
        f"  [h]: {sw_pos:.3f} × {h:.1f} = {std_base[3]:.3f}",
        "",
        f"NSA std = (1−{hero_conf:.2f}) × std:",
        f"  [x]: {std_nsa[0]:.4f}",
        f"  [y]: {std_nsa[1]:.4f}",
        f"  [a]: {std_nsa[2]:.4f}",
        f"  [h]: {std_nsa[3]:.4f}",
    ]
    axes[1, 0].text(0.5, 0.98, "\n".join(lines),
                    ha="center", va="top", fontsize=8.5,
                    fontfamily="monospace", transform=axes[1, 0].transAxes,
                    bbox=dict(facecolor="#f0f0f0", pad=4, boxstyle="round"))

    # 2. Pengukuran z vs proyeksi H·x_pred
    _noff(axes[1, 1])
    z = np.asarray(z_meas, dtype=float)
    lines2 = [
        "Pengukuran z dari deteksi YOLO:",
        f"  z[x] = {z[0]:.3f}",
        f"  z[y] = {z[1]:.3f}",
        f"  z[a] = {z[2]:.4f}  (w/h)",
        f"  z[h] = {z[3]:.3f}  (tinggi)",
        "",
        "Proyeksi state H·x_pred:",
        f"  [x] = {proj_mean[0]:.3f}",
        f"  [y] = {proj_mean[1]:.3f}",
        f"  [a] = {proj_mean[2]:.4f}",
        f"  [h] = {proj_mean[3]:.3f}",
        "",
        "(H = [I₄|0] → ambil\n4 elemen pertama\ndari state 8D)",
    ]
    axes[1, 1].text(0.5, 0.98, "\n".join(lines2),
                    ha="center", va="top", fontsize=8.5,
                    fontfamily="monospace", transform=axes[1, 1].transAxes,
                    bbox=dict(facecolor="#f0f0f0", pad=4, boxstyle="round"))

    # 3. Innovation
    _formula(axes[1, 2],
             "y = z − H·x_pred",
             "(seberapa jauh deteksi\n dari prediksi Kalman)")
    _vec_table(axes[2, 2], innov, MEAS_LBL,
               "Innovation y (4D)\n[x, y, a, h]", dec=3, fs=9)

    # 4. Kalman gain K (8×4 heatmap)
    im = axes[1, 3].imshow(K, cmap="RdBu_r", aspect="auto",
                            vmin=-abs(K).max(), vmax=abs(K).max())
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            axes[1, 3].text(j, i, f"{K[i,j]:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(K[i,j]) > abs(K).max()*0.5 else "black")
    axes[1, 3].set_xticks(range(4)); axes[1, 3].set_xticklabels(MEAS_LBL, fontsize=8)
    axes[1, 3].set_yticks(range(8)); axes[1, 3].set_yticklabels(KF_LBL, fontsize=8)
    axes[1, 3].set_title("Kalman Gain K (8×4)\n(seberapa besar koreksi tiap state)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.04)

    # 5. Formula koreksi
    _formula(axes[1, 4],
             "x_corr = x_pred\n       + K · y",
             "K·y = koreksi yang diterapkan\nke setiap elemen state")

    # ── Baris bawah ──

    # State pred vs corr perbandingan
    _vec_table(axes[2, 0], pred_mean, KF_LBL, "State Prediksi x_pred", dec=2, fs=8)
    _arrow(axes[2, 1], "+  K·y  =")

    # K·y vector
    ky = K @ innov
    _vec_table(axes[2, 3], ky, KF_LBL, "Koreksi K·y  (tambahan)", dec=3,
               highlight=[i for i, v in enumerate(ky) if abs(v) > abs(ky).max()*0.3], fs=8)

    _arrow(axes[2, 2], "→")

    _vec_table(axes[2, 4], corr_mean, KF_LBL,
               "State Terkoreksi x_corr", dec=2, fs=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_05_kalman_update.png")


# ════════════════════════════════════════════════════════════════
#  06 — Gate Mahalanobis
# ════════════════════════════════════════════════════════════════

def math_06_gate(kalman_extras, maha_hero_row, adids):
    """d² = y.T @ S⁻¹ @ y, dibanding threshold chi²(4, 0.95) = 9.4877."""
    innov = np.asarray(kalman_extras["innovation"], dtype=float)   # y (4,)
    S = np.asarray(kalman_extras["projected_cov"], dtype=float)    # 4×4
    S_inv = np.linalg.inv(S)
    d2_matched = float(innov @ S_inv @ innov)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8),
                             gridspec_kw={"height_ratios": [0.25, 1],
                                          "wspace": 0.55, "hspace": 0.45})
    _suptitle(fig, "05 — Gating Mahalanobis  |  d² = yᵀ · S⁻¹ · y  <  χ²(4, 0.95) = 9.4877")

    # Baris atas
    for ax, txt in zip(axes[0], [
        "Innovation y\n(4D)", "S = HPHᵀ+R\n(4×4)", "S⁻¹\n(invers S)", "d² = yᵀ·S⁻¹·y\n(skalar)", "Keputusan Gate"
    ]):
        _noff(ax); ax.text(0.5, 0.5, txt, ha="center", va="center",
                           fontsize=10, transform=ax.transAxes, fontweight="bold")

    # 1. Innovation
    _vec_table(axes[1, 0], innov, MEAS_LBL,
               "Innovation y\n= z − H·x_pred\n(4 dimensi)", dec=3, fs=9)

    # 2. S matrix (4×4)
    _mat_table(axes[1, 1], S, MEAS_LBL, MEAS_LBL,
               "S = HPHᵀ + R (4×4)\n(Innovation Covariance)", dec=2, fs=8)

    # 3. S inverse
    _mat_table(axes[1, 2], S_inv, MEAS_LBL, MEAS_LBL,
               "S⁻¹ (4×4)\n(Inverse S)", dec=4, fs=7.5)

    # 4. Perhitungan d² langkah demi langkah
    _noff(axes[1, 3])
    vy = S_inv @ innov  # S⁻¹·y
    lines = [
        "Langkah 1: S⁻¹ · y",
        f"  dim x: {vy[0]:+.4f}",
        f"  dim y: {vy[1]:+.4f}",
        f"  dim a: {vy[2]:+.4f}",
        f"  dim h: {vy[3]:+.4f}",
        "",
        "Langkah 2: yᵀ · (S⁻¹ · y)",
    ]
    terms = [float(innov[i] * vy[i]) for i in range(4)]
    for i, (lbl, t) in enumerate(zip(MEAS_LBL, terms)):
        lines.append(f"  {lbl}: {innov[i]:+.3f} × {vy[i]:+.4f} = {t:+.4f}")
    lines += ["", f"d² = Σ = {d2_matched:.4f}"]
    axes[1, 3].text(0.5, 0.98, "\n".join(lines),
                    ha="center", va="top", fontsize=8.5,
                    fontfamily="monospace", transform=axes[1, 3].transAxes,
                    bbox=dict(facecolor="#f0f0f0", pad=4, boxstyle="round"))

    # 5. Keputusan: semua deteksi
    n_det = len(maha_hero_row)
    ax_dec = axes[1, 4]
    colors = ["limegreen" if v <= MAHA_THR else "tomato" for v in maha_hero_row]
    bars = ax_dec.bar(range(n_det), maha_hero_row, color=colors, edgecolor="black", lw=0.5)
    ax_dec.axhline(MAHA_THR, color="red", lw=2, ls="--",
                   label=f"Threshold = {MAHA_THR}")
    ax_dec.axhline(d2_matched, color="blue", lw=1.5, ls=":",
                   label=f"Hero match = {d2_matched:.3f}")
    for i, (bar, v) in enumerate(zip(bars, maha_hero_row)):
        status = "✓" if v <= MAHA_THR else "✗"
        ax_dec.text(i, v + MAHA_THR * 0.05, f"d²={v:.1f}\n{status}",
                    ha="center", fontsize=7.5,
                    color="darkgreen" if v <= MAHA_THR else "darkred")
    ax_dec.set_xticks(range(n_det))
    ax_dec.set_xticklabels([f"Det {adids[i]}" for i in range(n_det)], fontsize=8)
    ax_dec.set_ylabel("Mahalanobis² d²", fontsize=8)
    ax_dec.set_title("Keputusan Gate per Deteksi\n✓ = lolos  ·  ✗ = dibuang", fontsize=9, fontweight="bold")
    ax_dec.legend(fontsize=7.5)
    ax_dec.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_06_gate.png")


# ════════════════════════════════════════════════════════════════
#  07 — Cost Matrix & Hungarian
# ════════════════════════════════════════════════════════════════

def math_07_cost(ap_row, maha_row, fused_row, gallery_feat, det_feat, matched_col, adids, mc_lambda=0.995):
    """Cosine → appearance cost → fused = mc_lambda·ap + (1-mc_lambda)·maha."""
    g = np.asarray(gallery_feat, dtype=float)
    d = np.asarray(det_feat, dtype=float)
    g = g / (np.linalg.norm(g) + 1e-12)
    d = d / (np.linalg.norm(d) + 1e-12)
    cos_sim = float(np.dot(g, d))
    ap_cost_val = 1 - cos_sim
    maha_val = float(maha_row[matched_col]) if matched_col is not None else 0.0
    fused_val = float(fused_row[matched_col]) if matched_col is not None else 0.0

    n_show = 12  # tampilkan n_show dims pertama

    fig, axes = plt.subplots(2, 5, figsize=(18, 8),
                             gridspec_kw={"height_ratios": [0.25, 1],
                                          "wspace": 0.55, "hspace": 0.45})
    _suptitle(fig, "06 — Cost Matrix: Appearance + Mahalanobis → Fused Cost")

    for ax, txt in zip(axes[0], [
        "Fitur gallery g\n(EMA smooth)", "Fitur deteksi d\n(OSNet raw)", "Cosine\nSimilarity",
        "Appearance\nCost", "Fused Cost"
    ]):
        _noff(ax); ax.text(0.5, 0.5, txt, ha="center", va="center",
                           fontsize=10, transform=ax.transAxes, fontweight="bold")

    # Baris bawah

    # 1. Gallery feature
    _vec_table(axes[1, 0], g[:n_show],
               [str(i) for i in range(n_show)],
               f"g: gallery fitur\n(dim 0–{n_show-1} dari 512)", dec=3, fs=7.5)
    axes[1, 0].text(0.5, 0.02, f"‖g‖₂ = {np.linalg.norm(g):.4f}",
                    ha="center", va="bottom", fontsize=8, transform=axes[1, 0].transAxes,
                    style="italic", color="#555")

    # 2. Detection feature
    _vec_table(axes[1, 1], d[:n_show],
               [str(i) for i in range(n_show)],
               f"d: fitur deteksi\n(dim 0–{n_show-1} dari 512)", dec=3, fs=7.5)
    axes[1, 1].text(0.5, 0.02, f"‖d‖₂ = {np.linalg.norm(d):.4f}",
                    ha="center", va="bottom", fontsize=8, transform=axes[1, 1].transAxes,
                    style="italic", color="#555")

    # 3. Cosine computation
    _noff(axes[1, 2])
    dot_val = float(np.dot(g[:8], d[:8]))
    lines_cos = [
        "g · d  (dot product):",
        "",
    ]
    for i in range(4):
        lines_cos.append(f"dim {i}: {g[i]:+.4f} × {d[i]:+.4f}")
        lines_cos.append(f"      = {g[i]*d[i]:+.4f}")
    lines_cos += [
        "  ...",
        f"Σ (512 dims) = {cos_sim:+.4f}",
        "",
        "cosine_sim = g · d",
        f"           = {cos_sim:.4f}",
        "",
        f"(range: -1.0 hingga +1.0)",
        f"(1.0 = identik)"
    ]
    axes[1, 2].text(0.5, 0.98, "\n".join(lines_cos),
                    ha="center", va="top", fontsize=8.5,
                    fontfamily="monospace", transform=axes[1, 2].transAxes,
                    bbox=dict(facecolor="#f0f0f0", pad=4, boxstyle="round"))

    # 4. Appearance cost
    _noff(axes[1, 3])
    n_det = len(ap_row)
    axes_inset = axes[1, 3]
    axes_inset.axis("off")
    axes_inset.text(0.5, 0.85,
                    "appearance_cost = 1 − cosine_sim",
                    ha="center", va="top", fontsize=10,
                    fontfamily="monospace", color="#c0392b", fontweight="bold",
                    transform=axes_inset.transAxes,
                    bbox=dict(facecolor="#fff9e6", pad=4, boxstyle="round",
                              edgecolor="#e67e22", linewidth=1.5))
    # bar per deteksi
    # Need inset axes
    ax2 = fig.add_axes([axes[1, 3].get_position().x0 + 0.01,
                        axes[1, 3].get_position().y0 + 0.0,
                        axes[1, 3].get_position().width - 0.02,
                        axes[1, 3].get_position().height * 0.6])
    colors_ap = ["limegreen" if i == matched_col else "#aed6f1" for i in range(n_det)]
    ax2.bar(range(n_det), ap_row, color=colors_ap, edgecolor="black", lw=0.5)
    for i, v in enumerate(ap_row):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold" if i == matched_col else "normal")
    ax2.set_xticks(range(n_det))
    ax2.set_xticklabels([f"Det {adids[j]}" for j in range(n_det)], fontsize=8)
    ax2.set_ylabel("Appearance cost", fontsize=8)
    ax2.set_ylim(0, min(1.1, ap_row.max() * 1.3))
    ax2.set_title("Hijau = terpilih", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # 5. Fused cost
    _noff(axes[1, 4])
    axes[1, 4].text(0.5, 0.90,
                    f"fused = {mc_lambda}·ap\n       + {1-mc_lambda:.3f}·maha",
                    ha="center", va="top", fontsize=10,
                    fontfamily="monospace", color="#c0392b", fontweight="bold",
                    transform=axes[1, 4].transAxes,
                    bbox=dict(facecolor="#fff9e6", pad=4, boxstyle="round",
                              edgecolor="#e67e22", linewidth=1.5))
    ax3 = fig.add_axes([axes[1, 4].get_position().x0 + 0.01,
                        axes[1, 4].get_position().y0 + 0.0,
                        axes[1, 4].get_position().width - 0.02,
                        axes[1, 4].get_position().height * 0.6])
    fu_disp = np.minimum(fused_row, 1.0)
    colors_f = ["limegreen" if i == matched_col else "#c39bd3" for i in range(n_det)]
    ax3.bar(range(n_det), fu_disp, color=colors_f, edgecolor="black", lw=0.5)
    for i, v in enumerate(fu_disp):
        ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold" if i == matched_col else "normal")
    ax3.set_xticks(range(n_det))
    ax3.set_xticklabels([f"Det {adids[j]}" for j in range(n_det)], fontsize=8)
    ax3.set_ylabel("Fused cost", fontsize=8)
    ax3.set_ylim(0, min(1.1, fu_disp.max() * 1.3))
    ax3.set_title("Hijau = terpilih (minimum)", fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_07_cost.png")


# ════════════════════════════════════════════════════════════════
#  08 — New Track: Inisiasi
# ════════════════════════════════════════════════════════════════

def math_08_new_track(z_first):
    """mean = [z, 0,0,0,0], cov = diag(std²) dari deteksi pertama."""
    z = np.asarray(z_first, dtype=float)
    sw_pos = 1.0 / 20
    sw_vel = 1.0 / 160
    h = z[3]
    mean = np.r_[z, np.zeros(4)]
    std = np.array([
        2 * sw_pos * h, 2 * sw_pos * h, 1e-2, 2 * sw_pos * h,
        10 * sw_vel * h, 10 * sw_vel * h, 1e-5, 10 * sw_vel * h
    ])
    cov = np.diag(std ** 2)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [0.25, 1],
                                          "wspace": 0.55, "hspace": 0.45})
    _suptitle(fig, "08 — New Track: Inisiasi Track Baru (Tentative)")

    for ax, txt in zip(axes[0], [
        "Deteksi pertama z\n(4D: cx,cy,a,h)", "→", "State mean\n(8D: z + [0,0,0,0])", "Covariance awal\n(8×8 diagonal)"
    ]):
        _noff(ax); ax.text(0.5, 0.5, txt, ha="center", va="center",
                           fontsize=11, transform=ax.transAxes, fontweight="bold")

    # 1. Detection z
    _vec_table(axes[1, 0], z, MEAS_LBL,
               "Deteksi pertama z\n[cx, cy, aspect, height]", dec=3, fs=9)

    # Arrow
    _arrow(axes[1, 1])

    # 2. Initial mean
    _vec_table(axes[1, 2], mean, KF_LBL,
               "State mean awal (8D)\n[Kuning = dari deteksi, putih = 0 (berhenti)]",
               highlight=[0, 1, 2, 3], dec=3, fs=8)
    axes[1, 2].text(0.5, 0.02,
                    "Kecepatan awal = 0\n(track baru diasumsikan berhenti)",
                    ha="center", va="bottom", fontsize=8.5,
                    transform=axes[1, 2].transAxes, style="italic", color="#555")

    # 3. Covariance (show diagonal + formula)
    _noff(axes[1, 3])
    lines_cov = [
        "cov = diag(std²)",
        "",
        "std = [2·sw_p·h, 2·sw_p·h,",
        "       1e-2, 2·sw_p·h,",
        "       10·sw_v·h, 10·sw_v·h,",
        "       1e-5, 10·sw_v·h]",
        "",
        f"sw_p = 1/20 = {sw_pos:.4f}",
        f"sw_v = 1/160 = {sw_vel:.4f}",
        f"h = {h:.2f}",
        "",
        "std aktual (diagonal cov):",
    ]
    for lbl, s in zip(KF_LBL, std):
        lines_cov.append(f"  [{lbl}]: std={s:.4f}  var={s**2:.6f}")
    axes[1, 3].text(0.5, 0.98, "\n".join(lines_cov),
                    ha="center", va="top", fontsize=8.2,
                    fontfamily="monospace", transform=axes[1, 3].transAxes,
                    bbox=dict(facecolor="#f0f0f0", pad=4, boxstyle="round"))
    axes[1, 3].text(0.5, 0.02,
                    "Catatan: std_pos > std_vel\n"
                    "→ lebih yakin pada kecepatan awal (0)\n"
                    "daripada posisi absolut",
                    ha="center", va="bottom", fontsize=8.5,
                    transform=axes[1, 3].transAxes, style="italic", color="#c0392b")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "math_08_new_track.png")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dataset/videos/ramai.mp4")
    ap.add_argument("--yolo", default="models/7 - tuned/yolov8.pt")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    print("[thesis_math] Menjalankan pipeline...")
    from my_strongsort.demo import run_and_collect
    ss, rich, ema_hist, life_hist = run_and_collect(
        args.video, args.yolo, args.frames, args.imgsz, args.conf, args.iou)
    fid, frame, prev, last = rich

    # Pilih hero
    atids = last.get("appearance_track_ids", [])
    corrected = last.get("corrected_state", {})
    candidates = [t for t in atids if t in corrected]
    if not candidates:
        candidates = list(corrected.keys())
    if not candidates:
        raise SystemExit("Tidak ada track matched.")
    hero_id = max(candidates, key=lambda k: len(ema_hist.get(k, [])))

    # Ekstrak data hero
    matches_local = last.get("appearance_matches_local", [])
    adids = last["appearance_det_ids"]
    hero_row = atids.index(hero_id)
    hero_col = next((c for r, c in matches_local if r == hero_row), None)
    hero_det_orig = adids[hero_col] if hero_col is not None else None
    hero_det_obj = (next((d for d in last["detections"] if d.det_ind == hero_det_orig), None)
                    if hero_det_orig is not None else None)

    if hero_det_obj is None:
        raise SystemExit("Hero track tidak ditemukan dalam deteksi. Naikkan --frames.")

    tl = hero_det_obj.tlwh
    hero_xyxy = [float(tl[0]), float(tl[1]), float(tl[0]+tl[2]), float(tl[1]+tl[3])]
    hero_feat = hero_det_obj.feat
    hero_conf = float(hero_det_obj.conf)
    z_meas = hero_det_obj.to_xyah()  # (cx, cy, a, h) measurement

    pred_mean, pred_cov = last["predicted_state"][hero_id]
    corr_mean, _ = last["corrected_state"][hero_id]
    kalman_extras = last["kalman_extras"][hero_id]

    # Gallery feature (best EMA smooth)
    gallery_feat = None
    for t in ss.tracks:
        if t.id == hero_id and t.features:
            gallery_feat = np.asarray(t.features[-1], dtype=float)
            break
    if gallery_feat is None:
        gallery_feat = hero_feat

    os.makedirs(OUT, exist_ok=True)
    print(f"[thesis_math] Hero = Track #{hero_id}  |  frame = {fid}\n"
          f"              conf = {hero_conf:.3f}  |  z = {np.round(z_meas, 2)}\n"
          f"              Menyimpan ke {OUT}/\n")

    math_01_reid(frame, hero_xyxy, hero_feat)
    if "warp_matrix" in last:
        math_02_ecc(pred_mean, last["warp_matrix"])
    else:
        print("  math_02_ecc: skip (ECC belum aktif)")
    if hero_id in ema_hist and len(ema_hist[hero_id]) >= 2:
        math_03_ema(ema_hist[hero_id])
    else:
        print("  math_03_ema: skip (histori pendek)")

    # Perkirakan state sebelum KF predict: x_before = x_after dikurangi kecepatan
    x_before = pred_mean.copy()
    x_before[:4] = pred_mean[:4] - pred_mean[4:]   # x_t-1 ≈ x_pred - velocity
    math_04_kalman_predict(pred_mean, x_before)
    math_05_kalman_update(pred_mean, corr_mean, kalman_extras, hero_conf, z_meas)

    if last.get("maha") is not None and hero_row < last["maha"].shape[0]:
        math_06_gate(kalman_extras, last["maha"][hero_row], adids)

    if last.get("appearance_cost") is not None and hero_row < last["appearance_cost"].shape[0]:
        math_07_cost(
            last["appearance_cost"][hero_row],
            last["maha"][hero_row],
            last["fused_cost"][hero_row],
            gallery_feat, hero_feat,
            hero_col, adids, ss.mc_lambda)

    # New track: gunakan pengukuran pertama dari histori (jika ada), atau z_meas saat ini
    first_z = z_meas  # gunakan pengukuran hero sebagai contoh inisiasi
    math_08_new_track(first_z)

    print(f"\n[thesis_math] ✓ Selesai. Output di {OUT}/")
    print("  math_01_reid  math_02_ecc  math_03_ema")
    print("  math_04a_kalman_predict  math_05_kalman_update")
    print("  math_06_gate  math_07_cost  math_08_new_track")


if __name__ == "__main__":
    main()
