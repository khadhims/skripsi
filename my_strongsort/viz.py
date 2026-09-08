"""Helper visualisasi bersama untuk semua komponen my_strongsort.

Boilerplate matplotlib + penyimpanan matrix ditaruh di satu tempat supaya tidak
diulang di 11 modul komponen. Dua helper utama:
    save_matrix(mat, path)  -> tulis .npy + .csv (nilai bisa dikutip di laporan)
    heatmap(mat, path)      -> simpan heatmap PNG berlabel
"""
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: aman dijalankan tanpa display / di server
import matplotlib.pyplot as plt


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_matrix(mat, path_no_ext, fmt="%.6f"):
    """Dump matrix sebagai .npy (presisi penuh) dan .csv (mudah dibaca/dikutip)."""
    mat = np.asarray(mat, dtype=float)
    ensure_dir(path_no_ext + ".npy")
    np.save(path_no_ext + ".npy", mat)
    np.savetxt(path_no_ext + ".csv", np.atleast_2d(mat), delimiter=",", fmt=fmt)
    return path_no_ext


def heatmap(mat, path, title="", xlabel="", ylabel="", cmap="viridis",
            annotate=None, xticklabels=None, yticklabels=None):
    """Simpan heatmap PNG dari matrix 2D, otomatis anotasi nilai bila kecil (<=144 sel)."""
    mat = np.atleast_2d(np.asarray(mat, dtype=float))
    ensure_dir(path)
    fig, ax = plt.subplots(figsize=(max(3.0, 0.6 * mat.shape[1] + 2),
                                    max(2.5, 0.6 * mat.shape[0] + 1.5)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate is None:
        annotate = mat.size <= 144
    if annotate and mat.size:
        thr = (np.nanmax(mat) + np.nanmin(mat)) / 2.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < thr else "black")
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels))); ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels))); ax.set_yticklabels(yticklabels)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def savefig(fig, path):
    ensure_dir(path)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def save_image(img_bgr, path):
    ensure_dir(path)
    cv2.imwrite(path, img_bgr)
    return path


def draw_boxes(img, boxes_xyxy, labels=None, color=(0, 255, 0), thickness=2):
    """Gambar box (xyxy) + label opsional pada salinan citra BGR."""
    out = img.copy()
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None:
            cv2.putText(out, str(labels[i]), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
