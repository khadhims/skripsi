"""Komponen 2 — ECC (Enhanced Correlation Coefficient), kompensasi gerak kamera.

Estimasi pergeseran global antar-frame via cv2.findTransformECC pada citra
grayscale yang di-downscale (scale=0.15), lalu translasi diskalakan balik ke
resolusi penuh. Output: matrix warp 2x3 (mode TRANSLATION), dipakai untuk
menggeser posisi tiap track (Track.camera_update). Mirror boxmot ECC.apply.
"""
import os

import cv2
import numpy as np

from my_strongsort import viz


class ECC:
    def __init__(self, scale=0.15, warp_mode=cv2.MOTION_TRANSLATION, eps=1e-5, max_iter=100):
        self.scale = float(scale)
        self.warp_mode = int(warp_mode)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iter), float(eps))
        self.prev = None

    def preprocess(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

    def apply(self, img):
        """Return matrix warp 2x3. Frame pertama / gagal konvergen -> identity."""
        warp = np.eye(2, 3, dtype=np.float32)
        if self.prev is None:
            self.prev = self.preprocess(img)
            return warp
        curr = self.preprocess(img)
        try:
            _, warp = cv2.findTransformECC(self.prev, curr, warp, self.warp_mode, self.criteria, None, 1) # type: ignore
        except cv2.error:
            self.prev = curr
            return warp
        if self.scale < 1.0:  # kembalikan translasi ke koordinat resolusi penuh
            warp = warp.copy()
            warp[0, 2] /= self.scale
            warp[1, 2] /= self.scale
        self.prev = curr
        return warp


def visualize(warp_matrix, out_dir, tag="frame", prev_img=None, curr_img=None):
    """Simpan matrix warp 2x3 (+ dx,dy) dan opsional overlay prev-vs-warped."""
    os.makedirs(out_dir, exist_ok=True)
    warp_matrix = np.asarray(warp_matrix, dtype=float)
    viz.save_matrix(warp_matrix, f"{out_dir}/{tag}_warp")
    viz.heatmap(warp_matrix, f"{out_dir}/{tag}_warp.png",
                title=f"ECC warp 2x3 dx={warp_matrix[0,2]:.2f} dy={warp_matrix[1,2]:.2f} ({tag})",
                xlabel="[a b tx / c d ty]", ylabel="row")
    if prev_img is not None and curr_img is not None:
        h, w = prev_img.shape[:2]
        warped = cv2.warpAffine(prev_img, warp_matrix.astype(np.float32), (w, h))
        overlay = cv2.addWeighted(curr_img, 0.5, warped, 0.5, 0)
        viz.save_image(overlay, f"{out_dir}/{tag}_overlay.png")
    return out_dir


def demo():
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    base = np.clip(128 + 60 * np.sin(xx / 40) + 40 * np.cos(yy / 30), 0, 255).astype(np.uint8)
    img1 = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    tx, ty = 8.0, -5.0
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    img2 = cv2.cvtColor(cv2.warpAffine(base, M, (640, 480)), cv2.COLOR_GRAY2BGR)

    ecc = ECC()
    w0 = ecc.apply(img1)
    assert np.allclose(w0, np.eye(2, 3)), "frame pertama harus identity"
    w1 = ecc.apply(img2)
    assert w1.shape == (2, 3)
    rec = (w1[0, 2], w1[1, 2])  # warp memetakan curr->prev, jadi ~ -(tx,ty)
    assert abs(abs(rec[0]) - tx) < 2 and abs(abs(rec[1]) - abs(ty)) < 2, f"translasi meleset: {rec}"
    print("ecc.py OK — warp dx,dy =", np.round(rec, 2))


if __name__ == "__main__":
    demo()
