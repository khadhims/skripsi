"""Komponen 1 — OSNet ReID (ekstraksi fitur appearance).

Hybrid: jaringan OSNet TIDAK ditulis ulang (itu CNN terlatih). Kita pakai bobot
terlatih `strongsort/ReID/osnet_x0_25_msmt17.pt` lewat backend boxmot. Modul ini
membungkusnya jadi API sederhana `extract(xyxy, img) -> (N, D)` fitur unit-norm,
plus visualisasi crop, matrix fitur, dan matrix cosine-similarity antar deteksi.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from boxmot.reid.core import ReID as _BoxmotReID

from my_strongsort import viz

DEFAULT_WEIGHTS = "strongsort/ReID/osnet_x0_25_msmt17.pt"


class OSNetReID:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="cpu", half=False):
        self.model = _BoxmotReID(
            weights=Path(weights), device=torch.device(device), half=half
        ).model

    def extract(self, xyxy, img):
        """xyxy: (N,4) box; img: BGR frame. Return fitur (N,D) sudah L2-normalisasi."""
        xyxy = np.asarray(xyxy, dtype=np.float32)
        if len(xyxy) == 0:
            return np.empty((0, 512), dtype=np.float32)
        return self.model.get_features(xyxy, img)


def _crop(img, box, size=(128, 256)):
    x1, y1, x2, y2 = [int(max(0, v)) for v in box[:4]]
    x2 = max(x2, x1 + 1); y2 = max(y2, y1 + 1)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((size[1], size[0], 3), dtype=img.dtype)
    return cv2.resize(crop, size)


def visualize(xyxy, img, feats, out_dir, tag="frame"):
    """Simpan montage crop, heatmap matrix fitur (N×D), dan cosine-sim antar deteksi."""
    os.makedirs(out_dir, exist_ok=True)
    xyxy = np.atleast_2d(np.asarray(xyxy, dtype=float))
    feats = np.atleast_2d(np.asarray(feats, dtype=float))
    if len(xyxy):
        montage = np.hstack([_crop(img, b) for b in xyxy])
        viz.save_image(montage, f"{out_dir}/{tag}_crops.png")
    if feats.size:
        viz.save_matrix(feats, f"{out_dir}/{tag}_features")
        viz.heatmap(feats, f"{out_dir}/{tag}_features.png",
                    title=f"OSNet features N×D ({tag})", xlabel="feature dim", ylabel="detection",
                    annotate=False)
        f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        sim = f @ f.T
        viz.save_matrix(sim, f"{out_dir}/{tag}_cosine_sim")
        viz.heatmap(sim, f"{out_dir}/{tag}_cosine_sim.png",
                    title=f"cosine similarity antar deteksi ({tag})",
                    xlabel="detection", ylabel="detection", cmap="magma")
    return out_dir


def demo():
    """Integration check: butuh bobot ReID + torch. Ekstrak 2 box dari citra dummy."""
    reid = OSNetReID()
    img = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))).astype(np.uint8)
    boxes = np.array([[50, 60, 130, 240], [300, 100, 380, 300]], dtype=np.float32)
    feats = reid.extract(boxes, img)
    assert feats.shape[0] == 2, feats.shape
    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"fitur harus unit-norm, dapat {norms}"
    print("reid.py OK — feats", feats.shape, "| norms", np.round(norms, 4))


if __name__ == "__main__":
    demo()
