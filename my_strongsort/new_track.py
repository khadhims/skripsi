"""Komponen 9 — New Track (inisiasi track baru).

Setiap deteksi yang tidak ter-match membentuk track baru berstatus Tentative,
dengan state Kalman di-inisiasi dari box-nya. Track tetap Tentative sampai
terkonfirmasi (n_init hits berturut) — melindungi dari deteksi palsu sesaat.
Mirror boxmot Tracker._initiate_track.
"""
import os

import numpy as np

from my_strongsort import viz
from my_strongsort.track import Track


def create_track(detection, track_id, n_init, max_age, ema_alpha):
    return Track(detection, track_id, n_init, max_age, ema_alpha)


def visualize(img, tracks, out_dir, tag="frame"):
    """Gambar box track baru (kuning=Tentative) + simpan mean/cov awalnya."""
    os.makedirs(out_dir, exist_ok=True)
    boxes = [t.to_tlbr() for t in tracks]
    labels = [f"new#{t.id}" for t in tracks]
    out = viz.draw_boxes(img, boxes, labels=labels, color=(0, 255, 255))
    viz.save_image(out, f"{out_dir}/{tag}_new_tracks.png")
    for t in tracks:
        viz.save_matrix(t.mean, f"{out_dir}/{tag}_new{t.id}_mean")
        viz.save_matrix(t.covariance, f"{out_dir}/{tag}_new{t.id}_covariance")
    return out_dir


def demo():
    from my_strongsort.track import Detection, TrackState
    feat = np.ones(8) / np.sqrt(8)
    t1 = create_track(Detection([10, 10, 40, 80], 0.9, 0, 0, feat), 1, 3, 30, 0.9)
    t2 = create_track(Detection([200, 50, 30, 60], 0.8, 0, 1, feat), 2, 3, 30, 0.9)
    assert t1.id == 1 and t2.id == 2
    assert t1.state == TrackState.Tentative and t1.hits == 1
    assert t1.mean.shape == (8,) and t1.covariance.shape == (8, 8)
    print("new_track.py OK — ids", (t1.id, t2.id), "| center#1", np.round(t1.mean[:2], 1))


if __name__ == "__main__":
    demo()
