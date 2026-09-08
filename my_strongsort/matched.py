"""Komponen 8 — Matched Detection & Track (update track ter-match).

Untuk tiap pasangan track↔deteksi hasil Hungarian: koreksi Kalman (NSA),
haluskan fitur (EMA), tambah hits & reset time_since_update, dan promosikan
Tentative→Confirmed bila hits>=n_init. Mirror boxmot Track.update.
"""
import os

import numpy as np

from my_strongsort import ema, viz
from my_strongsort.track import TrackState


def update_track(track, detection):
    """Terapkan update ke `track` dari `detection`. Return extras Kalman (utk viz)."""
    track.bbox = detection.to_xyah()
    track.conf = detection.conf
    track.cls = detection.cls
    track.det_ind = detection.det_ind

    track.mean, track.covariance, extras = track.kf.update(
        track.mean, track.covariance, track.bbox, track.conf
    )
    track.features = [ema.smooth(track.features[-1], detection.feat, track.ema_alpha)]

    track.hits += 1
    track.time_since_update = 0
    if track.state == TrackState.Tentative and track.hits >= track._n_init:
        track.state = TrackState.Confirmed
    return extras


def visualize(img, before_xyxy, after_xyxy, ids, out_dir, tag="frame"):
    """Gambar box prediksi (biru, sebelum update) vs terkoreksi (hijau, sesudah)."""
    os.makedirs(out_dir, exist_ok=True)
    out = viz.draw_boxes(img, before_xyxy, labels=[f"pred {i}" for i in ids], color=(255, 0, 0))
    out = viz.draw_boxes(out, after_xyxy, labels=[f"upd {i}" for i in ids], color=(0, 200, 0))
    viz.save_image(out, f"{out_dir}/{tag}_matched_update.png")
    return out_dir


def demo():
    from my_strongsort.track import Detection, Track
    feat = np.ones(8) / np.sqrt(8)
    trk = Track(Detection([100, 200, 40, 80], 0.9, 0, 0, feat), 1, n_init=2, max_age=30, ema_alpha=0.9)
    assert trk.state == TrackState.Tentative
    trk.predict()
    new_det = Detection([106, 203, 40, 80], 0.95, 0, 1, feat)
    update_track(trk, new_det)
    assert trk.hits == 2 and trk.time_since_update == 0
    assert trk.state == TrackState.Confirmed, "hits>=n_init harus konfirmasi track"
    assert abs(np.linalg.norm(trk.features[-1]) - 1.0) < 1e-9
    print("matched.py OK — state", trk.state, "| center", np.round(trk.mean[:2], 2))


if __name__ == "__main__":
    demo()
