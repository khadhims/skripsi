"""Komponen 5 — Gate (Mahalanobis distance gating).

Menyaring pasangan track-deteksi yang secara gerak tidak mungkin. Untuk tiap
track dihitung squared Mahalanobis distance dari distribusi state (hasil Kalman)
ke tiap measurement deteksi; pasangan dengan jarak > ambang chi-square 0.95
(4 dof = 9.4877) dianggap infeasible. Mirror boxmot gating_distance + gate mask.

Output: matrix N×M jarak Mahalanobis + mask boolean gate.
"""
import os

import numpy as np

from my_strongsort import viz
from my_strongsort.kalman import chi2inv95

GATING_THRESHOLD = chi2inv95[4]  # 9.4877


def gating_distance_matrix(tracks, measurements):
    """N×M squared Mahalanobis: baris=track, kolom=measurement (format xyah)."""
    meas = np.atleast_2d(np.asarray(measurements, dtype=float))
    out = np.zeros((len(tracks), len(meas)))
    for i, t in enumerate(tracks):
        out[i, :] = t.kf.gating_distance(t.mean, t.covariance, meas)
    return out


def gate_mask(maha, threshold=GATING_THRESHOLD):
    """True = infeasible (jarak melewati ambang)."""
    return np.asarray(maha) > threshold


def visualize(maha, out_dir, tag="frame", track_ids=None, det_ids=None, threshold=GATING_THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)
    maha = np.atleast_2d(np.asarray(maha, dtype=float))
    viz.save_matrix(maha, f"{out_dir}/{tag}_mahalanobis")
    viz.heatmap(maha, f"{out_dir}/{tag}_mahalanobis.png",
                title=f"squared Mahalanobis (thr={threshold:.2f}, {tag})",
                xlabel="detections", ylabel="tracks",
                xticklabels=det_ids, yticklabels=track_ids)
    viz.heatmap(gate_mask(maha, threshold).astype(float), f"{out_dir}/{tag}_gate_mask.png",
                title=f"gate mask (1=infeasible, {tag})", xlabel="detections", ylabel="tracks",
                cmap="Reds", xticklabels=det_ids, yticklabels=track_ids)
    return out_dir


def demo():
    from my_strongsort.track import Detection, Track
    feat = np.ones(4) / 2
    det = Detection([100, 200, 40, 80], 0.9, 0, 0, feat)
    trk = Track(det, 1, 3, 30, 0.9)
    near = det.to_xyah()
    far = near + np.array([300.0, 300.0, 0.0, 0.0])
    maha = gating_distance_matrix([trk], np.vstack([near, far]))
    assert maha.shape == (1, 2)
    assert maha[0, 0] < GATING_THRESHOLD, "measurement dekat harus feasible"
    assert maha[0, 1] > GATING_THRESHOLD, "measurement jauh harus ter-gate"
    m = gate_mask(maha)
    assert not m[0, 0] and m[0, 1]
    print("gate.py OK — maha near/far =", np.round(maha[0], 3), "| thr", round(GATING_THRESHOLD, 3))


if __name__ == "__main__":
    demo()
