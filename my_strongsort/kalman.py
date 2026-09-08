"""Komponen 4 — NSA Kalman Filter (prediksi posisi).

State: [x, y, a, h, vx, vy, va, vh]  (a = w/h, konstan-velocity).
"NSA" (Noise Scale Adaptive, dari GIAOTracker): noise pengukuran diskala
`Rk = (1 - conf) * Rk` di project(), jadi deteksi ber-confidence tinggi lebih
dipercaya. Mirror boxmot KalmanFilterXYAH (kasus AABB, ndim=4).

Output yang bisa dijelaskan: mean 8-vektor, covariance 8x8, ellipse ketidakpastian
(x,y), box predicted vs corrected, Kalman gain, dan innovation.
"""
import os

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from my_strongsort import viz

# 0.95 quantile chi-square (dipakai sbagai ambang gating), lihat gate.py
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
             6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

STATE_LABELS = ["x", "y", "a", "h", "vx", "vy", "va", "vh"]


class NSAKalmanFilterXYAH:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)              # F (8x8), constant velocity
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)        # H (4x8)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """measurement=[x,y,a,h] -> (mean 8, covariance 8x8) untuk track baru."""
        measurement = np.asarray(measurement, dtype=float)
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        mean[2] = max(mean[2], 1e-4); mean[3] = max(mean[3], 1e-4)
        return mean, covariance

    def predict(self, mean, covariance):
        """Langkah prediksi: mean = F·mean, cov = F·cov·Fᵀ + Q."""
        std_pos = [self._std_weight_position * mean[3], self._std_weight_position * mean[3],
                   1e-2, self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3],
                   1e-5, self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        mean[2] = max(mean[2], 1e-4); mean[3] = max(mean[3], 1e-4)
        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        """Proyeksi state ke ruang ukur. NSA: std pengukuran diskala (1 - confidence)."""
        std = [self._std_weight_position * mean[3], self._std_weight_position * mean[3],
               1e-1, self._std_weight_position * mean[3]]
        std = [(1 - confidence) * x for x in std]           # <-- NSA
        innovation_cov = np.diag(np.square(std))
        pmean = self._update_mat @ mean
        pcov = self._update_mat @ covariance @ self._update_mat.T
        return pmean, pcov + innovation_cov

    def update(self, mean, covariance, measurement, confidence=0.0):
        """Langkah koreksi Kalman dengan noise pengukuran NSA. Return (mean, cov, extras)."""
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), (covariance @ self._update_mat.T).T, check_finite=False
        ).T
        innovation = np.asarray(measurement, dtype=float) - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        new_mean[2] = max(new_mean[2], 1e-4); new_mean[3] = max(new_mean[3], 1e-4)
        extras = {"kalman_gain": kalman_gain, "innovation": innovation,
                  "projected_mean": projected_mean, "projected_cov": projected_cov}
        return new_mean, new_covariance, extras

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Squared Mahalanobis distance tiap measurement ke distribusi state (untuk gate.py)."""
        mean, covariance = self.project(mean, covariance)   # confidence=0 -> noise penuh
        measurements = np.atleast_2d(np.asarray(measurements, dtype=float))
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True,
                                          check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)


def _draw_cov_ellipse(ax, center, cov2, n_std=2.0, **kw):
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(xy=center, width=w, height=h, angle=angle, fill=False, **kw))


def visualize(mean, covariance, out_dir, tag="track", extras=None, predicted=None):
    """Simpan output NSA Kalman: mean, covariance (matrix+heatmap), ellipse (x,y),
    dan bila `extras`/`predicted` diberikan: gain, innovation, predicted-vs-corrected.
    """
    os.makedirs(out_dir, exist_ok=True)
    mean = np.asarray(mean, dtype=float); covariance = np.asarray(covariance, dtype=float)
    viz.save_matrix(mean, f"{out_dir}/{tag}_mean")
    viz.save_matrix(covariance, f"{out_dir}/{tag}_covariance")
    viz.heatmap(covariance, f"{out_dir}/{tag}_covariance.png",
                title=f"NSA Kalman covariance 8x8 ({tag})", xlabel="state", ylabel="state",
                xticklabels=STATE_LABELS, yticklabels=STATE_LABELS)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(STATE_LABELS, mean)
    ax.set_title(f"NSA Kalman mean state ({tag})"); ax.grid(True, axis="y", alpha=0.3)
    viz.savefig(fig, f"{out_dir}/{tag}_mean.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean[0], mean[1], "ro", label="corrected (x,y)")
    _draw_cov_ellipse(ax, mean[:2], covariance[:2, :2], edgecolor="red", label="±2σ")
    if predicted is not None:
        pm, pc = predicted
        ax.plot(pm[0], pm[1], "b^", label="predicted (x,y)")
        _draw_cov_ellipse(ax, pm[:2], pc[:2, :2], edgecolor="blue", ls="--")
    ax.set_aspect("equal", adjustable="datalim"); ax.invert_yaxis()
    ax.set_title(f"posisi & ketidakpastian ({tag})"); ax.set_xlabel("x px"); ax.set_ylabel("y px")
    ax.legend(fontsize=8)
    viz.savefig(fig, f"{out_dir}/{tag}_position_ellipse.png")

    if extras is not None:
        viz.save_matrix(extras["kalman_gain"], f"{out_dir}/{tag}_kalman_gain")
        viz.save_matrix(extras["innovation"], f"{out_dir}/{tag}_innovation")
        viz.heatmap(extras["kalman_gain"], f"{out_dir}/{tag}_kalman_gain.png",
                    title=f"Kalman gain 8x4 ({tag})", xlabel="measurement", ylabel="state",
                    xticklabels=["x", "y", "a", "h"], yticklabels=STATE_LABELS)
    return out_dir


def demo():
    """Self-check: bentuk, sifat NSA, koreksi menarik ke measurement, gating≈0."""
    kf = NSAKalmanFilterXYAH()
    box = np.array([100.0, 200.0, 0.5, 80.0])   # x,y,a,h
    mean, cov = kf.initiate(box)
    assert mean.shape == (8,) and cov.shape == (8, 8)

    pmean, pcov = kf.predict(mean, cov)
    assert pmean.shape == (8,) and pcov.shape == (8, 8)

    # NSA: confidence tinggi -> noise pengukuran lebih kecil -> covariance proyeksi lebih kecil
    _, cov_hi = kf.project(pmean, pcov, confidence=0.9)
    _, cov_lo = kf.project(pmean, pcov, confidence=0.1)
    assert np.trace(cov_hi) < np.trace(cov_lo), "NSA: conf tinggi harus turunkan noise ukur"

    # koreksi harus menarik mean menuju measurement
    meas = np.array([130.0, 210.0, 0.5, 82.0])
    nmean, ncov, extras = kf.update(pmean, pcov, meas, confidence=0.9)
    assert abs(nmean[0] - meas[0]) < abs(pmean[0] - meas[0]), "update harus menarik ke measurement"
    assert extras["kalman_gain"].shape == (8, 4)

    # gating: measurement = proyeksi mean -> squared maha ~ 0
    proj_mean, _ = kf.project(mean, cov)
    g = kf.gating_distance(mean, cov, proj_mean.reshape(1, -1))
    assert g[0] < 1e-6, f"gating measurement=mean harus ~0, dapat {g[0]}"
    print("kalman.py OK — mean", np.round(nmean[:4], 2), "| gate(self)", float(g[0]))


if __name__ == "__main__":
    demo()
