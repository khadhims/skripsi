"""Container pasif: Detection dan Track.

Track hanya menyimpan state (mean/covariance dari Kalman, penghitung
hits/age/time_since_update, galeri fitur appearance) + helper geometri yang
dipakai lintas komponen. LOGIKA yang mengubah Track hidup di modul komponennya
masing-masing supaya tiap komponen StrongSORT memiliki kodenya sendiri:
    - kalman.py   -> predict (dipakai Track.predict)
    - matched.py  -> update track ter-match (KF update + EMA + promosi state)
    - new_track.py-> pembuatan track baru
    - lifecycle.py-> mark_missed / hapus track
    - output.py   -> kumpulkan & tulis output
"""
import numpy as np

from my_strongsort.kalman import NSAKalmanFilterXYAH


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Detection:
    """Satu deteksi pada satu frame: box tlwh + conf/cls/index + fitur ReID."""

    def __init__(self, tlwh, conf, cls, det_ind, feat):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.conf = float(conf)
        self.cls = cls
        self.det_ind = det_ind
        self.feat = feat

    def to_xyah(self):
        """tlwh -> (center x, center y, aspect ratio w/h, height)."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class Track:
    def __init__(self, detection, track_id, n_init, max_age, ema_alpha):
        self.id = track_id
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.state = TrackState.Tentative
        self.features = []
        if detection.feat is not None:
            self.features.append(detection.feat / np.linalg.norm(detection.feat))
        self._n_init = n_init
        self._max_age = max_age
        self.kf = NSAKalmanFilterXYAH()
        self.mean, self.covariance = self.kf.initiate(self.bbox)

    # --- geometri ---
    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def camera_update(self, warp_matrix):
        """Geser state pakai warp 2x3 dari ECC (kompensasi gerak kamera)."""
        a, b = warp_matrix
        M = np.array([a, b, [0, 0, 1]], dtype=float)
        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = M @ np.array([x1, y1, 1.0])
        x2_, y2_, _ = M @ np.array([x2, y2, 1.0])
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]

    # --- lifecycle counters ---
    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
