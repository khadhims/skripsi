"""Orkestrator: merangkai 11 komponen jadi tracker StrongSORT yang berjalan.

`MyStrongSort.update(dets, img)` meniru urutan boxmot StrongSort.update:
    min_conf filter -> ECC + camera_update -> ReID -> Kalman predict ->
    _match (stage appearance+gate lalu stage IoU) -> update matched ->
    mark_missed -> initiate new -> purge deleted -> update galeri -> output.

Semua nilai antara tiap frame disimpan di `self.last` supaya demo/notebook bisa
memvisualisasikan tiap komponen dari data pipeline yang sesungguhnya.

API `update()` dan format output dibuat identik dengan boxmot agar bisa
diverifikasi (lihat verify_vs_boxmot.py).
"""
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from my_strongsort import assignment, cost_matrix, gate, lifecycle, matched, new_track, output
from my_strongsort.ecc import ECC
from my_strongsort.reid import OSNetReID, DEFAULT_WEIGHTS
from my_strongsort.track import Detection

DEFAULT_CONFIG = "strongsort/configs/default_config.yaml"


def build_detections_array(result):
    """Ultralytics result -> array (N,6) [x1,y1,x2,y2,conf,cls] (mirror main.py)."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.cpu().numpy().reshape(-1, 1).astype(np.float32)
    cls = result.boxes.cls.cpu().numpy().reshape(-1, 1).astype(np.float32)
    return np.hstack((xyxy, conf, cls))


class MyStrongSort:
    def __init__(self, reid_weights=DEFAULT_WEIGHTS, device="cpu", half=False,
                 min_conf=0.1, max_cos_dist=0.2, max_iou_dist=0.7, max_age=30,
                 n_init=3, nn_budget=100, mc_lambda=0.995, ema_alpha=0.9):
        self.reid = OSNetReID(reid_weights, device, half)
        self.ecc = ECC()
        self.min_conf = min_conf
        self.max_cos_dist = max_cos_dist
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.mc_lambda = mc_lambda
        self.ema_alpha = ema_alpha

        self.tracks = []
        self._next_id = 1
        self.samples = {}     # galeri fitur: id -> list[np.ndarray]
        self.last = {}        # nilai antara frame terakhir (untuk visualisasi)

    @classmethod
    def from_default_config(cls, config_path=DEFAULT_CONFIG, **overrides):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = (yaml.safe_load(f) or {}).get("STRONGSORT", {})
        kwargs = dict(
            max_cos_dist=cfg.get("MAX_DIST", 0.2),
            max_iou_dist=cfg.get("MAX_IOU_DISTANCE", 0.7),
            max_age=cfg.get("MAX_AGE", 30),
            n_init=cfg.get("N_INIT", 3),
            nn_budget=cfg.get("NN_BUDGET", 100),
            mc_lambda=cfg.get("MC_LAMBDA", 0.995),
            ema_alpha=cfg.get("EMA_ALPHA", 0.9),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---- per-frame ----
    def update(self, dets, img, embs=None):
        self.last = {}
        dets = np.asarray(dets, dtype=float).reshape(-1, 6)
        det_idx = np.arange(len(dets))
        keep = dets[:, 4] >= self.min_conf
        dets, det_idx = dets[keep], det_idx[keep]
        xyxy, confs, clss = dets[:, :4], dets[:, 4], dets[:, 5]

        # ECC + geser tiap track (hanya bila sudah ada track, seperti boxmot)
        if len(self.tracks) >= 1:
            warp = self.ecc.apply(img)
            for t in self.tracks:
                t.camera_update(warp)
            self.last["warp_matrix"] = warp

        # OSNet ReID (embs opsional untuk melewati ekstraksi)
        feats = embs[keep] if embs is not None else self.reid.extract(xyxy, img)
        self.last.update(features=feats, xyxy=xyxy)

        tlwh = xyxy.copy()
        tlwh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
        detections = [Detection(tlwh[i], confs[i], clss[i], int(det_idx[i]), feats[i])
                      for i in range(len(dets))]

        for t in self.tracks:
            t.predict()
        # semua snapshot di-key dgn track.id (stabil) — indeks berubah setelah purge
        predicted_state = {t.id: (t.mean.copy(), t.covariance.copy()) for t in self.tracks}

        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        before = {self.tracks[ti].id: self.tracks[ti].to_tlbr().copy() for ti, _ in matches}
        matched_feats = [(self.tracks[ti].id, detections[di].feat) for ti, di in matches]
        kalman_extras = {}
        for ti, di in matches:
            kalman_extras[self.tracks[ti].id] = matched.update_track(self.tracks[ti], detections[di])
        corrected_state = {self.tracks[ti].id: (self.tracks[ti].mean.copy(),
                                                self.tracks[ti].covariance.copy())
                           for ti, _ in matches}
        for ti in unmatched_tracks:
            lifecycle.mark_missed(self.tracks[ti])
        new_tracks = []
        for di in unmatched_dets:
            t = new_track.create_track(detections[di], self._next_id,
                                       self.n_init, self.max_age, self.ema_alpha)
            self.tracks.append(t)
            new_tracks.append(t)
            self._next_id += 1
        self.tracks = lifecycle.purge_deleted(self.tracks)

        self._partial_fit()

        out = output.collect_outputs(self.tracks)
        self.last.update(matches=matches, new_tracks=new_tracks, matched_before=before,
                         predicted_state=predicted_state, corrected_state=corrected_state,
                         kalman_extras=kalman_extras, matched_feats=matched_feats,
                         detections=detections, output=out)
        return out

    # ---- matching 2-stage ----
    def _match(self, detections):
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_a, remaining = self._match_appearance(
            detections, confirmed, list(range(len(detections))))

        iou_candidates = unconfirmed + [k for k in unmatched_a
                                        if self.tracks[k].time_since_update == 1]
        unmatched_a = [k for k in unmatched_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_b, remaining = self._match_iou(detections, iou_candidates, remaining)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_a + unmatched_b))
        return matches, unmatched_tracks, remaining

    def _match_appearance(self, detections, track_indices, det_indices):
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)
        det_feats = np.array([detections[j].feat for j in det_indices])
        galleries = [self.samples.get(self.tracks[i].id, [self.tracks[i].features[-1]])
                     for i in track_indices]
        appearance = cost_matrix.cosine_cost(galleries, det_feats)
        measurements = np.array([detections[j].to_xyah() for j in det_indices])
        maha = gate.gating_distance_matrix([self.tracks[i] for i in track_indices], measurements)
        mask = gate.gate_mask(maha)
        fused = cost_matrix.fuse(appearance, maha, mask, self.mc_lambda)
        m, ur, uc = assignment.solve(fused, self.max_cos_dist)

        self.last.update(appearance_cost=appearance, maha=maha, fused_cost=fused,
                         appearance_track_ids=[self.tracks[i].id for i in track_indices],
                         appearance_det_ids=[int(detections[j].det_ind) for j in det_indices],
                         appearance_matches_local=list(m))  # (row,col) di matrix ini
        return ([(track_indices[i], det_indices[j]) for i, j in m],
                [track_indices[i] for i in ur],
                [det_indices[j] for j in uc])

    def _match_iou(self, detections, track_indices, det_indices):
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)
        det_tlwhs = np.array([detections[j].tlwh for j in det_indices])
        cost = np.zeros((len(track_indices), len(det_indices)))
        for r, i in enumerate(track_indices):
            if self.tracks[i].time_since_update > 1:
                cost[r, :] = cost_matrix.INFTY_COST
            else:
                cost[r, :] = cost_matrix.iou_cost([self.tracks[i].to_tlwh()], det_tlwhs)[0]
        m, ur, uc = assignment.solve(cost, self.max_iou_dist)

        self.last.update(iou_cost=cost,
                         iou_track_ids=[self.tracks[i].id for i in track_indices],
                         iou_det_ids=[detections[j].det_ind for j in det_indices],
                         iou_matches=[(track_indices[i], det_indices[j]) for i, j in m])
        return ([(track_indices[i], det_indices[j]) for i, j in m],
                [track_indices[i] for i in ur],
                [det_indices[j] for j in uc])

    def _partial_fit(self):
        active = [t.id for t in self.tracks if t.is_confirmed()]
        for t in self.tracks:
            if not t.is_confirmed():
                continue
            for f in t.features:
                self.samples.setdefault(t.id, []).append(f)
                if self.nn_budget:
                    self.samples[t.id] = self.samples[t.id][-self.nn_budget:]
        self.samples = {k: self.samples[k] for k in active}

    # ---- jalankan atas satu video ----
    def run_video(self, video_path, yolo_model_path, out_txt,
                  conf=0.25, iou=0.7, imgsz=1088, max_frames=None, on_frame=None):
        from ultralytics import YOLO
        model = YOLO(yolo_model_path)
        cap = cv2.VideoCapture(str(video_path))
        d = os.path.dirname(out_txt)
        if d:
            os.makedirs(d, exist_ok=True)
        frame_id = 0
        with open(out_txt, "w") as f:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id += 1
                res = model.predict(frame, conf=conf, imgsz=imgsz, iou=iou, verbose=False)[0] # type: ignore
                dets = build_detections_array(res)
                out = self.update(dets, frame)
                for line in output.to_mot_lines(frame_id, out):
                    f.write(line + "\n")
                if on_frame is not None:
                    on_frame(frame_id, frame, dets, out, self)
                if max_frames and frame_id >= max_frames:
                    break
        cap.release()
        return out_txt


if __name__ == "__main__":
    # smoke test tanpa YOLO: dua deteksi konsisten selama beberapa frame -> ID stabil
    ss = MyStrongSort()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = np.array([[100, 100, 140, 180, 0.9, 0.0],
                     [300, 120, 340, 200, 0.9, 0.0]], dtype=np.float32)
    ids_seen = None
    for fr in range(5):
        jitter = np.random.default_rng(fr).normal(0, 1, dets[:, :4].shape)
        d = dets.copy(); d[:, :4] += jitter
        out = ss.update(d, img)
        if len(out):
            ids_seen = sorted(int(r[4]) for r in out)
    assert ids_seen == [1, 2], f"harusnya 2 ID stabil, dapat {ids_seen}"
    print("pipeline.py OK — confirmed IDs", ids_seen, "| tracks", len(ss.tracks))
