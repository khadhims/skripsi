"""Komponen 6 — Cost Matrix (appearance + motion + IoU).

Tiga jenis biaya asosiasi:
  1. appearance cost  : 1 - cosine similarity fitur ReID (nearest-neighbor atas galeri)
  2. fused cost       : mc_lambda * appearance + (1 - mc_lambda) * mahalanobis,
                        entri yang ter-gate diset INFTY (dipakai stage appearance)
  3. IoU cost         : 1 - IoU (dipakai stage kedua utk track tak-terkonfirmasi)
Mirror boxmot NearestNeighborDistanceMetric.distance, gate_cost_matrix, iou_cost.
"""
import os

import numpy as np

from my_strongsort import viz

INFTY_COST = 1e5


def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosine_cost(galleries, det_feats):
    """N×M cost = 1 - max cosine similarity antara tiap fitur deteksi dan galeri track.

    galleries : list panjang N; tiap elemen matrix (Gi × D) fitur historis track.
    det_feats : matrix (M × D) fitur deteksi frame ini.
    """
    det = _unit(np.atleast_2d(det_feats))
    out = np.zeros((len(galleries), len(det)))
    for i, g in enumerate(galleries):
        g = _unit(np.atleast_2d(g))
        dist = 1.0 - g @ det.T          # (Gi × M) cosine distance
        out[i, :] = dist.min(axis=0)    # nearest neighbor atas galeri
    return out


def fuse(appearance_cost, maha, mask=None, mc_lambda=0.995, gated_cost=INFTY_COST):
    """Gabung appearance + motion (Mahalanobis) seperti StrongSORT gate_cost_matrix."""
    fused = np.array(appearance_cost, dtype=float, copy=True)
    if mask is not None:
        fused[mask] = gated_cost
    return mc_lambda * fused + (1 - mc_lambda) * maha


def _iou(bbox, candidates):
    """IoU satu bbox tlwh terhadap kandidat tlwh (M×4)."""
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    cand_tl, cand_br = candidates[:, :2], candidates[:, :2] + candidates[:, 2:]
    tl = np.maximum(bbox_tl, cand_tl)
    br = np.minimum(bbox_br, cand_br)
    wh = np.maximum(0.0, br - tl)
    inter = wh.prod(axis=1)
    area_b = bbox[2:].prod()
    area_c = candidates[:, 2:].prod(axis=1)
    return inter / (area_b + area_c - inter)


def iou_cost(track_tlwhs, det_tlwhs):
    """N×M cost = 1 - IoU."""
    track_tlwhs = np.atleast_2d(np.asarray(track_tlwhs, dtype=float))
    det_tlwhs = np.atleast_2d(np.asarray(det_tlwhs, dtype=float))
    out = np.zeros((len(track_tlwhs), len(det_tlwhs)))
    for i, b in enumerate(track_tlwhs):
        out[i, :] = 1.0 - _iou(b, det_tlwhs)
    return out


def visualize(matrices, out_dir, tag="frame", track_ids=None, det_ids=None):
    """matrices: dict nama->matrix (mis. {'appearance':..,'fused':..,'iou':..})."""
    os.makedirs(out_dir, exist_ok=True)
    for name, mat in matrices.items():
        if mat is None or np.asarray(mat).size == 0:
            continue
        viz.save_matrix(mat, f"{out_dir}/{tag}_{name}")
        viz.heatmap(mat, f"{out_dir}/{tag}_{name}.png",
                    title=f"{name} cost ({tag})", xlabel="detections", ylabel="tracks",
                    xticklabels=det_ids, yticklabels=track_ids)
    return out_dir


def demo():
    a = _unit(np.array([1.0, 0, 0, 0]))
    b = _unit(np.array([0.0, 1, 0, 0]))
    cc = cosine_cost([[a], [b]], np.vstack([a, b]))
    assert cc[0, 0] < 1e-6 and cc[1, 1] < 1e-6, "fitur identik -> cost 0"
    assert cc[0, 1] > 0.9, "fitur orthogonal -> cost ~1"

    box = np.array([0.0, 0, 10, 10])
    ic = iou_cost([box], np.vstack([box, box + np.array([100, 100, 0, 0])]))
    assert ic[0, 0] < 1e-6, "box identik -> IoU cost 0"
    assert ic[0, 1] > 0.99, "box tak beririsan -> IoU cost 1"

    maha = np.array([[1.0, 50.0], [50.0, 1.0]])
    mask = maha > 9.4877
    fused = fuse(cc, maha, mask, mc_lambda=0.995)
    assert fused[0, 1] > fused[0, 0], "entri ter-gate harus lebih mahal"
    print("cost_matrix.py OK — appearance diag", np.round(np.diag(cc), 3),
          "| fused[0]", np.round(fused[0], 3))


if __name__ == "__main__":
    demo()
