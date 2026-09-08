"""Komponen 3 — EMA (Exponential Moving Average) update fitur appearance.

Alih-alih menyimpan galeri fitur mentah, StrongSORT menghaluskan fitur track:
    smooth = alpha * prev + (1 - alpha) * new_feat   (lalu di-L2-normalisasi)
alpha besar (0.9) -> fitur berubah pelan -> tahan terhadap noise/occlusion sesaat.
Mirror boxmot Track.update (baris 178-184).
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from my_strongsort import viz


def smooth(prev_feat, new_feat, alpha=0.9):
    """Update EMA satu langkah. Input/output vektor unit-norm."""
    prev_feat = np.asarray(prev_feat, dtype=float)
    new_feat = np.asarray(new_feat, dtype=float)
    new_feat = new_feat / np.linalg.norm(new_feat)
    out = alpha * prev_feat + (1 - alpha) * new_feat
    return out / np.linalg.norm(out)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def visualize(raw_feats, out_dir, tag="track", alpha=0.9):
    """Dari urutan fitur mentah per-frame satu track, tampilkan efek EMA:
    - drift cosine (smooth_t vs smooth_{t-1})  -> kestabilan
    - kemiripan smooth_t vs raw_t              -> seberapa 'mengikuti' deteksi
    """
    os.makedirs(out_dir, exist_ok=True)
    raw = [np.asarray(f, dtype=float) / np.linalg.norm(f) for f in raw_feats]
    sm = [raw[0]]
    for f in raw[1:]:
        sm.append(smooth(sm[-1], f, alpha))
    stab = [1.0] + [_cos(sm[t], sm[t - 1]) for t in range(1, len(sm))]
    follow = [_cos(sm[t], raw[t]) for t in range(len(sm))]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(stab, "o-", label="cos(smooth_t, smooth_{t-1}) — kestabilan")
    ax.plot(follow, "s--", label="cos(smooth_t, raw_t) — mengikuti deteksi")
    ax.set_ylim(0, 1.02); ax.set_xlabel("frame index"); ax.set_ylabel("cosine similarity")
    ax.set_title(f"EMA appearance (alpha={alpha}, {tag})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    viz.savefig(fig, f"{out_dir}/{tag}_ema_drift.png")

    # bar fitur sebelum/sesudah 1 langkah (potong 64 dim pertama agar terbaca)
    if len(raw) >= 2:
        d = min(64, raw[0].shape[0])
        fig, ax = plt.subplots(figsize=(9, 3))
        x = np.arange(d)
        ax.bar(x - 0.2, sm[-2][:d], width=0.4, label="prev smooth")
        ax.bar(x + 0.2, sm[-1][:d], width=0.4, label="new smooth")
        ax.set_title(f"fitur EMA sebelum/sesudah (dim 0..{d}, {tag})"); ax.legend(fontsize=8)
        viz.savefig(fig, f"{out_dir}/{tag}_ema_feature_bar.png")
    return out_dir


def demo():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(128); base /= np.linalg.norm(base)
    # fitur mentah = base + noise -> smooth harus mendekati base & stabil
    raw = [base + 0.3 * rng.standard_normal(128) for _ in range(10)]
    sm = [raw[0] / np.linalg.norm(raw[0])]
    for f in raw[1:]:
        sm.append(smooth(sm[-1], f))
    assert abs(np.linalg.norm(sm[-1]) - 1.0) < 1e-9, "output EMA harus unit-norm"
    # sebuah fitur baru mengubah smooth kurang dari perubahan penuh (alpha=0.9)
    step = smooth(base, -base, alpha=0.9)  # arah berlawanan
    assert _cos(step, base) > 0.5, "alpha besar: smooth harus tetap dekat prev"
    print("ema.py OK — |smooth|", round(float(np.linalg.norm(sm[-1])), 6),
          "| cos(smooth,base)", round(_cos(sm[-1], base), 3))


if __name__ == "__main__":
    demo()
