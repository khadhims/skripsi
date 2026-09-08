"""Komponen 7 — Hungarian Assignment (asosiasi ID optimal).

Menyelesaikan penugasan minimum-cost antara track (baris) dan deteksi (kolom)
dengan scipy.optimize.linear_sum_assignment, lalu membuang pasangan yang biaya-
nya melebihi ambang (max_distance). Mirror boxmot min_cost_matching.

Output: daftar pasangan (track_row, det_col), track tak-terpasang, deteksi tak-terpasang.
"""
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from my_strongsort import viz


def solve(cost_matrix, max_distance):
    """Return (matches, unmatched_rows, unmatched_cols) dalam indeks lokal matrix."""
    cost = np.asarray(cost_matrix, dtype=float)
    n, m = cost.shape if cost.ndim == 2 else (len(cost), 0)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    cost = cost.copy()
    cost[cost > max_distance] = max_distance + 1e-5
    rows, cols = linear_sum_assignment(cost)

    assigned_r, assigned_c = set(rows), set(cols)
    unmatched_rows = [i for i in range(n) if i not in assigned_r]
    unmatched_cols = [j for j in range(m) if j not in assigned_c]
    matches = []
    for i, j in zip(rows, cols):
        if cost[i, j] > max_distance:
            unmatched_rows.append(i)
            unmatched_cols.append(j)
        else:
            matches.append((int(i), int(j)))
    return matches, unmatched_rows, unmatched_cols


def visualize(cost_matrix, matches, out_dir, tag="frame", track_ids=None, det_ids=None):
    """Heatmap cost dengan sel yang terpilih (matched) ditandai kotak putih."""
    os.makedirs(out_dir, exist_ok=True)
    cost = np.atleast_2d(np.asarray(cost_matrix, dtype=float))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(max(3.0, 0.7 * cost.shape[1] + 2),
                                    max(2.5, 0.7 * cost.shape[0] + 1.5)))
    im = ax.imshow(cost, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j) in matches:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", lw=2.5))
    if cost.size <= 144:
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                ax.text(j, i, f"{cost[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    if track_ids is not None:
        ax.set_yticks(range(len(track_ids))); ax.set_yticklabels(track_ids)
    if det_ids is not None:
        ax.set_xticks(range(len(det_ids))); ax.set_xticklabels(det_ids)
    ax.set_title(f"Hungarian assignment ({tag}) — {len(matches)} match")
    ax.set_xlabel("detections"); ax.set_ylabel("tracks")
    viz.savefig(fig, f"{out_dir}/{tag}_assignment.png")
    return out_dir


def demo():
    # optimum jelas: diagonal murah -> match (0,0),(1,1),(2,2)
    cost = np.array([[0.1, 0.9, 0.8], [0.7, 0.05, 0.9], [0.9, 0.8, 0.2]])
    matches, ur, uc = solve(cost, max_distance=0.5)
    assert sorted(matches) == [(0, 0), (1, 1), (2, 2)], matches
    assert ur == [] and uc == []
    # semua di atas ambang -> tak ada match
    matches2, ur2, uc2 = solve(np.full((2, 2), 0.9), max_distance=0.5)
    assert matches2 == [] and sorted(ur2) == [0, 1] and sorted(uc2) == [0, 1]
    print("assignment.py OK — matches", matches)


if __name__ == "__main__":
    demo()
