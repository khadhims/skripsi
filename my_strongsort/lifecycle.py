"""Komponen 10 — Lost Track & Delete Track (manajemen umur track).

Track yang tidak ter-match pada suatu frame ditandai "missed":
  - masih Tentative            -> langsung Deleted (bukti belum cukup)
  - sudah lewat max_age frame   -> Deleted (dianggap hilang permanen)
Selain itu track hilang sementara tetap hidup (prediksi Kalman saja) sampai
kembali ter-match. Mirror boxmot Track.mark_missed + filter is_deleted.

Output: timeline time_since_update / age / hits dan transisi state per track.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from my_strongsort import viz
from my_strongsort.track import TrackState

_STATE_NAME = {TrackState.Tentative: "Tentative", TrackState.Confirmed: "Confirmed",
               TrackState.Deleted: "Deleted"}


def mark_missed(track):
    if track.state == TrackState.Tentative:
        track.state = TrackState.Deleted
    elif track.time_since_update > track._max_age:
        track.state = TrackState.Deleted


def purge_deleted(tracks):
    return [t for t in tracks if not t.is_deleted()]


def visualize(history, out_dir, tag="lifecycle"):
    """history: dict track_id -> list of dict(frame, time_since_update, age, hits, state)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for tid, rows in history.items():
        frames = [r["frame"] for r in rows]
        ax1.plot(frames, [r["time_since_update"] for r in rows], ".-", label=f"id {tid}")
        ax2.plot(frames, [r["hits"] for r in rows], ".-", label=f"id {tid}")
    ax1.set_ylabel("time_since_update"); ax1.set_title(f"umur track ({tag})"); ax1.grid(alpha=0.3)
    ax2.set_ylabel("hits"); ax2.set_xlabel("frame"); ax2.grid(alpha=0.3)
    if len(history) <= 12:
        ax1.legend(fontsize=7, ncol=2)
    viz.savefig(fig, f"{out_dir}/{tag}_timeline.png")
    return out_dir


def demo():
    from my_strongsort.track import Detection, Track
    feat = np.ones(8) / np.sqrt(8)

    # Tentative yang miss -> langsung dihapus
    t = Track(Detection([10, 10, 40, 80], 0.9, 0, 0, feat), 1, n_init=3, max_age=2, ema_alpha=0.9)
    mark_missed(t)
    assert t.is_deleted(), "Tentative miss harus Deleted"

    # Confirmed hilang sementara: bertahan sampai time_since_update > max_age
    t2 = Track(Detection([10, 10, 40, 80], 0.9, 0, 0, feat), 2, n_init=1, max_age=2, ema_alpha=0.9)
    t2.state = TrackState.Confirmed
    for _ in range(2):           # tsu -> 1, 2  (masih hidup)
        t2.predict(); mark_missed(t2)
    assert not t2.is_deleted(), "Confirmed dalam max_age harus tetap hidup"
    t2.predict(); mark_missed(t2)  # tsu -> 3 > max_age
    assert t2.is_deleted(), "Confirmed lewat max_age harus Deleted"

    assert purge_deleted([t, t2]) == []
    print("lifecycle.py OK — t1", _STATE_NAME[t.state], "| t2", _STATE_NAME[t2.state])


if __name__ == "__main__":
    demo()
