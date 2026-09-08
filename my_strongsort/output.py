"""Komponen 11 — Track Output.

Menghasilkan output akhir per frame: hanya track Confirmed yang baru ter-update
(time_since_update < 1). Tiap baris: [x1, y1, x2, y2, id, conf, cls, det_idx].
Juga menulis berkas MOTChallenge: `frame,id,x,y,w,h,conf,cls,1`.
Mirror boxmot StrongSort output loop + format tulis di main.py.
"""
import os

import numpy as np

from my_strongsort import viz


def collect_outputs(tracks):
    """Return array (K,8): [x1,y1,x2,y2,id,conf,cls,det_idx] untuk track aktif."""
    outs = []
    for t in tracks:
        if not t.is_confirmed() or t.time_since_update >= 1:
            continue
        x1, y1, x2, y2 = t.to_tlbr()
        outs.append([x1, y1, x2, y2, t.id, t.conf, t.cls, t.det_ind])
    return np.asarray(outs, dtype=float).reshape(-1, 8) if outs else np.empty((0, 8))


def to_mot_lines(frame_id, output_array, force_class=1):
    """Format MOTChallenge (kelas dipaksa agnostik=1, seperti main.py)."""
    lines = []
    for row in output_array:
        x1, y1, x2, y2 = row[:4]
        tid, conf = int(row[4]), float(row[5])
        w, h = x2 - x1, y2 - y1
        lines.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{force_class},1")
    return lines


def visualize(img, output_array, out_dir, tag="frame"):
    """Frame beranotasi ID + box (output final)."""
    os.makedirs(out_dir, exist_ok=True)
    boxes = output_array[:, :4]
    ids = [int(r[4]) for r in output_array]
    out = viz.draw_boxes(img, boxes, labels=[f"ID {i}" for i in ids], color=(0, 200, 0))
    viz.save_image(out, f"{out_dir}/{tag}_output.png")
    return out_dir


def demo():
    from my_strongsort.track import Detection, Track, TrackState
    feat = np.ones(8) / np.sqrt(8)
    confirmed = Track(Detection([100, 200, 40, 80], 0.9, 0, 5, feat), 7, 1, 30, 0.9)
    confirmed.state = TrackState.Confirmed  # tsu=0 -> harus keluar
    tentative = Track(Detection([10, 10, 40, 80], 0.8, 0, 6, feat), 8, 3, 30, 0.9)  # tak keluar

    out = collect_outputs([confirmed, tentative])
    assert out.shape == (1, 8) and int(out[0, 4]) == 7, out
    lines = to_mot_lines(3, out)
    assert lines[0].startswith("3,7,") and lines[0].endswith(",1,1"), lines
    print("output.py OK —", lines[0])


if __name__ == "__main__":
    demo()
