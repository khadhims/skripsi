"""my_strongsort — StrongSORT dipecah per-komponen untuk keperluan skripsi.

Setiap komponen StrongSORT punya modulnya sendiri, dengan fungsi inti + `visualize()`
agar hasil tiap tahap bisa dijelaskan/divisualisasikan (permintaan dosen penguji).

Komponen (satu file per komponen):
    reid, ecc, ema, kalman, gate, cost_matrix, assignment,
    matched, new_track, lifecycle, output

Infrastruktur bersama: track (Track/Detection), pipeline (MyStrongSort), viz (plot helpers).

Catatan: __init__ sengaja dibuat ringan (tidak meng-import submodul) supaya
`python -m my_strongsort.kalman` dsb. tidak menarik dependency berat (torch).
"""

__all__ = [
    "reid", "ecc", "ema", "kalman", "gate", "cost_matrix", "assignment",
    "matched", "new_track", "lifecycle", "output", "track", "pipeline", "viz",
]
