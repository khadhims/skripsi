# LOG KOMPONEN 8 — Hungarian Assignment

**Sumber input:** Cost matrix dari Komponen 7
**Algoritma:** Kuhn-Munkres (Hungarian Algorithm), kompleksitas $O(n^3)$
**Library:** `scipy.optimize.linear_sum_assignment`

---

## INPUT

Cost matrix yang sudah digating (nilai besar untuk pasangan yang diblok):

```
Untuk konteks sederhana (hero track saja):
C_fused = [0.000112,  3.7386,  2.2774,  8.2766, ...,  (37 nilai)]
           det-0★     det-1    det-2    det-3    ...

Pasangan yang diblok gate (d² > 9.4877) → nilai = INF (tidak dipilih)
Hanya det-0 yang lolos gate → dipaksa dipilih.
```

---

## PROSES INTERNAL — HUNGARIAN ALGORITHM

### Formulasi Masalah

Diberikan cost matrix $\mathbf{C} \in \mathbb{R}^{M \times N}$ (M track, N deteksi):

$$\text{Minimize} \sum_{i=1}^{M} \sum_{j=1}^{N} C_{ij} \cdot X_{ij}$$

**Subject to:**
$$\sum_{j=1}^{N} X_{ij} \leq 1, \quad \forall i \in \{1,...,M\}$$
$$\sum_{i=1}^{M} X_{ij} \leq 1, \quad \forall j \in \{1,...,N\}$$
$$X_{ij} \in \{0, 1\}$$

Ini adalah *Linear Sum Assignment Problem* (LSAP) — setiap track dipasangkan ke paling banyak satu deteksi, dan sebaliknya.

### Algoritma Kuhn-Munkres (Langkah Utama)

**Langkah 1 — Row Reduction:**
Kurangi setiap elemen dengan nilai minimum di barisnya:
$$C'_{ij} \leftarrow C_{ij} - \min_j(C_{ij})$$

**Langkah 2 — Column Reduction:**
Kurangi setiap elemen dengan nilai minimum di kolomnya:
$$C''_{ij} \leftarrow C'_{ij} - \min_i(C'_{ij})$$

**Langkah 3 — Temukan penugasan zero:**
Tandai nol dalam matriks tereduksi. Jika bisa ditemukan M penugasan nol yang tidak saling berbagi baris/kolom → optimal. Jika tidak, lakukan langkah 4.

**Langkah 4 — Cover zeros dengan garis minimal:**
Temukan jumlah minimum garis (baris/kolom) yang mencakup semua nol. Jika jumlah garis < M, modifikasi matriks dan ulangi.

**Kompleksitas:** $O(M^3)$ untuk M track = N deteksi.

### Threshold Penolakan

Setelah Hungarian Assignment, pasangan dengan cost melebihi threshold `max_dist` ditolak:

$$\text{Jika } C_{i^*j^*} > \text{max\_dist} = 0.2 \Rightarrow \text{pasangan ditolak (menjadi unmatched)}$$

---

## OUTPUT

### Untuk Skenario Hero

```
Track #1  →  dipasangkan dengan  det_idx=0
Cost assignment = 0.000112  (< max_dist=0.2 → VALID)
```

### Daftar Unmatched Detection (36 deteksi)

Seluruh 36 deteksi lain (det_idx=1 s.d. 36) tidak dipasangkan dengan track hero.
Dalam pipeline penuh dengan banyak track, setiap deteksi ini akan diuji terhadap track lain.
Jika tidak ada track yang cocok → **kandidat track baru (Tentative)**.

### Contoh Output Lengkap (Pipeline Penuh dengan Banyak Track)

Asumsikan ada 5 track aktif dan 37 deteksi:

```
Cost Matrix setelah gating (5×37), entri INF = diblok gate:

         det0   det1   det2   det3  ...  det36
Track1 [0.001   INF    INF    INF  ...   INF ]  ← hero, match det0
Track2 [ INF   0.120   INF   0.890 ...   INF ]  ← match det1
Track3 [ INF    INF   0.050   INF  ...   INF ]  ← match det2
Track4 [ INF    INF    INF   0.340 ...   INF ]  ← match det3
Track5 [ INF    INF    INF    INF  ...  0.155]  ← match det36

Hungarian Assignment result:
  Track1 ↔ det0  (cost=0.001)  ✓ MATCHED
  Track2 ↔ det1  (cost=0.120)  ✓ MATCHED
  Track3 ↔ det2  (cost=0.050)  ✓ MATCHED
  Track4 ↔ det3  (cost=0.340)  ✗ cost > 0.2 → UNMATCHED
  Track5 ↔ det36 (cost=0.155) ✓ MATCHED

Unmatched tracks: {Track4}     → mark_missed → state Predict saja
Unmatched dets:   {det4..det35} → kandidat track baru
```

### Visualisasi Cost Matrix (5 Track × 37 Det, Hasil Assignment)

```
        det0   det1   det2   det3   det4  ...  det36
Track1 [■0.001  ∞      ∞      ∞      ∞   ...   ∞   ]  ■ = SELECTED
Track2 [ ∞    ■0.120   ∞    0.890    ∞   ...   ∞   ]
Track3 [ ∞      ∞    ■0.050   ∞      ∞   ...   ∞   ]
Track4 [ ∞      ∞      ∞   [0.340]   ∞   ...   ∞   ]  [] = ditolak cost
Track5 [ ∞      ∞      ∞      ∞      ∞   ...  ■0.155]

Legenda:
  ■ = pasangan final yang dipilih (cost < max_dist)
  ∞ = diblok gate Mahalanobis (d² > 9.4877)
  [] = dipilih Hungarian tapi cost > 0.2 → unmatched
```

---

## RINGKASAN I/O KOMPONEN 8

| Output | Nilai |
|:-------|:------|
| Track #1 → det_idx=0 | assignment cost = 0.000112 |
| Status hero | MATCHED ✓ |
| Unmatched track hero | Tidak ada |
| Threshold max_dist | 0.2 |
| Total deteksi unmatched | 36 (menjadi kandidat new track) |
