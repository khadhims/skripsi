# LOG KOMPONEN 6 — Gating Mahalanobis

**Sumber input:** State prediksi hero (Komponen 4) + semua 37 pengukuran YOLO (Komponen 1)
**Tujuan:** Menolak secara cepat pasangan track-deteksi yang geometri posisinya tidak konsisten

---

## INPUT

```
x_pred_hero = [1201.26, 430.75, 1.212, 87.10, 0.4665, 0.139, 0, 0.1042]
P_pred_hero  → matriks 8×8 (lihat log_04)
Threshold    : χ²(df=4, p=0.95) = 9.4877

Total deteksi yang diuji: 37
```

---

## PROSES INTERNAL — DISTANCE MAHALANOBIS

### Matriks Proyeksi dan S

Pertama, proyeksi state ke ruang pengukuran (tanpa NSA — confidence=0 untuk gating):

$$\mathbf{S}_{gate} = \mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^\top + \mathbf{R}_{base}$$

di mana $\mathbf{R}_{base}$ menggunakan noise penuh (confidence=0, tidak ada NSA).

Karena P dan R bersifat diagonal, S juga diagonal:

```
S_gate[x,x] = P_pred[x,x] + std_base_x² = 21.5985 + 4.355² = 21.5985 + 18.9660 = 40.5645
S_gate[y,y] = 40.5645
S_gate[a,a] = 0.000162 + 0.01² = 0.000162 + 0.0001 = 0.000262
S_gate[h,h] = 40.5645
```

### Jarak Mahalanobis untuk Setiap Deteksi

Untuk setiap deteksi $i$ dengan pengukuran $\mathbf{z}_i = [cx_i, cy_i, a_i, h_i]^\top$:

**Langkah 1 — Perbedaan pengukuran:**
$$\mathbf{d}_i = \mathbf{z}_i - \mathbf{H}\mathbf{x}_{pred} = \mathbf{z}_i - [1201.26, 430.75, 1.212, 87.10]^\top$$

**Langkah 2 — Jarak Mahalanobis kuadrat:**

Karena S diagonal:
$$d^2_i = \mathbf{d}_i^\top \mathbf{S}^{-1} \mathbf{d}_i = \frac{d_x^2}{S_{xx}} + \frac{d_y^2}{S_{yy}} + \frac{d_a^2}{S_{aa}} + \frac{d_h^2}{S_{hh}}$$

$$d^2_i = \frac{d_x^2}{40.565} + \frac{d_y^2}{40.565} + \frac{d_a^2}{0.000262} + \frac{d_h^2}{40.565}$$

**Perhatikan:** $1/S_{aa} = 1/0.000262 = 3816$ — sangat besar. Jika deteksi memiliki rasio aspek yang berbeda dari prediksi, $d^2$ akan meledak besar.

---

## OUTPUT — Tabel Jarak Mahalanobis Semua Deteksi

**File data:** `06_gating_mahalanobis.csv`

| det | cx | cy | a | h | d²_maha | d² < 9.4877 | GATE |
|:---:|:------:|:------:|:------:|:-----:|:--------:|:-----------:|:----:|
| **0★** | 1202.51 | 429.90 | 1.2509 | 86.52 | **0.2134** | ✓ | **LOLOS** |
| 1 | 727.34 | 241.26 | 0.9037 | 82.50 | 6431.86 | ✗ | DITOLAK |
| 2 | 935.10 | 176.68 | 0.8985 | 72.87 | 3352.42 | ✗ | DITOLAK |
| 3 | 470.58 | 167.64 | 1.1929 | 45.25 | 14911.21 | ✗ | DITOLAK |
| 4 | 1027.10 | 44.24 | 1.4590 | 37.83 | 4496.42 | ✗ | DITOLAK |
| 5 | 311.66 | 270.49 | 0.7837 | 48.97 | 20196.36 | ✗ | DITOLAK |
| 6 | 675.70 | 73.20 | 1.2926 | 40.63 | 10014.48 | ✗ | DITOLAK |
| 7 | 527.84 | 324.03 | 0.6914 | 39.03 | 11544.04 | ✗ | DITOLAK |
| 8 | 373.13 | 380.09 | 1.4180 | 33.07 | 17045.58 | ✗ | DITOLAK |
| 9 | 463.31 | 452.97 | 0.7465 | 67.91 | 7783.16 | ✗ | DITOLAK |
| 10 | 1157.29 | 282.11 | 0.7068 | 64.97 | 5193.48 | ✗ | DITOLAK |
| 11–36 | (berbagai posisi) | ... | ... | ... | >> 9.4877 | ✗ | DITOLAK |

> Hanya **det_idx=0** (hero) yang lolos gate. 36 deteksi lainnya ditolak karena:
> - Posisi (cx, cy) terlalu jauh dari prediksi track hero (di area berbeda frame)
> - Rasio aspek (a) berbeda signifikan → kontribusi $d_a^2/S_{aa}$ sangat besar

---

## PERHITUNGAN DETAIL HERO (det_idx=0)

```
d_x = 1202.51 - 1201.26 = +1.25 px
d_y =  429.90 -  430.75 = -0.85 px
d_a =   1.2509 -  1.2120 = +0.0389
d_h =  86.522 -  87.100 = -0.578 px

Kontribusi per dimensi ke d²:
  d_x² / S_xx = (1.25)² / 40.565  =  1.5625 / 40.565  = 0.03851
  d_y² / S_yy = (0.85)² / 40.565  =  0.7225 / 40.565  = 0.01781
  d_a² / S_aa = (0.0389)² / 0.000262 = 0.001513 / 0.000262 = 5.7748
  d_h² / S_hh = (0.578)² / 40.565  =  0.3341 / 40.565  = 0.00824

d² = 0.03851 + 0.01781 + 5.7748 + 0.00824 = 5.8394
```

> **Catatan:** Nilai d²=0.2134 pada kode menggunakan S dari NSA (bukan gating_distance full noise).
> Hasil di atas (d²=5.84) menggunakan noise penuh. Keduanya < 9.4877 → hero tetap lolos.

---

## VISUALISASI KONSEP GATING

```
Distribusi prediksi hero di ruang cx-cy:

          cy=420
            │
      430 ──┼──────────────────────────────
            │                              │
            │     ╔═════════════════╗      │  Elipsoida 95%
            │     ║  x pred=1201.26 ║      │  threshold χ²=9.4877
            │     ║  y pred=430.75  ║      │
            │     ║     ×           ║      │
      435 ──┼─────╫───── hero ──────╫──── │
            │     ║    det=0 ✓      ║      │
            │     ╚═════════════════╝      │
            │                              │
      440 ──┼──────────────────────────────
            │
          cx=1190   1200   1210

Semua 36 deteksi lain berada di cx<1000 atau cy<300
→ jauh di luar elipsoida → d² >> 9.4877 → DITOLAK
```

**Threshold χ²(df=4, p=0.95) = 9.4877:**

Ini berarti jika error pengukuran mengikuti distribusi Gaussian, 95% pengukuran yang benar akan memiliki $d^2 < 9.4877$. Deteksi dengan $d^2 > 9.4877$ dianggap bukan berasal dari track ini dengan keyakinan 95%.
