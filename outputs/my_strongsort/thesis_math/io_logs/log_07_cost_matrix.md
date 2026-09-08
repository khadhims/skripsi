# LOG KOMPONEN 7 — Cost Matrix (Appearance + Motion Fusion)

**Sumber input:**
- Fitur gallery track hero $\mathbf{g}$ (EMA dari frame 1–29)
- Fitur deteksi baru $\mathbf{d}$ (OSNet dari frame 30, Komponen 2)
- Jarak Mahalanobis $d^2$ dari Komponen 6
**Parameter:** $\lambda_{mc} = 0.995$ (mc_lambda)

---

## INPUT

### Gallery Feature Hero g (512D, setelah 29 frame EMA)

Dalam skenario ini, gallery feature setelah 29 frame EMA sangat mendekati fitur frame saat ini (karena objek stabil). Untuk representasi nyata dari perbedaan gallery vs deteksi, gunakan nilai EMA di `09_ema_update_full512d.csv`.

Nilai $\mathbf{g}$ (gallery) = $\mathbf{f}_{smooth}$ dari EMA 29 frame (identik dengan f_hero untuk track yang stabil):

```
g = [0.00000000, 0.00000000, 0.04067962, 0.06070972, 0.02982996, ...]
(512 nilai, ‖g‖₂ = 1.000000)
```

### Detection Feature d (512D, dari Komponen 2)

```
d = [0.00000000, 0.00000000, 0.04067962, 0.06070972, 0.02982996, ...]
(512 nilai identik untuk hero — ‖d‖₂ = 1.000000)
```

---

## PROSES INTERNAL — COSINE SIMILARITY (512D)

### Definisi

Cosine similarity antara dua vektor unit-norm $\mathbf{g}$ dan $\mathbf{d}$:

$$\text{cos\_sim}(\mathbf{g}, \mathbf{d}) = \frac{\mathbf{g} \cdot \mathbf{d}}{\|\mathbf{g}\|_2 \|\mathbf{d}\|_2} = \mathbf{g} \cdot \mathbf{d} = \sum_{i=0}^{511} g_i \cdot d_i$$

Karena kedua vektor sudah unit-norm, dot product = cosine similarity.

### Perhitungan Dot Product (Semua 512 Dimensi)

Berikut kontribusi terbesar dari setiap dimensi $g_i \times d_i$:

| Dim | g_i | d_i | g_i × d_i | Catatan |
|:---:|:----------:|:----------:|:-----------:|:--------|
| 279 | 0.20649661 | 0.20649661 | **0.04264100** | Dimensi terbesar |
| 249 | 0.19264379 | 0.19264379 | **0.03711151** | |
| 54 | 0.16505015 | 0.16505015 | **0.02724155** | |
| 167 | 0.15746732 | 0.15746732 | **0.02479593** | |
| 225 | 0.16856834 | 0.16856834 | **0.02841527** | |
| 92 | 0.14734025 | 0.14734025 | **0.02170914** | |
| 453 | 0.14757493 | 0.14757493 | **0.02177836** | |
| 7 | 0.13333309 | 0.13333309 | **0.01777771** | |
| 224 | 0.13794310 | 0.13794310 | **0.01902830** | |
| 247 | 0.13300361 | 0.13300361 | **0.01768996** | |
| *(502 dimensi lain)* | ... | ... | Σ ≈ 0.84001 | |

Seluruh 512 dimensi (file: `02b_reid_output_feat512d.csv`):

$$\mathbf{g} \cdot \mathbf{d} = \sum_{i=0}^{511} g_i \cdot d_i = 1.000000$$

> **Catatan penting:** Nilai cos_sim = 1.000 terjadi karena dalam ekstraksi numerik ini, gallery $\mathbf{g}$ dan deteksi $\mathbf{d}$ menggunakan vektor yang sama (track sudah sangat stabil setelah 30 frame). Dalam kondisi nyata dengan EMA 29 frame sebelumnya, cos_sim ≈ 0.99, bukan tepat 1.0.

### Appearance Cost

$$c_{app} = 1 - \text{cos\_sim} = 1 - 1.000000 = 0.000000$$

(Untuk track yang stabil setelah 30 frame, hampir tidak ada perbedaan penampilan.)

---

## MATRIKS BIAYA PENUH (37 Deteksi × 1 Track)

Untuk setiap pasangan deteksi $j$ dan track hero:

$$c_{app}^{(j)} = 1 - (\mathbf{g}_{hero} \cdot \mathbf{d}_j)$$

**File data:** `07_cost_matrix.csv`

| det_idx | cos_sim | c_app | c_maha_norm | c_fused |
|:-------:|:-------:|:-----:|:-----------:|:-------:|
| **0★** | 1.0000 | 0.0000 | 0.0225 | **0.0001** |
| 1 | 0.6492 | 0.3508 | 677.92 | 3.7386 |
| 2 | 0.4868 | 0.5132 | 353.34 | 2.2774 |
| 3 | 0.5795 | 0.4205 | 1571.64 | 8.2766 |
| 4 | 0.7132 | 0.2868 | 473.92 | 2.6550 |
| 5 | 0.4764 | 0.5236 | 2128.69 | 11.1644 |
| 6 | 0.4872 | 0.5128 | 1055.52 | 5.7879 |
| 7 | 0.5238 | 0.4762 | 1216.74 | 6.5575 |
| 8 | 0.4500 | 0.5500 | 1796.60 | 9.5302 |
| 9 | 0.4912 | 0.5088 | 1572.01 | 8.7424 |
| 10 | 0.5341 | 0.4659 | 944.16 | 5.1906 |
| *(26 det lain)* | ... | ... | ... | >> 1.0 |

> **det_idx=0** memiliki c_fused = 0.0001 yang jauh lebih kecil dibanding semua deteksi lain (minimum ke-2 ≈ 2.28). Ini menjamin Hungarian Assignment akan memilih pasangan ini.

---

## FUSION COST

### Formula

$$c_{fused} = \lambda_{mc} \cdot c_{app} + (1 - \lambda_{mc}) \cdot c_{maha\_norm}$$

$$c_{fused} = 0.995 \cdot c_{app} + 0.005 \cdot c_{maha\_norm}$$

### Perhitungan untuk Hero (det_idx=0)

```
c_app       = 1 - cos_sim = 1 - 1.000 = 0.0000
c_maha_norm = d² / threshold = 0.2134 / 9.4877 = 0.02249

c_fused = 0.995 × 0.0000 + 0.005 × 0.02249
        = 0.000000 + 0.000112
        = 0.000112
```

**Interpretasi $\lambda_{mc} = 0.995$:**
- 99.5% bobot pada appearance (fitur ReID yang kaya informasi)
- 0.5% bobot pada jarak Mahalanobis (tambahan konsistensi geometri)

Strategi ini masuk akal karena:
- Appearance feature 512D jauh lebih diskriminatif daripada posisi 4D
- Mahalanobis tetap diikutsertakan sebagai "tiebreaker" jika dua orang berpenampilan mirip

---

## KONTEKS MULTI-TRACK (Lebih dari 1 Track Aktif)

Dalam skenario nyata dengan $M$ track aktif dan $N$ deteksi:

Cost matrix $\mathbf{C} \in \mathbb{R}^{M \times N}$:

$$C_{ij} = 0.995 \cdot (1 - \mathbf{g}_i \cdot \mathbf{d}_j) + 0.005 \cdot \frac{d^2_{ij}}{\chi^2_{thresh}}$$

Elemen yang sudah digating (terblokir gate Mahalanobis) diberi nilai $\infty$ (atau nilai besar seperti $10^5$) agar Hungarian Assignment tidak memilihnya.

**Contoh cost matrix 3 track × 3 deteksi:**

```
           det-0   det-1   det-2
Track-1  [ 0.001   2.345   INF  ]   ← hero, match dengan det-0
Track-2  [ INF     0.120   1.890]   ← match dengan det-1
Track-3  [ INF     INF     0.050]   ← match dengan det-2
```
