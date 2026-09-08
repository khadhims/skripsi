# LOG KOMPONEN 3 — ECC (Enhanced Correlation Coefficient)

**Sumber input:** Frame ke-29 (t−1) dan Frame ke-30 (t)
**Tujuan:** Mengestimasi pergerakan kamera antara dua frame berurutan

---

## INPUT

```
Frame t-1  (frame ke-29): 1920 × 1080, BGR, uint8
Frame t    (frame ke-30): 1920 × 1080, BGR, uint8
```

Langkah pre-processing sebelum ECC:
1. Konversi BGR → Grayscale: $I = 0.114B + 0.587G + 0.299R$
2. Resize ke 15% ukuran asli (scale=0.15):

```
Ukuran after resize:
  Lebar : int(1920 × 0.15) = 288 px
  Tinggi: int(1080 × 0.15) = 162 px

Frame grayscale kecil:
  I_{t-1}: 162 × 288 (uint8)
  I_t    : 162 × 288 (uint8)
```

**Kenapa di-resize?** Untuk efisiensi — ECC adalah iterasi yang mahal. Pada skala 15%, translasi kamera sub-piksel tetap terdeteksi dengan akurasi memadai.

---

## PROSES INTERNAL ECC — MATEMATIKA LENGKAP

### Model Warp (Translation-Only)

StrongSORT hanya menggunakan translasi (`MOTION_TRANSLATION`).
Matriks warp afin $\mathbf{W}$ berukuran $2 \times 3$:

$$\mathbf{W} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$$

Hanya dua parameter yang dioptimasi: $\mathbf{p} = [t_x, t_y]^\top$.

Efek pada koordinat piksel:
$$\mathbf{x}' = \mathbf{W} \cdot \tilde{\mathbf{x}} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \end{bmatrix}$$

### Kriteria ECC (Enhanced Correlation Coefficient)

ECC memaksimalkan korelasi antara template $T$ (frame t-1) dan gambar yang di-warp $I(\mathbf{W}(\mathbf{x};\mathbf{p}))$:

$$\rho(\mathbf{p}) = \frac{\mathbf{T}^\top \cdot \overline{I(\mathbf{W})}}{\|\mathbf{T}\| \cdot \|\overline{I(\mathbf{W})}\|}$$

di mana:
- $\mathbf{T}$ = vektor piksel template (frame t-1, zero-mean)
- $\overline{I(\mathbf{W})}$ = vektor piksel gambar target setelah warp (zero-mean)

### Algoritma Lucas-Kanade ECC (Iteratif)

Pada setiap iterasi $k$:

**Langkah 1 — Warp gambar target:**
$$I_w(\mathbf{x}) = I_t(\mathbf{W}(\mathbf{x}; \mathbf{p}^{(k)}))$$

(Interpolasi bilinear pada koordinat $[x + t_x^{(k)},\ y + t_y^{(k)}]$)

**Langkah 2 — Normalisasi zero-mean:**
$$\overline{I_w}(\mathbf{x}) = I_w(\mathbf{x}) - \frac{1}{N}\sum_{\mathbf{x}} I_w(\mathbf{x})$$

$$\overline{T}(\mathbf{x}) = T(\mathbf{x}) - \frac{1}{N}\sum_{\mathbf{x}} T(\mathbf{x})$$

di mana $N$ = jumlah piksel dalam ROI

**Langkah 3 — Hitung gradien gambar target (terhadap piksel):**

$$\nabla I_w = \left[\frac{\partial I_w}{\partial x},\ \frac{\partial I_w}{\partial y}\right]$$

Implementasi praktis dengan filter Sobel atau finite difference:
$$\frac{\partial I_w}{\partial x}[r,c] \approx \frac{I_w[r,c+1] - I_w[r,c-1]}{2}$$
$$\frac{\partial I_w}{\partial y}[r,c] \approx \frac{I_w[r+1,c] - I_w[r-1,c]}{2}$$

**Langkah 4 — Jacobian Warp $\mathbf{J}_w$ (untuk translasi):**

Jacobian menghubungkan gradien piksel ke parameter warp $\mathbf{p} = [t_x, t_y]^\top$:

$$\mathbf{J}_w(\mathbf{x}) = \nabla I_w \cdot \frac{\partial \mathbf{W}}{\partial \mathbf{p}} = \left[\frac{\partial I_w}{\partial x},\ \frac{\partial I_w}{\partial y}\right] \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \left[\frac{\partial I_w}{\partial x},\ \frac{\partial I_w}{\partial y}\right]$$

Untuk translasi, Jacobian warp sama dengan gradien gambar (karena $\frac{\partial \mathbf{W}}{\partial \mathbf{p}} = \mathbf{I}_{2 \times 2}$).

**Langkah 5 — Hitung Hessian $\mathbf{H}$ (2×2):**

$$\mathbf{H} = \sum_{\mathbf{x}} \mathbf{J}_w^\top(\mathbf{x}) \cdot \mathbf{J}_w(\mathbf{x})$$

$$= \sum_{\mathbf{x}} \begin{bmatrix} \left(\frac{\partial I_w}{\partial x}\right)^2 & \frac{\partial I_w}{\partial x}\frac{\partial I_w}{\partial y} \\ \frac{\partial I_w}{\partial x}\frac{\partial I_w}{\partial y} & \left(\frac{\partial I_w}{\partial y}\right)^2 \end{bmatrix}$$

$\mathbf{H}$ adalah matriks simetri positif definit berukuran $2 \times 2$.

**Langkah 6 — Hitung error image $\mathbf{e}$:**

$$e(\mathbf{x}) = \frac{\|\overline{I_w}\|}{\|\overline{T}\|} \cdot \overline{T}(\mathbf{x}) - \overline{I_w}(\mathbf{x})$$

**Langkah 7 — Update parameter:**

$$\Delta \mathbf{p} = \mathbf{H}^{-1} \left(\sum_{\mathbf{x}} \mathbf{J}_w^\top(\mathbf{x}) \cdot e(\mathbf{x})\right)$$

$$\mathbf{p}^{(k+1)} \leftarrow \mathbf{p}^{(k)} + \Delta \mathbf{p}$$

**Konvergensi:** Iterasi berhenti ketika $\|\Delta \mathbf{p}\| < \epsilon = 0.001$ atau iterasi mencapai 50.

**Catatan untuk translasi:** Inversi $\mathbf{H}^{-1}$ hanya berukuran $2 \times 2$:
$$\mathbf{H}^{-1} = \frac{1}{H_{11}H_{22} - H_{12}^2} \begin{bmatrix} H_{22} & -H_{12} \\ -H_{12} & H_{11} \end{bmatrix}$$

### Penerapan Warp ke Track

Hasil akhir $[t_x, t_y]$ dikembalikan ke skala frame asli (bagi faktor resize 0.15):
$$t_x^{full} = t_x^{small} / 0.15, \quad t_y^{full} = t_y^{small} / 0.15$$

Namun pada kode StrongSORT/ECC ini, nilai displacement sudah dalam skala piksel frame asli setelah dikomputasi pada skala kecil dan direscale kembali.

---

## OUTPUT — Matriks Warp W (2×3)

```
        col-0   col-1   col-2 (translasi)
        ──────  ──────  ─────────────────
baris-0 │  1.0    0.0   tx = +0.021967 px
baris-1 │  0.0    1.0   ty = −0.021734 px
```

**File data:** `03_ecc_warp_matrix.csv`

| baris | col-0 | col-1 | col-2 (translasi) |
|:-----:|:-----:|:-----:|:-----------------:|
| 0 | 1.00000000 | 0.00000000 | **+0.02196710 px** |
| 1 | 0.00000000 | 1.00000000 | **−0.02173385 px** |

**Interpretasi fisik:**

```
tx = +0.0220 px → kamera bergerak ke kanan sebesar 0.022 piksel
ty = −0.0217 px → kamera bergerak ke atas sebesar 0.022 piksel
```

Ini adalah gerakan kamera yang sangat kecil (sub-piksel), menunjukkan kamera hampir statis antara frame 29 dan 30. Meski kecil, koreksi ini penting untuk menjaga akurasi prediksi posisi track.

---

## PENERAPAN WARP KE STATE HERO

Setiap track aktif diperbarui posisi pusatnya:

```
Sebelum ECC:
  cx_t-1 = 1200.77,  cy_t-1 = 430.63

Setelah ECC:
  cx_ecc = 1200.77 + 0.0220  = 1200.792
  cy_ecc = 430.63  + (−0.0217) = 430.608

Kecepatan [vx, vy, va, vh] tidak diubah oleh ECC.
```

**Visualisasi konsep ECC:**

```
Frame t-1 (grayscale 162×288):          Frame t (grayscale 162×288):
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │  ▓▓▓ objek-A            │              │  ▓▓▓ objek-A (sedikit   │
  │     ▒▒▒▒▒▒              │   →ECC→     │        bergeser kanan)  │
  │  ████ objek-B            │              │  ████ objek-B            │
  └─────────────────────────┘              └─────────────────────────┘

ECC menemukan: frame t ≈ warp(frame t-1, W)
W = [[1, 0, +0.0220],   ← translate kanan 0.022 px
     [0, 1, -0.0217]]   ← translate atas  0.022 px
```
