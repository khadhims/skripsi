# LOG KOMPONEN 9 — EMA (Exponential Moving Average) — Update Fitur Appearance

**Sumber input:** Fitur gallery hero $\mathbf{f}_{prev}$ + fitur deteksi baru $\mathbf{d}$ (dari Komponen 2)
**Dipanggil:** Hanya saat track BERHASIL di-match (bukan saat lost)
**Parameter:** $\alpha = 0.9$

---

## INPUT

### f_prev — Fitur Gallery Hero (sebelum update)

$\mathbf{f}_{prev}$ adalah vektor 512D hasil EMA 29 frame sebelumnya.
Untuk track yang stabil, $\mathbf{f}_{prev} \approx \mathbf{f}_{hero}$ dari Komponen 2.

```
f_prev = [0.00000000, 0.00000000, 0.04067962, 0.06070972, ...]
‖f_prev‖₂ = 1.000000  (selalu unit-norm)
```

### f_new — Fitur Deteksi Baru (dari Komponen 2)

```
f_new = [0.00000000, 0.00000000, 0.04067962, 0.06070972, ...]
‖f_new‖₂ = 1.000000
```

---

## PROSES INTERNAL — LANGKAH MATEMATIS EMA

### Formula EMA

$$\mathbf{f}_{smooth} = \frac{\alpha \cdot \mathbf{f}_{prev} + (1-\alpha) \cdot \hat{\mathbf{f}}_{new}}{\|\alpha \cdot \mathbf{f}_{prev} + (1-\alpha) \cdot \hat{\mathbf{f}}_{new}\|_2}$$

di mana $\hat{\mathbf{f}}_{new} = \mathbf{f}_{new} / \|\mathbf{f}_{new}\|_2$ (normalisasi ulang input baru).

Dengan $\alpha = 0.9$:
- 90% dari fitur lama dipertahankan
- 10% dari fitur baru ditambahkan

**Mengapa penting dinormalisasi dua kali?**
1. $\hat{\mathbf{f}}_{new}$ dinormalisasi dulu → memastikan skala seragam
2. Output $\mathbf{f}_{smooth}$ dinormalisasi → menjaga unit-norm untuk cosine similarity

### Langkah 1: Normalisasi f_new

$$\hat{\mathbf{f}}_{new} = \frac{\mathbf{f}_{new}}{\|\mathbf{f}_{new}\|_2} = \frac{\mathbf{f}_{new}}{1.000000} = \mathbf{f}_{new}$$

(Sudah unit-norm karena OSNet menghasilkan unit-norm.)

### Langkah 2: Kombinasi Linear

$$\mathbf{f}_{weighted}[i] = 0.9 \times \mathbf{f}_{prev}[i] + 0.1 \times \hat{\mathbf{f}}_{new}[i]$$

### Langkah 3: Normalisasi Output

$$\mathbf{f}_{smooth}[i] = \frac{\mathbf{f}_{weighted}[i]}{\|\mathbf{f}_{weighted}\|_2}$$

---

## OUTPUT — Vektor f_smooth (512D Lengkap)

**File data:** `09_ema_update_full512d.csv`

Format kolom: `dim_idx | f_prev | f_new | 0.9·f_prev + 0.1·f_new (pre-norm) | f_smooth`

Seluruh 512 dimensi (tabel ringkas dengan nilai perwakilan):

| dim | f_prev | f_new | pre_norm | f_smooth |
|:---:|:----------:|:----------:|:----------:|:----------:|
| 0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| 1 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| 2 | 0.04067962 | 0.04067962 | 0.04067962 | 0.04067962 |
| 3 | 0.06070972 | 0.06070972 | 0.06070972 | 0.06070972 |
| 4 | 0.02982996 | 0.02982996 | 0.02982996 | 0.02982996 |
| 5 | 0.04440424 | 0.04440424 | 0.04440424 | 0.04440424 |
| 6 | 0.02890598 | 0.02890598 | 0.02890598 | 0.02890598 |
| 7 | 0.13333309 | 0.13333309 | 0.13333309 | 0.13333309 |
| 8 | 0.10189638 | 0.10189638 | 0.10189638 | 0.10189638 |
| *(... semua 512 dim di file CSV)* | | | | |
| 279 | 0.20649661 | 0.20649661 | 0.20649661 | 0.20649661 |
| 511 | 0.00000104 | 0.00000104 | 0.00000104 | 0.00000104 |

> Karena f_prev ≈ f_new (track stabil), f_smooth ≈ f_prev. Semua 512 nilai lengkap tersedia di file CSV.

**Verifikasi:**
```
‖f_smooth‖₂ = 1.000000  ✓
```

---

## SIMULASI DENGAN TRACK BARU (Fitur Berbeda)

Untuk menunjukkan efek EMA yang lebih nyata, asumsikan track baru dengan fitur yang berbeda:

```
f_prev = [0.0, 0.0, 0.040, 0.061, 0.030, ...]  (stabil)
f_new  = [0.0, 0.0, 0.100, 0.010, 0.080, ...]  (berbeda — mungkin deteksi parsial)

f_weighted[2] = 0.9 × 0.040 + 0.1 × 0.100 = 0.036 + 0.010 = 0.046
f_weighted[3] = 0.9 × 0.061 + 0.1 × 0.010 = 0.055 + 0.001 = 0.056

Perubahan dari f_prev ke f_smooth sangat lambat:
  Dim 2: 0.040 → 0.046  (berubah 15% dari f_new, bukan 100%)
  Dim 3: 0.061 → 0.056  (bergerak menuju f_new 10%)
```

### Keuntungan α=0.9 (besar):

```
Kestabilan: cos(f_smooth, f_prev) ≈ 0.998  (hampir tidak berubah per frame)
Ketahanan: jika 1 frame ada oklusi/noise, hanya mempengaruhi 10% fitur
```

### Kerugian α kecil (misal α=0.1):

```
Adaptasi cepat: f_smooth ≈ f_new per frame
Rentan noise: satu frame buruk mengubah gallery drastis
```

---

## PERBANDINGAN SEBELUM/SESUDAH UPDATE

| Metrik | Sebelum Update | Sesudah Update |
|:------:|:--------------:|:--------------:|
| ‖f_gallery‖₂ | 1.000000 | 1.000000 |
| cos(f_prev, f_smooth) | — | ≈ 1.0000 (track stabil) |
| cos(f_smooth, f_new) | 1.0000 | 1.0000 |
| Dimensi nonzero | 319/512 | 319/512 |

---

## RINGKASAN I/O KOMPONEN 9

| Item | Nilai |
|:-----|:------|
| α (alpha) | 0.9 |
| Dimensi vektor | 512 |
| f_prev norm | 1.000000 |
| f_new norm | 1.000000 |
| f_smooth norm | 1.000000 |
| File lengkap | `09_ema_update_full512d.csv` |
| Pengaruh deteksi baru | 10% per frame |
| Waktu "lupa" objek lama | ≈ ln(0.01)/ln(0.9) ≈ 44 frame |
