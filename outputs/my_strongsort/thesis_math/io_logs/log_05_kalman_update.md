# LOG KOMPONEN 5 — NSA Kalman Filter: Langkah UPDATE

**Sumber input:** State prediksi dari Komponen 4 + Pengukuran YOLO dari Komponen 1
**Kunci NSA:** Noise pengukuran **R** diskalakan oleh `(1 − confidence)` → deteksi ber-confidence tinggi lebih dipercaya

---

## INPUT

### State yang Diprediksi (dari Komponen 4)

```
x_pred = [1201.2585, 430.7473, 1.2120, 87.1002, 0.4665, 0.1390, 0.0000, 0.1042]
P_pred  → lihat log_04 untuk matriks penuh 8×8
```

### Pengukuran YOLO (dari Komponen 1)

```
z = [cx, cy, a, h] = [1202.5073, 429.8998, 1.2509, 86.5222]
confidence = 0.9055
```

---

## PROSES INTERNAL — NSA (Noise Scale Adaptive)

### Noise Pengukuran Dasar (tanpa NSA)

Parameter: $\sigma_w^{pos} = 1/20 = 0.05$, $h_{pred} = 87.10$ px

$$\text{std}_{base}[x] = \sigma_w^{pos} \times h_{pred} = 0.05 \times 87.10 = 4.355 \text{ px}$$
$$\text{std}_{base}[y] = 0.05 \times 87.10 = 4.355 \text{ px}$$
$$\text{std}_{base}[a] = 0.10 \text{ (konstanta aspek)}$$
$$\text{std}_{base}[h] = 0.05 \times 87.10 = 4.355 \text{ px}$$

### Faktor NSA

$$\text{NSA\_factor} = 1 - \text{conf} = 1 - 0.9055 = 0.0945$$

### Noise Terkoreksi NSA

$$\text{std}_{NSA}[x] = 0.0945 \times 4.355 = 0.4115 \text{ px}$$
$$\text{std}_{NSA}[y] = 0.0945 \times 4.355 = 0.4115 \text{ px}$$
$$\text{std}_{NSA}[a] = 0.0945 \times 0.10 = 0.00945$$
$$\text{std}_{NSA}[h] = 0.0945 \times 4.355 = 0.4115 \text{ px}$$

**Perbandingan sebelum/sesudah NSA:**

| Dimensi | std_base (conf=0) | std_NSA (conf=0.9055) | Rasio penurunan |
|:-------:|:-----------------:|:---------------------:|:---------------:|
| x | 4.355 px | 0.4115 px | **10.6× lebih kecil** |
| y | 4.355 px | 0.4115 px | **10.6× lebih kecil** |
| a | 0.100 | 0.00945 | **10.6× lebih kecil** |
| h | 4.355 px | 0.4115 px | **10.6× lebih kecil** |

### Matriks Noise R (4×4) = diag(std_NSA²)

$$\mathbf{R}_{NSA} = \begin{bmatrix} 0.16931 & 0 & 0 & 0 \\ 0 & 0.16931 & 0 & 0 \\ 0 & 0 & 0.00009 & 0 \\ 0 & 0 & 0 & 0.16931 \end{bmatrix}$$

**File data:** `05a_kalman_update_S_4x4.csv`

---

## MATRIKS H — Proyeksi State ke Ruang Pengukuran (4×8)

H memilih 4 elemen pertama (posisi/ukuran) dari state 8D:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}$$

Proyeksi mean:

$$\mathbf{H} \cdot \mathbf{x}_{pred} = [1201.2585,\ 430.7473,\ 1.2120,\ 87.1002]$$

---

## INNOVATION y = z − H·x_pred

$$\mathbf{y} = \mathbf{z} - \mathbf{H}\mathbf{x}_{t|t-1}$$

**File data:** `05c_kalman_update_innovation.csv`

| Dim | Simbol | z (YOLO) | H·x_pred | y = z − H·x_pred | Interpretasi |
|:---:|:------:|:--------:|:--------:|:----------------:|:------------|
| 0 | x | 1202.5073 | 1201.2585 | **+1.2488** | Deteksi 1.25 px lebih ke kanan |
| 1 | y | 429.8998 | 430.7473 | **−0.8474** | Deteksi 0.85 px lebih ke atas |
| 2 | a | 1.2509 | 1.2120 | **+0.0389** | Deteksi sedikit lebih lebar |
| 3 | h | 86.5222 | 87.1002 | **−0.5780** | Deteksi 0.58 px lebih pendek |

Innovation kecil → prediksi Kalman sangat akurat.

---

## INNOVATION COVARIANCE S (4×4)

$$\mathbf{S} = \mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^\top + \mathbf{R}_{NSA}$$

Karena H hanya memilih 4 baris pertama, $\mathbf{H}\mathbf{P}\mathbf{H}^\top$ = blok $4 \times 4$ kiri-atas dari P_pred, kemudian dijumlahkan R_NSA:

**File data:** `05a_kalman_update_S_4x4.csv`

$$\mathbf{S} = \begin{bmatrix} 21.5985+0.16931 & 0 & 0 & 0 \\ 0 & 21.5985+0.16931 & 0 & 0 \\ 0 & 0 & 0.000162+0.00009 & 0 \\ 0 & 0 & 0 & 21.5985+0.16931 \end{bmatrix}$$

$$= \begin{bmatrix} 21.7677 & 0 & 0 & 0 \\ 0 & 21.7677 & 0 & 0 \\ 0 & 0 & 0.0002510 & 0 \\ 0 & 0 & 0 & 21.7677 \end{bmatrix}$$

---

## KALMAN GAIN K (8×4)

$$\mathbf{K} = \mathbf{P}_{t|t-1} \mathbf{H}^\top \mathbf{S}^{-1}$$

Karena S diagonal, $\mathbf{S}^{-1} = \mathrm{diag}(1/21.7677,\ 1/21.7677,\ 1/0.0002510,\ 1/21.7677)$
$= \mathrm{diag}(0.04594,\ 0.04594,\ 3982.07,\ 0.04594)$

**File data:** `05b_kalman_update_K_8x4.csv`

Nilai K lengkap (8 baris × 4 kolom):

| State | K[:,x] | K[:,y] | K[:,a] | K[:,h] | Interpretasi |
|:-----:|:------:|:------:|:------:|:------:|:------------|
| **x** | **0.99223** | 0 | 0 | 0 | Posisi x sangat mengikuti deteksi |
| **y** | 0 | **0.99223** | 0 | 0 | Posisi y sangat mengikuti deteksi |
| **a** | 0 | 0 | **0.64462** | 0 | Rasio aspek, campuran prediksi+deteksi |
| **h** | 0 | 0 | 0 | **0.99223** | Tinggi sangat mengikuti deteksi |
| vx | **0.11370** | 0 | 0 | 0 | Kecepatan x sedikit dikoreksi |
| vy | 0 | **0.11370** | 0 | 0 | Kecepatan y sedikit dikoreksi |
| va | 0 | 0 | **0.0000189** | 0 | Kecepatan aspek hampir tidak berubah |
| vh | 0 | 0 | 0 | **0.11370** | Kecepatan tinggi sedikit dikoreksi |

**Implikasi K diagonal:**
- K[x,x]=0.992 → state x dikoreksi 99.2% dari innovation y[x]
- K[a,a]=0.645 → state aspek hanya 64.5% mengikuti deteksi (noise lebih besar)
- K[vx,x]=0.114 → kecepatan dikoreksi 11.4% dari perubahan posisi

---

## KOREKSI STATE: x_corr = x_pred + K · y

**File data:** `05d_kalman_corrected_mean.csv`

Perhitungan K·y (koreksi):

$$K \cdot \mathbf{y} = \mathbf{K} \cdot [1.2488,\ -0.8474,\ 0.0389,\ -0.5780]^\top$$

| Dim | Simbol | K·y[i] | Hitungan |
|:---:|:------:|:------:|:--------|
| 0 | x | +1.2391 | 0.99223 × 1.2488 |
| 1 | y | −0.8409 | 0.99223 × (−0.8474) |
| 2 | a | +0.02508 | 0.64462 × 0.0389 |
| 3 | h | −0.5735 | 0.99223 × (−0.5780) |
| 4 | vx | +0.14199 | 0.11370 × 1.2488 |
| 5 | vy | −0.09635 | 0.11370 × (−0.8474) |
| 6 | va | +0.000001 | 0.0000189 × 0.0389 |
| 7 | vh | −0.06572 | 0.11370 × (−0.5780) |

**State Terkoreksi:** $\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1} + \mathbf{K}\mathbf{y}$

**File data:** `05d_kalman_corrected_mean.csv`

| Dim | Simbol | x_pred | K·y | **x_corr** |
|:---:|:------:|:------:|:---:|:----------:|
| 0 | x | 1201.2585 | +1.2391 | **1202.4976** |
| 1 | y | 430.7473 | −0.8409 | **429.9064** |
| 2 | a | 1.2120 | +0.0251 | **1.2371** |
| 3 | h | 87.1002 | −0.5735 | **86.5267** |
| 4 | vx | 0.4665 | +0.1420 | **0.6085** |
| 5 | vy | 0.1390 | −0.0964 | **0.0426** |
| 6 | va | 0.0000 | +0.0000 | **0.0000** |
| 7 | vh | 0.1042 | −0.0657 | **0.0385** |

---

## COVARIANCE TERKOREKSI P_corr (8×8)

$$\mathbf{P}_{t|t} = \mathbf{P}_{t|t-1} - \mathbf{K} \cdot \mathbf{S} \cdot \mathbf{K}^\top$$

**File data:** `05e_kalman_corrected_covariance_8x8.csv`

```
          x          y          a          h         vx         vy         va         vh
x    [ 0.1679    0.0000    0.0000    0.0000    0.0192    0.0000    0.0000    0.0000 ]
y    [ 0.0000    0.1679    0.0000    0.0000    0.0000    0.0192    0.0000    0.0000 ]
a    [ 0.0000    0.0000    0.0001    0.0000    0.0000    0.0000    0.0000    0.0000 ]
h    [ 0.0000    0.0000    0.0000    0.1679    0.0000    0.0000    0.0000    0.0192 ]
vx   [ 0.0192    0.0000    0.0000    0.0000    2.4677    0.0000    0.0000    0.0000 ]
vy   [ 0.0000    0.0192    0.0000    0.0000    0.0000    2.4677    0.0000    0.0000 ]
va   [ 0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000 ]
vh   [ 0.0000    0.0000    0.0000    0.0192    0.0000    0.0000    0.0000    2.4677 ]
```

**Penurunan ketidakpastian posisi:**

```
Sebelum update (P_pred[x,x]): 21.5985  →  σ_x = ±4.65 px
Sesudah update (P_corr[x,x]):  0.1679  →  σ_x = ±0.41 px

Penurunan: 21.5985 / 0.1679 = 128.6× lebih pasti!
```

Ini menunjukkan bahwa Kalman Filter sangat efisien menggabungkan prediksi dengan pengukuran ber-confidence tinggi.

---

## RINGKASAN I/O KOMPONEN 5

| Parameter | Nilai |
|:---------|------:|
| Confidence deteksi | 0.9055 |
| NSA factor | 0.0945 |
| std_base_x | 4.355 px |
| std_NSA_x | 0.4115 px |
| Innovation ‖y‖₂ | 1.4816 |
| K diagonal (x,y,h) | 0.9922 |
| K diagonal (a) | 0.6446 |
| P_pred[x,x] → P_corr[x,x] | 21.5985 → 0.1679 |
| x_pred[x] → x_corr[x] | 1201.2585 → 1202.4976 |
