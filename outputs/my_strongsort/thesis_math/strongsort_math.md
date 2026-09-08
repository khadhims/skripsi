# Proses Matematika StrongSORT — Langkah demi Langkah

**Objek:** Track #1 | **Frame:** 30 | **Confidence deteksi:** 0.9028
**Crop objek hero:** `hero_crop.png` — ukuran **85 × 106 × 3** (H × W × Channel RGB)

---

## 1. OSNet ReID — Ekstraksi Fitur Appearance

Crop objek (85 × 106 × 3) dimasukkan ke OSNet sebagai **tensor 3D**.  
Output: **vektor 512 dimensi**, lalu di-L2-normalisasi:

```
f  ←  f / ||f||₂
```

8 dimensi pertama dari 512 (setelah normalisasi):

| Dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Nilai | 0.00000 | 0.00000 | +0.03713 | +0.07477 | +0.02165 | +0.04337 | +0.03178 | +0.13600 |

Verifikasi: **‖f‖₂ = 1.000000** ✓

---

## 2. ECC — Kompensasi Gerak Kamera

`cv2.findTransformECC` membandingkan frame t−1 dan t (grayscale, diperkecil 15%), menghasilkan **matriks warp W (2×3)**:

```
        col-0   col-1   col-2 (translasi)
        ──────  ──────  ─────────────────
baris-0 │  1      0     dx = +0.0420 px
baris-1 │  0      1     dy = −0.0227 px
```

- **dx = +0.0420 px** → kamera bergeser ke kanan
- **dy = −0.0227 px** → kamera bergeser ke atas

Setiap track diperbarui posisinya:

```
cx_baru  =  cx_lama + 0.0420
cy_baru  =  cy_lama − 0.0227
```

Kecepatan [vx, vy, va, vh] **tidak diubah** oleh ECC.

---

## 3. NSA Kalman Filter — Langkah Predict

### 3.1 State Vector Sebelum Prediksi

State 8 dimensi — posisi, ukuran, dan kecepatannya:

```
Simbol  │  x       y       a       h      vx      vy      va      vh
────────┼──────────────────────────────────────────────────────────────
x(t−1)  │ 1200.77  430.63  1.2120  86.996  0.4665  0.1390  0.0000  0.1042
```

Keterangan: x = posisi cx, y = posisi cy, a = rasio w/h, h = tinggi objek, vx/vy/va/vh = kecepatan.

### 3.2 Matriks Transisi F (8×8)

Model *constant velocity*: posisi baru = posisi lama + kecepatan (dt = 1).

```
     x   y   a   h   vx  vy  va  vh
  ┌─────────────────────────────────────┐
x │  1   0   0   0  [1]  0   0   0     │
y │  0   1   0   0   0  [1]  0   0     │
a │  0   0   1   0   0   0  [1]  0     │
h │  0   0   0   1   0   0   0  [1]    │
vx│  0   0   0   0   1   0   0   0     │
vy│  0   0   0   0   0   1   0   0     │
va│  0   0   0   0   0   0   1   0     │
vh│  0   0   0   0   0   0   0   1     │
  └─────────────────────────────────────┘
```

Nilai **[1]** di blok kanan atas = pengaruh kecepatan ke posisi (posisi += kecepatan).

### 3.3 Hasil x_pred = F · x

```
Dimensi │ Hitungan                              │ x_pred
────────┼───────────────────────────────────────┼──────────
x       │ 1200.77 + 0.4665 (vx)                 │ 1201.24
y       │  430.63 + 0.1390 (vy)                 │  430.77
a       │   1.212 + 0.0000 (va)                 │   1.2120
h       │  86.996 + 0.1042 (vh)                 │  87.10
vx      │  0.4665 (tetap)                        │   0.4665
vy      │  0.1390 (tetap)                        │   0.1390
va      │  0.0000 (tetap)                        │   0.0000
vh      │  0.1042 (tetap)                        │   0.1042
```

---

## 4. NSA — Penyesuaian Noise Pengukuran

NSA (*Noise Scale Adaptive*) mengecilkan noise pengukuran **R** secara proporsional terhadap confidence:

```
R_k  =  (1 − conf) · R_base
```

Semakin tinggi confidence → R semakin kecil → Kalman lebih percaya pada deteksi.

**Parameter:** h = 87.10 px, sw_pos = 1/20 = 0.0500, conf = 0.9028

### 4.1 Noise Dasar std_base (seolah conf = 0)

```
std_base[x]  =  sw_pos × h  =  0.0500 × 87.10  =  4.3550
std_base[y]  =  sw_pos × h  =  0.0500 × 87.10  =  4.3550
std_base[a]  =  0.1000  (konstanta)
std_base[h]  =  sw_pos × h  =  0.0500 × 87.10  =  4.3550
```

### 4.2 Faktor NSA dan std Terkoreksi

```
Faktor NSA  =  1 − 0.9028  =  0.0972

std_NSA[x]  =  0.0972 × 4.3550  =  0.4232
std_NSA[y]  =  0.0972 × 4.3550  =  0.4232
std_NSA[a]  =  0.0972 × 0.1000  =  0.0097
std_NSA[h]  =  0.0972 × 4.3550  =  0.4232
```

### 4.3 Matriks Noise R = diag(std_NSA²)

```
     x          y          a          h
  ┌──────────────────────────────────────────┐
x │ 0.17906     0          0          0      │
y │    0      0.17906      0          0      │
a │    0         0      0.00009       0      │
h │    0         0          0      0.17906   │
  └──────────────────────────────────────────┘
```

> **Perbandingan:** std_base = 4.355 → std_NSA = 0.423. NSA menurunkan noise 10× karena conf = 0.90 tinggi.

---

## 5. NSA Kalman Filter — Innovation & Update State

### 5.1 Pengukuran z dari Deteksi YOLO

Format: (cx, cy, a, h) = posisi tengah, rasio lebar/tinggi, tinggi.

```
z  =  [1201.5063,  429.9582,  1.2463,  85.2687]
```

### 5.2 Proyeksi State ke Ruang Pengukuran

Matriks H memilih 4 elemen pertama dari state 8D:

```
     x   y   a   h   vx  vy  va  vh
  ┌──────────────────────────────────┐
x │  1   0   0   0   0   0   0   0  │
y │  0   1   0   0   0   0   0   0  │
a │  0   0   1   0   0   0   0   0  │
h │  0   0   0   1   0   0   0   0  │
  └──────────────────────────────────┘

H · x_pred  =  [1201.2351,  430.7727,  1.2120,  87.1004]
```

### 5.3 Innovation y = z − H·x_pred

```
y[x]  =  1201.5063 − 1201.2351  =  +0.2712  (deteksi sedikit di kanan prediksi)
y[y]  =   429.9582 −  430.7727  =  −0.8146  (deteksi sedikit di atas prediksi)
y[a]  =     1.2463 −    1.2120  =  +0.0342  (rasio sedikit lebih lebar)
y[h]  =    85.2687 −   87.1004  =  −1.8316  (deteksi sedikit lebih pendek)
```

### 5.4 Kalman Gain K (8×4)

K menentukan seberapa besar setiap elemen state dikoreksi oleh innovation:

```
K  =  P · Hᵀ · S⁻¹

         z[x]       z[y]       z[a]       z[h]
      ┌─────────────────────────────────────────────┐
x     │ +0.99181   0.00000   0.00000   0.00000      │
y     │  0.00000  +0.99181   0.00000   0.00000      │
a     │  0.00000   0.00000  +0.63413   0.00000      │
h     │  0.00000   0.00000   0.00000  +0.99181      │
vx    │ +0.11583   0.00000   0.00000   0.00000      │
vy    │  0.00000  +0.11583   0.00000   0.00000      │
va    │  0.00000   0.00000  +0.00002   0.00000      │
vh    │  0.00000   0.00000   0.00000  +0.11583      │
      └─────────────────────────────────────────────┘
```

- K[x,:] ≈ 0.992 dan K[h,:] ≈ 0.992 → posisi/ukuran hampir sepenuhnya mengikuti deteksi
- K[vx,:] ≈ 0.116 → kecepatan dikoreksi sedikit (dampak tidak langsung dari perubahan posisi)
- K diagonal dan nyaris nol di luar diagonal → setiap dimensi pengukuran hanya memengaruhi dimensi state yang bersesuaian

### 5.5 Koreksi State: x_corr = x_pred + K·y

```
Dim  │  x_pred     K·y           x_corr
─────┼──────────────────────────────────────────
x    │  1201.2351  + 0.26901  =  1201.5041
y    │   430.7727  − 0.80788  =   429.9649
a    │     1.2120  + 0.02172  =     1.2337
h    │    87.1004  − 1.81664  =    85.2837
vx   │     0.4665  + 0.03142  =     0.4979
vy   │     0.1390  − 0.09435  =     0.0447
va   │     0.0000  + 0.00000  =     0.0000
vh   │     0.1042  − 0.21216  =    −0.1080
```

---

## 6. Gating Mahalanobis

Sebelum menghitung cost, setiap pasangan track–deteksi diuji apakah secara geometri konsisten.
Pasangan yang **d² melebihi threshold** langsung dibuang.

```
d²  =  yᵀ · S⁻¹ · y

Threshold: χ²(df = 4, p = 0.95)  =  9.4877
```

### 6.1 Matriks S — Innovation Covariance (4×4)

S = H·P·Hᵀ + R (kovariansi prediksi yang diproyeksikan + noise pengukuran NSA).

```
        x          y          a          h
     ┌──────────────────────────────────────────┐
x    │  21.856      0          0          0     │
y    │    0       21.856       0          0     │
a    │    0         0        0.0003       0     │
h    │    0         0          0       21.856   │
     └──────────────────────────────────────────┘
```

### 6.2 Invers S⁻¹

```
        x          y          a          h
     ┌──────────────────────────────────────────┐
x    │  0.04576     0          0          0     │
y    │    0       0.04576      0          0     │
a    │    0         0       3875.4        0     │
h    │    0         0          0       0.04576  │
     └──────────────────────────────────────────┘
```

> S⁻¹[a,a] = 3875.4 sangat besar → dimensi rasio a sangat sensitif, karena R[a,a] sangat kecil (noise NSA dimensi a kecil sekali).

### 6.3 Langkah Perhitungan d²

**Langkah 1 — S⁻¹ · y:**

```
(S⁻¹·y)[x]  =  0.04576 × (+0.2712)  =  +0.01241
(S⁻¹·y)[y]  =  0.04576 × (−0.8146)  =  −0.03727
(S⁻¹·y)[a]  =  3875.4  × (+0.0342)  =  +132.71
(S⁻¹·y)[h]  =  0.04576 × (−1.8316)  =  −0.08381
```

**Langkah 2 — d² = yᵀ · (S⁻¹·y):**

```
Dim  │  y[i]        (S⁻¹·y)[i]    y[i] × (S⁻¹·y)[i]
─────┼────────────────────────────────────────────────
x    │ +0.2712   ×   +0.01241   =   +0.003366
y    │ −0.8146   ×   −0.03727   =   +0.030358
a    │ +0.0342   ×  +132.71     =   +4.544651   ← kontributor terbesar
h    │ −1.8316   ×   −0.08381   =   +0.153504
─────┼────────────────────────────────────────────────
     │                  d²  =    4.731879

4.7319  <  9.4877  →  LOLOS GATE ✓
```

---

## 7. Cost Matrix — Appearance Cost dan Fused Cost

### 7.1 Fitur Gallery g vs Fitur Deteksi d

**g** = fitur EMA-smooth track hero setelah 29 frame  
**d** = fitur OSNet dari deteksi frame ini

8 dimensi pertama dari 512:

| Dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **g** | 0.00000 | 0.00000 | +0.03157 | +0.08744 | +0.02451 | +0.04523 | +0.03128 | +0.13075 |
| **d** | 0.00000 | 0.00000 | +0.03713 | +0.07477 | +0.02165 | +0.04337 | +0.03178 | +0.13600 |

### 7.2 Cosine Similarity = g · d

```
cosine_sim  =  Σ (g[i] × d[i])  untuk i = 0 sampai 511

Contoh 3 dimensi yang nilainya besar:

  dim 3:  +0.08744 × +0.07477  =  +0.006539
  dim 7:  +0.13075 × +0.13600  =  +0.017782
  dim 5:  +0.04523 × +0.04337  =  +0.001962
  ... (512 dimensi lainnya) ...
  ──────────────────────────────────────────
  g · d (total 512 dim)  =  0.991866
```

> Nilai mendekati 1.0 → gallery dan deteksi sangat mirip → objek yang sama.

### 7.3 Appearance Cost

```
appearance_cost  =  1 − cosine_sim  =  1 − 0.991866  =  0.008134
```

### 7.4 Fused Cost (mc_lambda = 0.995)

```
fused  =  λ × appearance  +  (1−λ) × maha

       =  0.995 × 0.008134   +   0.005 × 0.216064
       =  0.008093            +   0.001080
       =  0.009173
```

> Nilai 0.009 sangat kecil → pasangan ini dipilih oleh Hungarian Assignment sebagai match.

---

## 8. EMA — Update Fitur Setelah Match

Setelah track berhasil di-match, fitur appearance diperbarui:

```
f_smooth  =  (α × f_prev  +  (1−α) × f_new)  /  ‖...‖₂

α = 0.9  →  fitur lama lebih dominan (stabil terhadap oklusi sesaat)
```

8 dimensi pertama (dim 0 dan 1 kebetulan bernilai 0 untuk objek ini):

```
Dim  │  f_prev        f_new         Perhitungan                    f_weighted
─────┼────────────────────────────────────────────────────────────────────────
2    │ +0.03883    +0.03713   0.9×0.03883 + 0.1×0.03713 = 0.034947+0.003713 │ +0.03866
3    │ +0.06652    +0.07477   0.9×0.06652 + 0.1×0.07477 = 0.059868+0.007477 │ +0.06735
4    │ +0.02286    +0.02165   0.9×0.02286 + 0.1×0.02165 = 0.020574+0.002165 │ +0.02274
5    │ +0.05050    +0.04337   0.9×0.05050 + 0.1×0.04337 = 0.045450+0.004337 │ +0.04979
6    │ +0.03014    +0.03178   0.9×0.03014 + 0.1×0.03178 = 0.027126+0.003178 │ +0.03031
7    │ +0.13744    +0.13600   0.9×0.13744 + 0.1×0.13600 = 0.123696+0.013600 │ +0.13729
```

**Normalisasi L2:**

```
‖f_weighted‖₂  =  0.999181

f_smooth[i]  =  f_weighted[i] / 0.999181

Contoh:
  dim 2:  0.03866 / 0.999181  =  0.038692
  dim 3:  0.06735 / 0.999181  =  0.067400

‖f_smooth‖₂  =  1.000000  ✓
```

> Setelah 29 frame, fitur track sangat stabil. Pengaruh deteksi baru hanya 10% per frame.

---

## 9. New Track — Inisiasi Track Baru

Saat deteksi tidak cocok dengan track manapun, track baru dibuat. Status awal: **Tentative**.  
Menjadi **Confirmed** setelah berhasil di-match selama **n_init = 3 frame**.

### 9.1 State Mean Awal

```
z  =  [cx=1201.51,  cy=429.96,  a=1.2463,  h=85.2687]

mean_0  =  [1201.51,  429.96,  1.2463,  85.2687,  0,  0,  0,  0]
            ←─────── dari deteksi ─────────────→  ←── kecepatan = 0 ──→
```

### 9.2 Covariance Awal = diag(std²)

Formula: std_pos = 2 × sw_pos × h, std_vel = 10 × sw_vel × h

```
sw_pos  =  1/20   =  0.0500
sw_vel  =  1/160  =  0.00625

Dim  │  Formula                         std         var (= std²)
─────┼────────────────────────────────────────────────────────────
x    │  2 × 0.0500 × 85.27          =   8.527       72.708
y    │  2 × 0.0500 × 85.27          =   8.527       72.708
a    │  0.010  (konstanta)           =   0.010        0.000100
h    │  2 × 0.0500 × 85.27          =   8.527       72.708
vx   │  10 × 0.00625 × 85.27        =   5.329       28.401
vy   │  10 × 0.00625 × 85.27        =   5.329       28.401
va   │  0.00001  (konstanta)         ≈   0.000    ≈  0
vh   │  10 × 0.00625 × 85.27        =   5.329       28.401

cov_0  =  diag([72.708, 72.708, 0.0001, 72.708, 28.401, 28.401, ~0, 28.401])
```

> var_posisi (72.7) > var_kecepatan (28.4): posisi absolut lebih tidak pasti  
> daripada asumsi kecepatan awal = 0 (objek diasumsikan berhenti saat pertama terdeteksi).
