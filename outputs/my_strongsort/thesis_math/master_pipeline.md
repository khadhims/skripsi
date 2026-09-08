# Penjabaran Matematis End-to-End: Pipeline YOLO + StrongSORT

**Objek Kajian:** Track #1 (Hero) — Frame ke-30, Video `ramai.mp4`
**Konfigurasi:** YOLOv5 (fine-tuned) + StrongSORT (OSNet-x0.25 ReID)
**Tujuan dokumen:** Menjabarkan secara mekanis dan matematis setiap komponen pipeline,
dari piksel raw hingga bounding box ber-ID yang dirender ke frame.

---

## DAFTAR ISI

1. [Gambaran Umum Pipeline](#1-gambaran-umum-pipeline)
2. [Komponen 1: Output Deteksi YOLO](#2-komponen-1-output-deteksi-yolo)
3. [Komponen 2: OSNet ReID — Ekstraksi Fitur](#3-komponen-2-osnet-reid--ekstraksi-fitur)
4. [Komponen 3: ECC — Kompensasi Gerak Kamera](#4-komponen-3-ecc--kompensasi-gerak-kamera)
5. [Komponen 4: NSA Kalman Filter — Langkah PREDICT](#5-komponen-4-nsa-kalman-filter--langkah-predict)
6. [Komponen 5: NSA Kalman Filter — Langkah UPDATE](#6-komponen-5-nsa-kalman-filter--langkah-update)
7. [Komponen 6: Gating Mahalanobis](#7-komponen-6-gating-mahalanobis)
8. [Komponen 7: Cost Matrix — Appearance & Motion](#8-komponen-7-cost-matrix--appearance--motion)
9. [Komponen 8: Hungarian Assignment](#9-komponen-8-hungarian-assignment)
10. [Komponen 9: EMA Update Fitur Appearance](#10-komponen-9-ema-update-fitur-appearance)
11. [Komponen 10: Manajemen Siklus Hidup Track](#11-komponen-10-manajemen-siklus-hidup-track)
12. [Output Akhir — Track Terkoreksi](#12-output-akhir--track-terkoreksi)

---

## 1. Gambaran Umum Pipeline

### 1.1 Urutan Eksekusi Per Frame

Untuk setiap frame video, StrongSORT menjalankan urutan berikut:

```
Frame t
  │
  ├──[1] YOLO Deteksi → {[x1,y1,x2,y2,conf,cls]_j}  (N deteksi)
  │
  ├──[2] OSNet ReID → {f_j ∈ ℝ^512}  (satu vektor per deteksi)
  │
  ├──[3] ECC → W (2×3 warp matrix) → koreksi posisi semua track aktif
  │
  ├──[4] Kalman PREDICT → {x_pred_i, P_pred_i}  (satu per track aktif)
  │
  ├──[5] Gating Mahalanobis → Mask (M×N) binary
  │
  ├──[6] Cost Matrix → C (M×N) biaya appearance+motion
  │
  ├──[7] Hungarian Assignment → {(i,j)} matched pairs
  │
  ├──[8] Update Track Matched:
  │        - Kalman UPDATE (x_corr, P_corr)
  │        - EMA update fitur gallery
  │
  ├──[9] Mark Missed Track (unmatched track → predict saja)
  │
  ├──[10] Inisiasi New Track (unmatched detection)
  │
  └──[OUTPUT] Track Confirmed dengan time_since_update < 1
              → [x1,y1,x2,y2,track_id,conf,cls] per track
```

### 1.2 Nilai Aktual Frame ke-30, Track Hero #1

| Parameter | Nilai |
|:---------|:------|
| Frame | 30 |
| Hero detection | det_idx=0, [1148.39, 386.64, 1256.62, 473.16, 0.9055, 4] |
| Hero crop | 87 × 108 × 3 (H × W × C, BGR) |
| ECC displacement | dx=+0.0220 px, dy=−0.0217 px |
| Predicted state x | [1201.26, 430.75, 1.212, 87.10, 0.4665, 0.139, 0, 0.1042] |
| Measurement z | [1202.51, 429.90, 1.2509, 86.52] |
| Mahalanobis d² | 0.2134 (< threshold 9.4877 → LOLOS) |
| Cosine similarity | ≈ 1.000 (track sangat stabil) |
| Corrected state x | [1202.50, 429.91, 1.2371, 86.53, 0.6085, 0.0426, ~0, 0.0385] |

---

## 2. Komponen 1: Output Deteksi YOLO

### 2.1 Asal Input

YOLOv5 menerima satu frame video sebagai input:

$$\mathbf{I}_t \in \mathbb{R}^{H \times W \times 3} = \mathbb{R}^{1080 \times 1920 \times 3}$$

Model melakukan forward pass melalui backbone CSPDarknet → neck PANet → detection head YOLOv5, menghasilkan prediksi kotak pada tiga skala (anchor-based).

### 2.2 Output: Vektor 6-Dimensi per Deteksi

Untuk setiap objek yang terdeteksi:

$$\mathbf{b}_j = [x_1^{(j)},\ y_1^{(j)},\ x_2^{(j)},\ y_2^{(j)},\ s^{(j)},\ c^{(j)}] \in \mathbb{R}^6$$

di mana:
- $x_1, y_1$ = koordinat sudut kiri-atas (piksel, dalam sistem koordinat gambar — arah y ke bawah)
- $x_2, y_2$ = koordinat sudut kanan-bawah
- $s$ = skor kepercayaan (*confidence score*) $\in [0,1]$
- $c$ = indeks kelas

**Hero detection (det_idx=0):**

$$\mathbf{b}_0 = [1148.39,\ 386.64,\ 1256.62,\ 473.16,\ 0.9055,\ 4]$$

**Konversi ke format XYAH** (digunakan StrongSORT):

$$c_x = \frac{x_1 + x_2}{2} = \frac{1148.39 + 1256.62}{2} = 1202.51 \text{ px}$$

$$c_y = \frac{y_1 + y_2}{2} = \frac{386.64 + 473.16}{2} = 429.90 \text{ px}$$

$$h = y_2 - y_1 = 473.16 - 386.64 = 86.52 \text{ px}$$

$$w = x_2 - x_1 = 1256.62 - 1148.39 = 108.23 \text{ px}$$

$$a = \frac{w}{h} = \frac{108.23}{86.52} = 1.2509$$

$$\mathbf{z}_{hero} = [c_x,\ c_y,\ a,\ h] = [1202.51,\ 429.90,\ 1.2509,\ 86.52]$$

**Total frame ke-30:** 37 deteksi. File log: `io_logs/log_01_yolo_detection.md`

### 2.3 Visualisasi Output YOLO

```
┌────────────────────────────────────────────────────────────┐
│ Frame ke-30 (1920×1080)                                    │
│                                                            │
│ ┌──────┐ ┌──────┐     ┌──────┐  ┌──────┐  ... (37 total) │
│ │cls=4 │ │cls=4 │     │cls=4 │  │cls=4 │                  │
│ │0.886 │ │0.877 │     │0.863 │  │0.845 │                  │
│ └──────┘ └──────┘     └──────┘  └──────┘                  │
│                                                            │
│                                     ┌────────────────────┐ │
│                                     │ det_idx=0 ★ HERO   │ │
│  x1=1148.4  y1=386.6 ──────────────►│ conf=0.9055        │ │
│                                     │ class=4            │ │
│                                     │                    │ │
│                                     │ cx=1202.5          │ │
│                                     │ cy=429.9           │ │
│                                     │ w=108.2  h=86.5    │ │
│                         x2=1256.6,  │ a=1.2509           │ │
│                         y2=473.2 ───►└────────────────────┘ │
└────────────────────────────────────────────────────────────┘
   Komponen vektor 6D yang divisualisasikan:
   ● x1=1148.39 (sudut kiri-atas, horizontal)
   ● y1=386.64  (sudut kiri-atas, vertikal)
   ● x2=1256.62 (sudut kanan-bawah, horizontal)
   ● y2=473.16  (sudut kanan-bawah, vertikal)
   ● conf=0.9055 (keyakinan deteksi: 90.6%)
   ● class=4    (ID kelas objek)
```

---

## 3. Komponen 2: OSNet ReID — Ekstraksi Fitur

### 3.1 Asal Input

Crop objek diekstrak dari frame menggunakan koordinat bounding box YOLO:

$$\text{Crop}_{hero} = \mathbf{I}_t[y_1:y_2,\ x_1:x_2,\ :] \in \mathbb{R}^{87 \times 108 \times 3}$$

File piksel lengkap (9.396 piksel): `io_logs/02a_reid_input_crop_pixels.csv`
Gambar hero crop: `hero_crop.png`

### 3.2 Pre-processing

$$\text{Crop (87×108×3)} \xrightarrow{\text{Resize}} \text{Tensor (256×128×3)} \xrightarrow{\text{Normalize}} \hat{\mathbf{X}}$$

Normalisasi per-channel (ImageNet):

$$\hat{X}_{c}[h,w] = \frac{X_c[h,w] / 255.0 - \mu_c}{\sigma_c}$$

dengan $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ dan $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$ (RGB).

### 3.3 Arsitektur OSNet-x0.25

OSNet (*Omni-Scale Network*) menggunakan **Omni-Scale Residual Block (OSBlock)** yang menggabungkan konvolusi multi-skala:

**Alur dimensi tensor:**

$$\underbrace{256 \times 128 \times 3}_{\text{input}} \xrightarrow{\text{Conv7×7, s=2}} \underbrace{128 \times 64 \times 64}_{\text{stem}} \xrightarrow{\text{MaxPool3×3, s=2}} \underbrace{64 \times 32 \times 64}_{}$$

$$\xrightarrow{\text{OSBlock Layer1}} \underbrace{64 \times 32 \times 256}_{} \xrightarrow{\text{OSBlock Layer2, s=2}} \underbrace{32 \times 16 \times 384}_{}$$

$$\xrightarrow{\text{OSBlock Layer3, s=2}} \underbrace{16 \times 8 \times 512}_{} \xrightarrow{\text{Conv1×1}} \underbrace{16 \times 8 \times 512}_{}$$

$$\xrightarrow{\text{GAP}} \underbrace{512}_{\text{vektor}} \xrightarrow{\text{FC}} \underbrace{512}_{\text{vektor}} \xrightarrow{L2\text{-norm}} \underbrace{\mathbf{f} \in \mathbb{R}^{512}}_{\text{output, }\|\mathbf{f}\|_2=1}$$

### 3.4 Matematika Konvolusi (Satu Layer)

Diberikan input $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}$ dan filter $\mathbf{W}^{(k)} \in \mathbb{R}^{K \times K \times C_{in}}$ untuk output channel $k$:

$$\mathbf{Y}[h, w, k] = \text{ReLU}\left(\text{BN}\left(\sum_{c=0}^{C_{in}-1} \sum_{p=0}^{K-1} \sum_{q=0}^{K-1} W^{(k)}[p,q,c] \cdot X[hs+p, ws+q, c] + b^{(k)}\right)\right)$$

**Batch Normalization:**

$$\text{BN}(y) = \gamma \cdot \frac{y - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

**Global Average Pooling:**

$$f_{pool,k} = \frac{1}{H' \cdot W'} \sum_{h=0}^{H'-1} \sum_{w=0}^{W'-1} \mathbf{Y}_{final}[h, w, k], \quad k=0,\ldots,511$$

### 3.5 OSBlock — Multi-Scale Feature Aggregation

Satu OSBlock menggabungkan beberapa cabang dengan kernel berbeda:

$$\text{OSBlock}(\mathbf{x}) = \text{Normalize}\left(\sum_{s=1}^{S} g_s(\mathbf{x}) \cdot f_s(\mathbf{x})\right)$$

di mana $f_s$ = konvolusi depthwise dengan kernel skala $s$, dan $g_s(\mathbf{x}) = \sigma(w_s^\top \text{GAP}(\mathbf{x}))$ adalah *channel attention gate* berbasis softmax.

### 3.6 L2 Normalisasi

$$\mathbf{f} = \frac{\mathbf{f}_{pool}}{\|\mathbf{f}_{pool}\|_2} = \frac{\mathbf{f}_{pool}}{\sqrt{\sum_{i=0}^{511} f_{pool,i}^2}}$$

**Properti:** $\|\mathbf{f}\|_2 = 1$ → setiap fitur merupakan titik pada *hypersphere* $S^{511}$ → cosine similarity valid sebagai metrik jarak.

### 3.7 Output: Vektor Fitur 512D

$$\mathbf{f}_{hero} = [0.00000, 0.00000, 0.04068, 0.06071, 0.02983, \ldots, 0.00000]^\top \in \mathbb{R}^{512}$$

$$\|\mathbf{f}_{hero}\|_2 = 1.000000 \checkmark$$

- Dimensi terbesar: dim 279 = 0.20650
- Dimensi nonzero: 319 dari 512 (ReLU menghasilkan banyak nol)

**File lengkap 512D:** `io_logs/02b_reid_output_feat512d.csv`
**Tabel lengkap:** lihat `io_logs/log_02_reid_osnet.md`

---

## 4. Komponen 3: ECC — Kompensasi Gerak Kamera

### 4.1 Asal Input

$$\text{Frame}_{t-1} = \mathbf{I}_{29} \in \mathbb{R}^{1080 \times 1920 \times 3}$$
$$\text{Frame}_{t} = \mathbf{I}_{30} \in \mathbb{R}^{1080 \times 1920 \times 3}$$

### 4.2 Pre-processing

1. Konversi ke grayscale: $I_{gray} = 0.299R + 0.587G + 0.114B$
2. Resize ke 15%: dimensi menjadi $162 \times 288$ piksel

### 4.3 Model Warp Translation-Only

$$\mathbf{W}(\mathbf{x}; t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix} \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \end{bmatrix}$$

Hanya dua parameter yang diestimasi: $\mathbf{p} = [t_x, t_y]^\top$.

### 4.4 Kriteria dan Algoritma ECC

**Kriteria ECC (Enhanced Correlation Coefficient):**

$$\rho^* = \max_{\mathbf{p}} \frac{\bar{\mathbf{T}}^\top \cdot \overline{I(\mathbf{W}(\cdot;\mathbf{p}))}}{\|\bar{\mathbf{T}}\| \cdot \|\overline{I(\mathbf{W}(\cdot;\mathbf{p}))}\|}$$

**Iterasi Lucas-Kanade (satu iterasi ke-$k$):**

**Langkah 1:** Warp gambar target:
$$I_w^{(k)}(\mathbf{x}) = I_t\left(\mathbf{x} + \begin{bmatrix}t_x^{(k)} \\ t_y^{(k)}\end{bmatrix}\right)$$

**Langkah 2:** Zero-mean normalisasi (menghilangkan bias intensitas):
$$\bar{I}_w^{(k)} = I_w^{(k)} - \frac{1}{N}\mathbf{1}^\top I_w^{(k)}, \quad \bar{T} = T - \frac{1}{N}\mathbf{1}^\top T$$

**Langkah 3:** Gradien gambar target:
$$\nabla I_w^{(k)} = \left[\frac{\partial I_w^{(k)}}{\partial x},\ \frac{\partial I_w^{(k)}}{\partial y}\right]$$

**Langkah 4:** Hessian (2×2):
$$\mathbf{H}^{(k)} = \sum_{\mathbf{x}} \begin{bmatrix} \left(\frac{\partial I_w}{\partial x}\right)^2 & \frac{\partial I_w}{\partial x}\frac{\partial I_w}{\partial y} \\ \frac{\partial I_w}{\partial x}\frac{\partial I_w}{\partial y} & \left(\frac{\partial I_w}{\partial y}\right)^2 \end{bmatrix}$$

**Langkah 5:** Error image:
$$e^{(k)}(\mathbf{x}) = \frac{\|\bar{I}_w^{(k)}\|}{\|\bar{T}\|} \cdot \bar{T}(\mathbf{x}) - \bar{I}_w^{(k)}(\mathbf{x})$$

**Langkah 6:** Update parameter:
$$\Delta \mathbf{p}^{(k)} = \left(\mathbf{H}^{(k)}\right)^{-1} \sum_{\mathbf{x}} \nabla I_w^{(k)}(\mathbf{x})^\top \cdot e^{(k)}(\mathbf{x})$$

$$\mathbf{p}^{(k+1)} = \mathbf{p}^{(k)} + \Delta \mathbf{p}^{(k)}$$

**Konvergensi:** $\|\Delta \mathbf{p}^{(k)}\| < 0.001$ atau $k = 50$.

### 4.5 Output: Matriks Warp W (2×3)

$$\mathbf{W}^* = \begin{bmatrix} 1 & 0 & +0.02197 \\ 0 & 1 & -0.02173 \end{bmatrix}$$

Artinya kamera bergerak $\approx +0.022$ px ke kanan dan $\approx -0.022$ px ke atas antara frame 29 dan 30 — gerakan sub-piksel yang menandakan kamera hampir diam.

### 4.6 Koreksi Posisi Setiap Track

$$c_x^{(i)} \leftarrow c_x^{(i)} + t_x = c_x^{(i)} + 0.02197$$
$$c_y^{(i)} \leftarrow c_y^{(i)} + t_y = c_y^{(i)} - 0.02173$$

Untuk hero:
$$[1200.770,\ 430.630] \xrightarrow{\text{ECC}} [1200.792,\ 430.608]$$

---

## 5. Komponen 4: NSA Kalman Filter — Langkah PREDICT

### 5.1 State Space Model

**State 8-dimensi:**

$$\mathbf{x} = [c_x,\ c_y,\ a,\ h,\ \dot{c}_x,\ \dot{c}_y,\ \dot{a},\ \dot{h}]^\top$$

di mana $a = w/h$ adalah rasio aspek dan titik ($\dot{\cdot}$) menunjukkan kecepatan.

**State hero di frame ke-29 (setelah ECC):**

$$\mathbf{x}_{29|29} = [1200.792,\ 430.608,\ 1.2120,\ 86.996,\ 0.4665,\ 0.1390,\ 0.0000,\ 0.1042]^\top$$

### 5.2 Matriks Transisi F (Constant Velocity Model)

$$\mathbf{F} = \begin{bmatrix} \mathbf{I}_4 & \mathbf{I}_4 \\ \mathbf{0}_4 & \mathbf{I}_4 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$

Blok kanan-atas $\mathbf{I}_4$ → posisi baru = posisi lama + kecepatan (Δt = 1 frame).

### 5.3 Matriks Noise Proses Q (8×8)

Memodelkan akselerasi tak terduga objek ($\sigma_w^{pos} = 1/20$, $\sigma_w^{vel} = 1/160$):

$$\sigma_{pos} = \frac{1}{20} \times h = 0.05 \times 87.10 = 4.355 \text{ px}$$
$$\sigma_{vel} = \frac{1}{160} \times h = 0.00625 \times 87.10 = 0.544 \text{ px/f}$$

$$\mathbf{Q} = \mathrm{diag}(4.355^2,\ 4.355^2,\ 0.01^2,\ 4.355^2,\ 0.544^2,\ 0.544^2,\ 10^{-10},\ 0.544^2)$$

### 5.4 Prediksi Mean

$$\mathbf{x}_{30|29} = \mathbf{F} \cdot \mathbf{x}_{29|29}$$

$$= \begin{bmatrix} 1200.792 + 0.4665 \\ 430.608 + 0.1390 \\ 1.2120 + 0.0000 \\ 86.996 + 0.1042 \\ 0.4665 \\ 0.1390 \\ 0.0000 \\ 0.1042 \end{bmatrix} = \begin{bmatrix} 1201.259 \\ 430.747 \\ 1.2120 \\ 87.100 \\ 0.4665 \\ 0.1390 \\ 0.0000 \\ 0.1042 \end{bmatrix}$$

### 5.5 Prediksi Covariance

$$\mathbf{P}_{30|29} = \mathbf{F} \cdot \mathbf{P}_{29|29} \cdot \mathbf{F}^\top + \mathbf{Q}$$

Diagonal P yang diprediksi (nilai aktual):

$$P_{pred}[x,x] = P_{pred}[y,y] = P_{pred}[h,h] = 21.5985$$
$$P_{pred}[a,a] = 0.000162$$
$$P_{pred}[v_x,v_x] = P_{pred}[v_y,v_y] = P_{pred}[v_h,v_h] = 2.7491$$

Ketidakpastian posisi: $\sigma_{x,pred} = \sqrt{21.5985} \approx 4.65$ px (radius elipsoida 1σ).

**File data:** `io_logs/04_kalman_predict_covariance_8x8.csv`

### 5.6 Bounding Box Prediksi

$$w_{pred} = a \times h = 1.2120 \times 87.10 = 105.56 \text{ px}$$

$$[x_1, y_1, x_2, y_2]_{pred} = [1148.48,\ 387.20,\ 1254.04,\ 474.30]$$

---

## 6. Komponen 5: NSA Kalman Filter — Langkah UPDATE

### 6.1 Noise Scale Adaptive (NSA) — Kunci StrongSORT

NSA menskalakan noise pengukuran $\mathbf{R}$ secara **adaptif** berdasarkan confidence deteksi YOLO.

**Intuisi:** Jika YOLO sangat yakin (conf tinggi), noise pengukuran harus kecil → Kalman lebih percaya ke deteksi. Jika conf rendah → noise besar → Kalman lebih percaya ke prediksi.

**Formula NSA:**

$$\mathbf{R}_k = (1 - s_k) \cdot \mathbf{R}_{base}$$

di mana $s_k$ = confidence score pada frame $k$.

Untuk hero ($s = 0.9055$):

$$\text{NSA factor} = 1 - 0.9055 = 0.0945$$

$$\sigma_{NSA,x} = 0.0945 \times 4.355 = 0.4115 \text{ px} \quad (\text{vs } 4.355 \text{ px tanpa NSA})$$

$$\mathbf{R}_{NSA} = \mathrm{diag}(0.1693,\ 0.1693,\ 0.0000894,\ 0.1693)$$

### 6.2 Matriks Observasi H (4×8)

H memilih elemen posisi/ukuran dari state 8D:

$$\mathbf{H} = [\mathbf{I}_4 \mid \mathbf{0}_4] = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}$$

### 6.3 Innovation (Residual Pengukuran)

$$\mathbf{y} = \mathbf{z} - \mathbf{H} \mathbf{x}_{30|29}$$

$$= \begin{bmatrix}1202.51 \\ 429.90 \\ 1.2509 \\ 86.522\end{bmatrix} - \begin{bmatrix}1201.26 \\ 430.75 \\ 1.2120 \\ 87.100\end{bmatrix} = \begin{bmatrix}+1.2488 \\ -0.8474 \\ +0.0389 \\ -0.5780\end{bmatrix}$$

Interpretasi: Deteksi YOLO menunjukkan objek sedikit bergeser ke kanan (+1.25 px), sedikit lebih tinggi (−0.85 px), sedikit lebih lebar (+0.039 rasio), dan sedikit lebih pendek (−0.578 px) dari yang diprediksi.

**File data:** `io_logs/05c_kalman_update_innovation.csv`

### 6.4 Innovation Covariance S (4×4)

$$\mathbf{S} = \mathbf{H} \mathbf{P}_{30|29} \mathbf{H}^\top + \mathbf{R}_{NSA}$$

Karena H dan P diagonal-dominan:

$$\mathbf{S} = \mathrm{diag}(21.5985 + 0.1693,\ 21.5985 + 0.1693,\ 0.000162 + 0.0000894,\ 21.5985 + 0.1693)$$

$$= \mathrm{diag}(21.7677,\ 21.7677,\ 0.0002510,\ 21.7677)$$

**File data:** `io_logs/05a_kalman_update_S_4x4.csv`

### 6.5 Kalman Gain K (8×4)

$$\mathbf{K} = \mathbf{P}_{30|29} \mathbf{H}^\top \mathbf{S}^{-1}$$

Karena S diagonal, $S^{-1} = \mathrm{diag}(1/21.7677,\ 1/21.7677,\ 1/0.0002510,\ 1/21.7677)$:

$$K[x,z_x] = \frac{P_{pred}[x,x]}{S[x,x]} = \frac{21.5985}{21.7677} = 0.9922$$

$$K[a,z_a] = \frac{P_{pred}[a,a]}{S[a,a]} = \frac{0.000162}{0.000251} = 0.6446$$

$$K[v_x,z_x] = \frac{P_{pred}[v_x,x]}{S[x,x]} = \frac{2.4749}{21.7677} = 0.1137$$

**K lengkap (8×4):**

$$\mathbf{K} = \begin{bmatrix} 0.9922 & 0 & 0 & 0 \\ 0 & 0.9922 & 0 & 0 \\ 0 & 0 & 0.6446 & 0 \\ 0 & 0 & 0 & 0.9922 \\ 0.1137 & 0 & 0 & 0 \\ 0 & 0.1137 & 0 & 0 \\ 0 & 0 & 0.0000189 & 0 \\ 0 & 0 & 0 & 0.1137 \end{bmatrix}$$

**File data:** `io_logs/05b_kalman_update_K_8x4.csv`

### 6.6 Koreksi State

$$\mathbf{x}_{30|30} = \mathbf{x}_{30|29} + \mathbf{K} \cdot \mathbf{y}$$

$$\mathbf{K} \cdot \mathbf{y} = \begin{bmatrix} 0.9922 \times 1.2488 \\ 0.9922 \times (-0.8474) \\ 0.6446 \times 0.0389 \\ 0.9922 \times (-0.5780) \\ 0.1137 \times 1.2488 \\ 0.1137 \times (-0.8474) \\ 0.0000189 \times 0.0389 \\ 0.1137 \times (-0.5780) \end{bmatrix} = \begin{bmatrix} +1.2391 \\ -0.8409 \\ +0.02508 \\ -0.5735 \\ +0.1420 \\ -0.09635 \\ +0.00000074 \\ -0.06572 \end{bmatrix}$$

$$\mathbf{x}_{30|30} = \begin{bmatrix} 1201.259 + 1.2391 \\ 430.747 - 0.8409 \\ 1.2120 + 0.02508 \\ 87.100 - 0.5735 \\ 0.4665 + 0.1420 \\ 0.1390 - 0.09635 \\ 0.0000 + 0.0000007 \\ 0.1042 - 0.06572 \end{bmatrix} = \begin{bmatrix} 1202.498 \\ 429.906 \\ 1.2371 \\ 86.527 \\ 0.6085 \\ 0.04265 \\ 0.0000007 \\ 0.03848 \end{bmatrix}$$

### 6.7 Koreksi Covariance

$$\mathbf{P}_{30|30} = \mathbf{P}_{30|29} - \mathbf{K} \cdot \mathbf{S} \cdot \mathbf{K}^\top$$

Diagonal P terkoreksi:

$$P_{corr}[x,x] = P_{corr}[y,y] = P_{corr}[h,h] = 0.1679 \quad (\sigma = 0.41 \text{ px})$$
$$P_{corr}[a,a] = 0.0000575$$
$$P_{corr}[v_x,v_x] = P_{corr}[v_y,v_y] = P_{corr}[v_h,v_h] = 2.4677$$

**Penurunan ketidakpastian:**
$$P_{pred}[x,x] = 21.5985 \xrightarrow{\text{Update}} P_{corr}[x,x] = 0.1679 \quad (129\times \text{ lebih pasti})$$

**Visualisasi:**

```
Distribusi Kalman (Elipsoida Ketidakpastian):

Sebelum Update (PREDIKSI):           Sesudah Update (KOREKSI):
  ●━━━━━━━━━━━━━━━━━━●                       ●━━●
  │  σ_x = ±4.65 px  │                   σ_x = ±0.41 px
  │  (elipsoida besar)│                   (sangat presisi)
  ●━━━━━━━━━━━━━━━━━━●

  cx_pred = 1201.26              cx_corr = 1202.50
  cy_pred = 430.75               cy_corr = 429.91

  YOLO deteksi di (1202.51, 429.90)
  → berada dalam lingkup 1σ prediksi
  → Kalman menarik state mendekati deteksi
  → Elipsoida mengecil drastis setelah update
```

---

## 7. Komponen 6: Gating Mahalanobis

### 7.1 Tujuan

Sebelum menghitung cost appearance, eliminasi pasangan track-deteksi yang **tidak mungkin** secara geometri. Ini menghemat komputasi dan mencegah match yang salah.

### 7.2 Threshold Chi-Square

$$\chi^2(df=4,\ p=0.95) = 9.4877$$

Threshold ini berarti: jika error pengukuran berdistribusi Gaussian, 95% pengukuran yang "benar" akan memiliki $d^2_{Maha} < 9.4877$.

### 7.3 Jarak Mahalanobis

Untuk setiap pasangan track $i$ – deteksi $j$:

$$d^2_{ij} = (\mathbf{z}_j - \mathbf{H}\mathbf{x}_{i,pred})^\top \mathbf{S}_i^{-1} (\mathbf{z}_j - \mathbf{H}\mathbf{x}_{i,pred})$$

di mana $\mathbf{S}_i = \mathbf{H}\mathbf{P}_{i,pred}\mathbf{H}^\top + \mathbf{R}_{base}$ (menggunakan $\mathbf{R}_{base}$, bukan NSA, untuk gating).

Karena S diagonal:

$$d^2_{ij} = \frac{(z_{j,x} - x_{i,cx})^2}{S_{xx}} + \frac{(z_{j,y} - x_{i,cy})^2}{S_{yy}} + \frac{(z_{j,a} - x_{i,a})^2}{S_{aa}} + \frac{(z_{j,h} - x_{i,h})^2}{S_{hh}}$$

**Efek dimensi $a$ (rasio aspek):** $S_{aa}$ sangat kecil ($\approx 0.000262$) → $1/S_{aa} \approx 3816$ sangat besar → perbedaan rasio aspek sedikit saja akan membuat $d^2$ besar → menolak deteksi dari kelas berbeda yang kebetulan dekat secara posisi.

### 7.4 Hasil untuk Hero

**Deteksi hero (det_idx=0):**
$$d^2_{hero} = 0.2134 < 9.4877 \quad \checkmark \text{ LOLOS GATE}$$

**Semua 36 deteksi lain:** $d^2 > 100$ (rata-rata ribuan) → semua ditolak.

File data: `io_logs/06_gating_mahalanobis.csv`

### 7.5 Keluaran: Mask Gate (M×N)

$$\text{Gate}[i,j] = \begin{cases} 1 & \text{jika } d^2_{ij} < 9.4877 \\ 0 & \text{sebaliknya (diblok)} \end{cases}$$

Untuk hero track: hanya 1 dari 37 deteksi yang lolos → mask Gate[0,:] = [1, 0, 0, ..., 0].

---

## 8. Komponen 7: Cost Matrix — Appearance & Motion

### 8.1 Cosine Appearance Cost

Untuk setiap pasangan track $i$ dengan gallery feature $\mathbf{g}_i \in \mathbb{R}^{512}$ dan deteksi $j$ dengan feature $\mathbf{d}_j \in \mathbb{R}^{512}$ (keduanya unit-norm):

$$\text{cos\_sim}_{ij} = \mathbf{g}_i \cdot \mathbf{d}_j = \sum_{k=0}^{511} g_{i,k} \cdot d_{j,k}$$

$$c^{app}_{ij} = 1 - \text{cos\_sim}_{ij}$$

**Properti:**
- $c^{app}_{ij} = 0$ → objek yang persis sama
- $c^{app}_{ij} = 1$ → objek yang sama sekali tidak mirip (orthogonal)
- $c^{app}_{ij} = 2$ → objek yang berlawanan arah fitur (sangat berbeda)

**Untuk hero (track stabil 30 frame):**
$$\text{cos\_sim}_{hero} = \mathbf{g}_{hero} \cdot \mathbf{d}_{hero} \approx 1.000$$
$$c^{app}_{hero} \approx 0.000$$

### 8.2 Motion (Mahalanobis) Cost

$$c^{maha}_{ij} = \frac{d^2_{ij}}{\chi^2_{thresh}} = \frac{d^2_{ij}}{9.4877}$$

(Ternormalisasi ke [0,1] untuk threshold nominal, meski bisa >1 untuk deteksi jauh.)

**Untuk hero:**
$$c^{maha}_{hero} = \frac{0.2134}{9.4877} = 0.0225$$

### 8.3 Fused Cost

$$c^{fused}_{ij} = \lambda_{mc} \cdot c^{app}_{ij} + (1 - \lambda_{mc}) \cdot c^{maha}_{ij}$$

$$= 0.995 \cdot c^{app}_{ij} + 0.005 \cdot c^{maha}_{ij}$$

**Untuk hero:**
$$c^{fused}_{hero} = 0.995 \times 0.000 + 0.005 \times 0.0225 = 0.000112$$

### 8.4 Blok Gate pada Cost Matrix

Pasangan yang diblok gate (Gate[i,j]=0) diberi nilai besar:
$$C_{ij} \leftarrow \infty \quad \text{jika Gate}[i,j] = 0$$

**Matriks cost akhir (1 track × 5 deteksi representatif):**

| | det-0★ | det-1 | det-2 | det-3 | det-4 |
|:---|:------:|:-----:|:-----:|:-----:|:-----:|
| Track-hero | **0.000112** | ∞ | ∞ | ∞ | ∞ |

File data: `io_logs/07_cost_matrix.csv`

---

## 9. Komponen 8: Hungarian Assignment

### 9.1 Formulasi Linear Sum Assignment

Diberikan $\mathbf{C} \in \mathbb{R}^{M \times N}$ (M track, N deteksi):

$$\min_{\mathbf{X}} \sum_{i=1}^{M} \sum_{j=1}^{N} C_{ij} X_{ij}$$

**Subject to:**

$$\sum_{j=1}^{N} X_{ij} \leq 1, \quad \forall i; \qquad \sum_{i=1}^{M} X_{ij} \leq 1, \quad \forall j; \qquad X_{ij} \in \{0,1\}$$

### 9.2 Algoritma Kuhn-Munkres

Algoritma bekerja pada matriks tereduksi untuk menemukan penugasan optimal:

1. **Row reduction:** $C'_{ij} \leftarrow C_{ij} - \min_j C_{ij}$
2. **Column reduction:** $C''_{ij} \leftarrow C'_{ij} - \min_i C'_{ij}$
3. **Find zero assignment:** Cari penugasan maksimal pada elemen nol
4. **Augment via alternating paths** jika penugasan belum optimal

Kompleksitas: $O(M^3)$.

### 9.3 Threshold Penolakan

Setelah assignment, terapkan threshold:

$$\text{Jika } C_{i^*j^*} > \text{max\_dist} = 0.2 \Rightarrow (i^*, j^*) \text{ bukan pasangan valid}$$

### 9.4 Hasil

| Track | Det | Cost | Status |
|:-----:|:---:|:----:|:------:|
| Hero #1 | det-0 | 0.000112 | ✓ MATCHED (cost < 0.2) |

---

## 10. Komponen 9: EMA Update Fitur Appearance

### 10.1 Asal Input

- $\mathbf{f}_{prev}$ = fitur gallery hero (hasil EMA 29 frame sebelumnya), unit-norm
- $\hat{\mathbf{f}}_{new}$ = fitur deteksi baru dari OSNet (Komponen 2), unit-norm

### 10.2 Formula EMA

$$\mathbf{f}_{weighted} = \alpha \cdot \mathbf{f}_{prev} + (1-\alpha) \cdot \hat{\mathbf{f}}_{new}$$

$$\mathbf{f}_{smooth} = \frac{\mathbf{f}_{weighted}}{\|\mathbf{f}_{weighted}\|_2}$$

dengan $\alpha = 0.9$.

### 10.3 Perhitungan Per Dimensi (Contoh Dim 279 — Terbesar)

$$f_{weighted,279} = 0.9 \times 0.20650 + 0.1 \times 0.20650 = 0.20650$$

$$\|\mathbf{f}_{weighted}\|_2 = 1.000000$$

$$f_{smooth,279} = \frac{0.20650}{1.000000} = 0.20650$$

(Karena fitur stabil, perubahan hampir nol.)

### 10.4 Interpretasi α=0.9

$$\mathbf{f}_{smooth,t} = \alpha^t \mathbf{f}_0 + \sum_{k=1}^{t} \alpha^{t-k} (1-\alpha) \mathbf{f}_{new,k}$$

Waktu "pelupaan" eksponensial: fitur dari $n$ frame yang lalu berkontribusi dengan bobot $\alpha^n$.

**Untuk $\alpha = 0.9$:**
- Frame 1 yang lalu: bobot = $0.9^1 = 90\%$
- Frame 5 yang lalu: bobot = $0.9^5 = 59\%$
- Frame 10 yang lalu: bobot = $0.9^{10} = 35\%$
- Frame 22 yang lalu: bobot = $0.9^{22} = 9.8\%$

### 10.5 Output: f_smooth (512D)

$$\mathbf{f}_{smooth} \in \mathbb{R}^{512}, \quad \|\mathbf{f}_{smooth}\|_2 = 1.000000$$

File lengkap: `io_logs/09_ema_update_full512d.csv`

Gallery hero diperbarui untuk digunakan di frame ke-31.

---

## 11. Komponen 10: Manajemen Siklus Hidup Track

### 11.1 Diagram Status

$$\text{UNMATCHED\_DET} \xrightarrow{\text{Inisiasi}} \text{TENTATIVE} \xrightarrow{hits \geq 3} \text{CONFIRMED}$$

$$\text{CONFIRMED/TENTATIVE} \xrightarrow{\text{missed + tsu > max\_age}} \text{DELETED}$$

### 11.2 Inisiasi Track Baru

State mean awal untuk deteksi baru $\mathbf{z}_{new} = [cx, cy, a, h]$:

$$\mathbf{x}_0 = \begin{bmatrix}cx & cy & a & h & 0 & 0 & 0 & 0\end{bmatrix}^\top$$

Covariance awal (besar — ketidakpastian maksimum):

$$\text{std}_0^{pos} = 2 \sigma_w^{pos} h = 2 \times 0.05 \times h = 0.1h$$
$$\text{std}_0^{vel} = 10 \sigma_w^{vel} h = 10 \times 0.00625 \times h = 0.0625h$$

$$\mathbf{P}_0 = \mathrm{diag}\left((0.1h)^2,\ (0.1h)^2,\ 10^{-4},\ (0.1h)^2,\ (0.0625h)^2,\ (0.0625h)^2,\ 10^{-10},\ (0.0625h)^2\right)$$

Untuk deteksi baru dengan $h = 86.52$ px:

$$\mathbf{P}_0 = \mathrm{diag}(74.86,\ 74.86,\ 0.0001,\ 74.86,\ 29.25,\ 29.25,\ 10^{-10},\ 29.25)$$

**Perbandingan:** $P_0[x,x] = 74.86$ vs $P_{corr}[x,x] = 0.168$ untuk track mature (446× lebih tidak pasti).

### 11.3 Lost Track — Propagasi Tanpa Update

Saat track tidak mendapat deteksi:

$$\mathbf{x}_{t|t-1} = \mathbf{F} \mathbf{x}_{t-1|t-1} \qquad \text{(Predict saja)}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^\top + \mathbf{Q} \qquad \text{(P terus membesar)}$$

Tidak ada langkah Update. Fitur gallery $\mathbf{g}$ **tidak diubah** → berguna untuk re-ID nanti.

**Pertumbuhan ketidakpastian per frame tanpa update:**

$$P_{lost}[x,x]^{(n)} \approx P_{corr}[x,x] + n \times \sigma_{pos}^2 = 0.168 + n \times 18.966$$

Setelah 5 frame lost: $P \approx 0.168 + 5 \times 18.966 = 95.0$ ($\sigma_x \approx 9.7$ px)

### 11.4 Penghapusan Track

```
Kondisi DELETE:
  (a) Status TENTATIVE dan time_since_update > 0
      → Deteksi sesaat, tidak perlu dilacak
  (b) Status CONFIRMED dan time_since_update > max_age=30
      → Objek hilang terlalu lama
```

---

## 12. Output Akhir — Track Terkoreksi

### 12.1 Filter Track untuk Output

Hanya track dengan kriteria berikut yang dioutput:

```python
track.is_confirmed() and track.time_since_update < 1
```

### 12.2 Konversi ke Format Bounding Box

Dari state terkoreksi $\mathbf{x}_{30|30}$:

$$[c_x, c_y, a, h] = [1202.498,\ 429.906,\ 1.2371,\ 86.527]$$

$$w = a \times h = 1.2371 \times 86.527 = 107.07 \text{ px}$$

$$x_1 = c_x - w/2 = 1202.498 - 53.535 = 1148.963$$
$$y_1 = c_y - h/2 = 429.906 - 43.264 = 386.642$$
$$x_2 = c_x + w/2 = 1202.498 + 53.535 = 1256.033$$
$$y_2 = c_y + h/2 = 429.906 + 43.264 = 473.170$$

### 12.3 Format Output StrongSORT

$$\mathbf{out}_{hero} = [x_1,\ y_1,\ x_2,\ y_2,\ \text{track\_id},\ \text{conf},\ \text{cls},\ \text{det\_idx}]$$

$$= [1148.963,\ 386.642,\ 1256.033,\ 473.170,\ 1,\ 0.9055,\ 4,\ 0]$$

### 12.4 Visualisasi Track Output Akhir

```
┌────────────────────────────────────────────────────────────────┐
│ Frame ke-30 — OUTPUT TRACK FINAL (setelah Kalman Update)       │
│                                                                │
│ Track #1 (Hero) — CONFIRMED — hits=30 — warna: Biru           │
│                                                                │
│              x1=1149.0                 x2=1256.0              │
│                │                           │                  │
│      y1=386.6──┌───────────────────────────┐                  │
│                │                           │                  │
│                │   ID: 1    conf: 0.906    │                  │
│                │   class: 4                │                  │
│                │                           │                  │
│                │   cx=1202.5               │                  │
│                │   cy=429.9                │                  │
│                │   w=107.1 px h=86.5 px   │                  │
│                │                           │                  │
│      y2=473.2──└───────────────────────────┘                  │
│                                                                │
│ [Perbandingan YOLO raw vs Track output]                        │
│   YOLO raw  : [1148.39, 386.64, 1256.62, 473.16]  conf=0.906  │
│   Track out : [1148.96, 386.64, 1256.03, 473.17]  ID=1        │
│   Δ = sangat kecil (< 1 px) → Kalman hampir mengikuti penuh   │
└────────────────────────────────────────────────────────────────┘
```

### 12.5 Format MOT (untuk Evaluasi)

Untuk file evaluasi MOT format:

```
frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1
30, 1, 1148.963, 386.642, 107.070, 86.527, 0.905550, -1, -1, -1
```

---

## RINGKASAN AKHIR: PERJALANAN PIKSEL KE TRACK

```
FRAME KE-30 (1920×1080×3)
          │
          ▼
[1] YOLO → det_idx=0: [1148.4, 386.6, 1256.6, 473.2, 0.906, 4]
          │              Crop hero: 87×108×3 piksel
          ▼
[2] OSNet → f_hero ∈ ℝ^512, ‖f‖=1 (dim terbesar: f[279]=0.2065)
          │
          ▼
[3] ECC → W=[[1,0,+0.022],[0,1,-0.022]] → koreksi posisi track
          │
          ▼
[4] Kalman PREDICT → x_pred=[1201.26, 430.75, 1.212, 87.10, ...]
                     P_pred[x,x]=21.60 (σ=4.65 px)
          │
          ▼
[5] Gating → d²=0.213 < 9.488 → det-0 LOLOS
             36 deteksi lain: d²>>1000 → DITOLAK
          │
          ▼
[6] Cost  → c_app=0.000, c_maha=0.022, c_fused=0.000112
          │
          ▼
[7] Hungarian → Track#1 ↔ det-0 (cost=0.0001 << max_dist=0.2)
          │
          ▼
[8] Kalman UPDATE → x_corr=[1202.50, 429.91, 1.237, 86.53, 0.609, 0.043, 0, 0.038]
                    P_corr[x,x]=0.168 (σ=0.41 px) ← 129× lebih presisi!
          │
[9] EMA → f_gallery diperbarui (α=0.9 → 90% lama + 10% baru)
          │
          ▼
OUTPUT: [1148.96, 386.64, 1256.03, 473.17, ID=1, conf=0.906, class=4]
```

---

## DAFTAR FILE LOG

| File | Komponen | Deskripsi |
|:-----|:--------:|:---------|
| `log_01_yolo_detection.md` | 1 | 37 deteksi YOLO frame 30, hero 6D vector |
| `01_yolo_detections_frame30.csv` | 1 | Data CSV lengkap 37 deteksi |
| `log_02_reid_osnet.md` | 2 | Arsitektur OSNet, vektor 512D lengkap |
| `02a_reid_input_crop_pixels.csv` | 2 | Semua 9.396 piksel hero crop (B,G,R) |
| `02b_reid_output_feat512d.csv` | 2 | Semua 512 nilai fitur hero (L2-norm=1) |
| `02c_reid_all_dets_features.csv` | 2 | Fitur 512D semua 37 deteksi |
| `log_03_ecc.md` | 3 | Matematika ECC, matriks warp W |
| `03_ecc_warp_matrix.csv` | 3 | Matriks W (2×3) |
| `log_04_kalman_predict.md` | 4 | Prediksi state dan covariance |
| `04_kalman_predict_covariance_8x8.csv` | 4 | Matriks P_pred (8×8) |
| `04_kalman_predict_mean.csv` | 4 | Vektor mean prediksi (8D) |
| `log_05_kalman_update.md` | 5 | NSA, innovation, K, koreksi |
| `05a_kalman_update_S_4x4.csv` | 5 | Matriks S inovasi (4×4) |
| `05b_kalman_update_K_8x4.csv` | 5 | Matriks K gain (8×4) |
| `05c_kalman_update_innovation.csv` | 5 | Vektor y = z − Hx |
| `05d_kalman_corrected_mean.csv` | 5 | State terkoreksi (8D) |
| `05e_kalman_corrected_covariance_8x8.csv` | 5 | Matriks P_corr (8×8) |
| `log_06_gating.md` | 6 | Jarak Mahalanobis semua 37 deteksi |
| `06_gating_mahalanobis.csv` | 6 | Tabel d² semua deteksi |
| `log_07_cost_matrix.md` | 7 | Cosine sim, appearance cost, fused cost |
| `07_cost_matrix.csv` | 7 | Cost matrix semua 37 deteksi |
| `log_08_hungarian.md` | 8 | Algoritma assignment, hasil |
| `log_09_ema.md` | 9 | EMA update 512D gallery |
| `09_ema_update_full512d.csv` | 9 | EMA: f_prev, f_new, f_smooth (512D) |
| `log_10_lifecycle.md` | 10 | Inisiasi, lost, delete track |
| `00_summary.json` | All | Nilai kunci semua komponen |
