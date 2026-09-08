# LOG KOMPONEN 10 — Track Lifecycle Management

**Sumber input:** Hasil Hungarian Assignment (Komponen 8)
**Topik:** Inisiasi track baru, kondisi lost, dan penghapusan track

---

## TIGA SKENARIO LIFECYCLE

```
DETEKSI BARU ──────────────────────────────────┐
                                               │
         Tidak cocok track apapun?             │
              ↓ Ya                             │ Tidak
    [INISIASI TRACK BARU]                      │
    Status: TENTATIVE                          ↓
    hits=1, age=1                   [UPDATE TRACK YANG MATCH]
         │                              hits++, age++
         │ (Berlanjut ke frame berikut) time_since_update=0
         ↓                                     │
    hits >= n_init=3?                           │
     Tidak → tetap Tentative           TRACK MATCH → Kalman Update
     Ya    → CONFIRMED                     + EMA Update

TRACK TIDAK MATCH DETEKSI APAPUN ──────────────┐
         │ (frame ini tidak terdeteksi)         │
         ▼                                      │
  [MARK MISSED]                                 │
  time_since_update++                           │
  age++                                         │
         │                                      │
  time_since_update > max_age=30?               │
     Ya → [DELETE]                              │
     Tidak → Kalman Predict saja (no update)    │
```

---

## 10A. INISIASI TRACK BARU

### Kondisi Pemicu

Deteksi yang **tidak cocok** dengan track manapun setelah dua tahap Hungarian:
1. Stage 1: matching terhadap track Confirmed (pakai appearance + Mahalanobis)
2. Stage 2: matching terhadap track Tentative (pakai IoU saja)

Jika tidak cocok di kedua stage → **unmatched detection** → track baru.

### State Mean Awal

Dari pengukuran YOLO: $\mathbf{z} = [cx, cy, a, h]$

$$\mathbf{x}_0 = \begin{bmatrix} cx \\ cy \\ a \\ h \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1202.51 \\ 429.90 \\ 1.2509 \\ 86.52 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}$$

Empat elemen terakhir = 0 → asumsi kecepatan awal = nol.

### Covariance Awal (Matriks Ketidakpastian Besar)

Pada track baru, posisi aktual memang diketahui dari deteksi, namun ketidakpastiannya masih besar (deteksi bisa noise). Kecepatan awal sama sekali tidak diketahui.

Parameter: $\sigma_w^{pos} = 1/20 = 0.05$, $\sigma_w^{vel} = 1/160 = 0.00625$, $h_0 = 86.52$ px

**Standar deviasi awal:**

$$\text{std}_0[x] = 2 \times \sigma_w^{pos} \times h = 2 \times 0.05 \times 86.52 = 8.652 \text{ px}$$
$$\text{std}_0[y] = 2 \times 0.05 \times 86.52 = 8.652 \text{ px}$$
$$\text{std}_0[a] = 0.01 \text{ (konstanta aspek)}$$
$$\text{std}_0[h] = 2 \times 0.05 \times 86.52 = 8.652 \text{ px}$$
$$\text{std}_0[v_x] = 10 \times \sigma_w^{vel} \times h = 10 \times 0.00625 \times 86.52 = 5.408 \text{ px/f}$$
$$\text{std}_0[v_y] = 10 \times 0.00625 \times 86.52 = 5.408 \text{ px/f}$$
$$\text{std}_0[v_a] = 10^{-5} \text{ (hampir nol)}$$
$$\text{std}_0[v_h] = 10 \times 0.00625 \times 86.52 = 5.408 \text{ px/f}$$

**Covariance awal** $\mathbf{P}_0 = \mathrm{diag}(\text{std}_0^2)$:

$$\mathbf{P}_0 = \begin{bmatrix} 74.857 & & & & & & & \\ & 74.857 & & & & & & \\ & & 0.0001 & & & & & \\ & & & 74.857 & & & & \\ & & & & 29.247 & & & \\ & & & & & 29.247 & & \\ & & & & & & 10^{-10} & \\ & & & & & & & 29.247 \end{bmatrix}$$

**Perbandingan dengan track yang sudah lama (Track #1 di frame 30):**

| Elemen | P_0 (track baru) | P_corr (Track #1, frame 30) |
|:------:|:----------------:|:---------------------------:|
| P[x,x] | **74.857** | **0.1679** |
| P[y,y] | **74.857** | **0.1679** |
| P[vx,vx] | **29.247** | **2.4677** |

→ Track baru jauh lebih tidak pasti dibanding track yang sudah ter-establish (446× lebih besar untuk posisi!).

### Status Awal: TENTATIVE

```
Track baru:
  state     = TENTATIVE
  hits      = 1
  age       = 1
  time_since_update = 0
  features  = [f_deteksi]   (satu vektor dari frame ini)
```

Track Tentative **tidak diikutsertakan dalam output** → tidak tampil di video hasil tracking.

### Menjadi CONFIRMED

Track menjadi **Confirmed** setelah `n_init = 3` frame berturut-turut berhasil di-match:

```
Frame 1: hits=1, TENTATIVE
Frame 2: hits=2, TENTATIVE
Frame 3: hits=3 ≥ n_init=3 → CONFIRMED  ← mulai tampil di output
```

---

## 10B. LOST TRACK — Prediksi Tanpa Update

### Kondisi Pemicu

Track yang sudah Confirmed **tidak mendapat pasangan deteksi** di frame ini.

### Apa yang Terjadi Matematis?

Kalman Filter hanya menjalankan langkah **Predict**, tanpa Update:

$$\mathbf{x}_{t|t-1} = \mathbf{F} \cdot \mathbf{x}_{t-1|t-1}$$
$$\mathbf{P}_{t|t-1} = \mathbf{F} \cdot \mathbf{P}_{t-1|t-1} \cdot \mathbf{F}^\top + \mathbf{Q}$$

**Tidak ada Update** → state terus bergerak sesuai kecepatan terakhir, tapi:
- **Covariance P terus membesar** setiap frame (karena +Q tanpa pengurangan dari K)
- Ketidakpastian posisi meningkat seiring waktu

**Simulasi covariance P[x,x] untuk track yang lost:**

| Frame sejak lost | P[x,x] | σ_x |
|:----------------:|:------:|:---:|
| 0 (saat match terakhir) | 0.168 | 0.41 px |
| 1 | 0.168 + 4.355² = 19.13 | 4.37 px |
| 2 | 19.13 + 19.13 = 38.27 | 6.19 px |
| 5 | ≈ 95.7 | 9.78 px |
| 10 | ≈ 191.3 | 13.8 px |
| 30 (max_age) | ≈ 574 | 23.9 px |

Semakin lama track tidak terdeteksi → ketidakpastian posisi semakin besar → gating semakin ketat (nilai d² threshold tetap 9.4877, tapi P besar membuat S besar → jarak Mahalanobis kecil → mudah lolos, tapi appearance berbeda).

**Fitur appearance (EMA) tidak diupdate saat lost:**

$$\mathbf{f}_{gallery} \leftarrow \text{tidak berubah}$$

Gallery feature tetap dari frame match terakhir → berguna untuk re-identification jika objek muncul kembali.

**State counter:**
```
time_since_update++   (bertambah setiap frame tanpa match)
age++                 (bertambah setiap frame, baik match maupun tidak)
hits                  (tidak bertambah)
```

---

## 10C. PENGHAPUSAN TRACK (DELETE)

### Kondisi Delete 1: Track Tentative tidak Match

Track dengan status **Tentative** yang tidak mendapat match → langsung dihapus.
(Tujuan: hindari polusi dari deteksi false positive sesaat.)

```
Track (TENTATIVE, hits=2) + tidak match di frame 3
→ time_since_update > 0 dan status TENTATIVE → DELETE
```

### Kondisi Delete 2: Max Age Terlampaui

Track **Confirmed** yang sudah tidak terdeteksi selama `max_age = 30` frame:

```
if time_since_update > max_age:
    track.state = DELETED
```

```
Contoh:
Frame 30: match terakhir   → time_since_update=0
Frame 31: tidak match       → time_since_update=1
Frame 32: tidak match       → time_since_update=2
...
Frame 60: tidak match       → time_since_update=30 > max_age=30 → DELETE
```

### Dampak Penghapusan

Track yang dihapus:
- Dikeluarkan dari daftar track aktif
- Tidak menghasilkan output bounding box
- ID-nya **tidak digunakan kembali** (ID monoton naik → tidak ada ID reuse)

---

## RINGKASAN STATUS LIFECYCLE

```
┌──────────────────────────────────────────────────────┐
│              STATUS TRACK                            │
│                                                      │
│  TENTATIVE ─── hits≥3 ──────→ CONFIRMED             │
│      │                             │                 │
│  tidak match                   tidak match           │
│  (Tentative)                   (Confirmed)           │
│      ↓                             ↓                 │
│   DELETED ←── time_since_update > max_age ──────────┘│
│                                                      │
│  Parameter:                                          │
│    n_init   = 3   (frame untuk jadi Confirmed)       │
│    max_age  = 30  (frame Lost sebelum Delete)        │
└──────────────────────────────────────────────────────┘
```

### Status Track Hero di Frame 30

```
Track #1:
  state    = CONFIRMED   (sudah ≥ 3 frame match)
  hits     = 30          (berhasil match 30 frame berturut-turut)
  age      = 30          (sudah ada 30 frame)
  time_since_update = 0  (baru saja di-match di frame ini)
```

---

## PARAMETER LIFECYCLE LENGKAP

| Parameter | Nilai | Efek |
|:---------|:-----:|:-----|
| n_init | 3 | Frame match sebelum Confirmed |
| max_age | 30 | Frame Lost sebelum Delete |
| max_dist (appearance) | 0.2 | Threshold cost appearance |
| max_iou_distance | 0.7 | Threshold cost IoU (stage 2) |
| nn_budget | 100 | Maksimum fitur gallery per track |
