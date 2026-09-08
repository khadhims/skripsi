# LOG KOMPONEN 1 — Output Deteksi YOLO

**Video:** `dataset/videos/ramai.mp4`
**Frame:** 30 (dari total 750 frame)
**Model YOLO:** YOLOv5 (fine-tuned) — `models/7 - tuned/yolov5.pt`
**Total deteksi frame ini:** 37 bounding box

---

## INPUT

Input YOLO adalah tensor gambar RGB dari frame ke-30 dengan dimensi:

```
Frame ke-30  →  Tensor  [1920 × 1080 × 3]  (W × H × C, BGR, dtype=uint8)
```

---

## PROSES INTERNAL (Ringkasan YOLO)

YOLOv5 menjalankan forward pass:
1. Backbone (CSPDarknet): ekstrasi fitur multi-skala
2. Neck (PANet): agregasi piramida fitur
3. Head (YOLOv5 Detection Head): prediksi kotak per anchor

---

## OUTPUT — Vektor 6 Dimensi per Deteksi

Format setiap baris: `[x1, y1, x2, y2, confidence, class_id]`

Keterangan kolom:
- `x1, y1` = sudut **kiri-atas** bounding box (piksel)
- `x2, y2` = sudut **kanan-bawah** bounding box (piksel)
- `confidence` = skor kepercayaan (0.0 – 1.0)
- `class_id` = ID kelas objek (sesuai label YOLO)

### Seluruh 37 Deteksi Frame 30

Data lengkap di: `01_yolo_detections_frame30.csv`

| det_idx | x1 | y1 | x2 | y2 | confidence | class |
|:-------:|:------:|:------:|:------:|:------:|:----------:|:-----:|
| **0 ★** | **1148.39** | **386.64** | **1256.62** | **473.16** | **0.905550** | **4** |
| 1 | 690.06 | 200.01 | 764.61 | 282.51 | 0.898200 | 4 |
| 2 | 902.36 | 140.24 | 967.83 | 213.11 | 0.898025 | 1 |
| 3 | 443.59 | 145.01 | 497.57 | 190.27 | 0.893523 | 4 |
| 4 | 999.50 | 25.32 | 1054.69 | 63.15 | 0.869840 | 4 |
| 5 | 292.47 | 246.01 | 330.85 | 294.97 | 0.843054 | 2 |
| 6 | 649.44 | 52.89 | 701.96 | 93.52 | 0.835694 | 1 |
| 7 | 514.34 | 304.51 | 541.33 | 343.55 | 0.831021 | 0 |
| 8 | 349.69 | 363.55 | 396.58 | 396.63 | 0.827993 | 0 |
| 9 | 846.96 | 653.61 | 983.59 | 757.89 | 0.817478 | 4 |
| 10 | 944.37 | 846.20 | 998.69 | 885.87 | 0.816947 | 0 |
| 11 | 449.59 | 189.72 | 491.97 | 218.52 | 0.810078 | 2 |
| 12 | 265.18 | 226.73 | 326.88 | 279.93 | 0.775952 | 4 |
| 13 | 733.45 | 146.80 | 765.23 | 169.44 | 0.768010 | 0 |
| 14 | 925.04 | 1.00 | 946.25 | 15.29 | 0.767647 | 2 |
| 15 | 448.04 | 421.88 | 527.02 | 522.05 | 0.755937 | 1 |
| 16 | 1453.65 | 0.00 | 1491.03 | 30.01 | 0.738141 | 1 |
| 17 | 0.04 | 693.89 | 34.46 | 817.48 | 0.688895 | 4 |
| 18 | 1194.82 | 553.29 | 1243.08 | 598.43 | 0.678902 | 0 |
| 19 | 533.01 | 431.70 | 574.65 | 487.95 | 0.661810 | 0 |
| 20 | 1195.53 | 553.81 | 1245.17 | 618.18 | 0.655224 | 0 |
| 21 | 1204.77 | 554.87 | 1246.92 | 629.79 | 0.653113 | 0 |
| 22 | 470.04 | 329.58 | 500.79 | 350.51 | 0.630513 | 0 |
| 23 | 999.47 | 51.99 | 1024.81 | 81.86 | 0.603697 | 2 |
| 24 | 101.35 | 608.45 | 145.24 | 656.94 | 0.591749 | 0 |
| 25 | 448.04 | 424.19 | 527.65 | 518.62 | 0.590890 | 4 |
| 26 | 669.68 | 87.26 | 698.32 | 122.04 | 0.410114 | 2 |
| 27 | 959.02 | 73.72 | 983.97 | 92.29 | 0.404832 | 0 |
| 28 | 711.45 | 156.01 | 735.27 | 177.37 | 0.343565 | 0 |
| 29 | 169.63 | 536.13 | 192.52 | 561.35 | 0.332514 | 0 |
| 30 | 953.06 | 764.18 | 1012.00 | 808.17 | 0.314082 | 0 |
| 31 | 668.21 | 854.41 | 804.11 | 994.11 | 0.313048 | 1 |
| 32 | 18.36 | 602.51 | 41.75 | 632.91 | 0.303553 | 0 |
| 33 | 73.27 | 451.02 | 110.39 | 482.50 | 0.241718 | 2 |
| 34 | 964.26 | 73.96 | 989.38 | 91.88 | 0.227176 | 0 |
| 35 | 376.60 | 319.44 | 401.49 | 354.42 | 0.142694 | 0 |
| 36 | 61.84 | 575.17 | 87.81 | 609.97 | 0.117884 | 0 |

> **★ det_idx=0** adalah objek **Hero** yang dilacak (Track #1).

---

## INTERPRETASI HERO DETECTION

```
Vektor 6D Hero:
  [x1, y1, x2, y2, conf,  class]
  [1148.39, 386.64, 1256.62, 473.16, 0.9055, 4]

Lebar bounding box  : x2 - x1 = 1256.62 - 1148.39 = 108.23 px
Tinggi bounding box : y2 - y1 =  473.16 - 386.64  =  86.52 px
Pusat (cx, cy)      : cx = (1148.39 + 1256.62) / 2 = 1202.51 px
                      cy = ( 386.64 +  473.16) / 2 =  429.90 px
Rasio aspek (a)     : a  = lebar / tinggi = 108.23 / 86.52 = 1.2509
```

**Visualisasi Bounding Box YOLO — Frame 30:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Frame 1920×1080                                                 │
│                                                                 │
│  ... (36 objek lain di berbagai posisi) ...                     │
│                                                                 │
│                           ┌──────────────────────┐             │
│                           │  class=4  conf=90.6% │  ← Hero     │
│                           │                      │  (det_idx=0)│
│                           │  cx=1202.5           │             │
│                           │  cy=429.9            │             │
│                           │  w=108.2  h=86.5     │             │
│                           │                      │             │
│      x1=1148.4, y1=386.6  └──────────────────────┘             │
│                                          x2=1256.6, y2=473.2   │
└─────────────────────────────────────────────────────────────────┘

Komponen Vektor 6D yang terlabeli:
  x1=1148.39 ──┐  (sudut kiri-atas, kolom/horizontal)
  y1=386.64  ──┘  (sudut kiri-atas, baris/vertikal)
  x2=1256.62 ──┐  (sudut kanan-bawah, kolom/horizontal)
  y2=473.16  ──┘  (sudut kanan-bawah, baris/vertikal)
  conf=0.9055    (tingkat keyakinan model: 90.6%)
  class=4        (ID kelas; sesuai label dataset)
```

---

## KONVERSI KE FORMAT XYAH (untuk StrongSORT)

StrongSORT tidak menggunakan format `[x1,y1,x2,y2]` secara langsung.
Deteksi dikonversi ke format `z = [cx, cy, a, h]`:

```
cx = (x1 + x2) / 2  = (1148.39 + 1256.62) / 2  = 1202.51
cy = (y1 + y2) / 2  = ( 386.64 +  473.16) / 2  =  429.90
h  = y2 - y1        =  473.16 - 386.64          =   86.52
w  = x2 - x1        = 1256.62 - 1148.39         =  108.23
a  = w / h           =  108.23 / 86.52           =    1.2509

Vektor pengukuran z = [1202.51, 429.90, 1.2509, 86.52]
```
