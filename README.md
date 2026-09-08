# Skripsi Tracking Pipeline (YOLO + StrongSORT)

Proyek ini merupakan pipeline deteksi dan tracking objek (pekerja APD) menggunakan YOLOv5/YOLOv8 + StrongSORT untuk keperluan skripsi. Mencakup pelatihan model, optimasi hyperparameter, analisis sensitivitas, dan evaluasi metrik MOT.

## 1) Ringkasan Folder Utama

- `main.py`
  - Runner utama tracking realtime/video (YOLO + StrongSORT) untuk generate prediksi MOT.
- `baseline.ipynb`
  - Notebook Kaggle untuk pelatihan model baseline YOLOv5 dan YOLOv8 (500 epoch, seed 42).
- `hyperparameter_optimization.ipynb`
  - Notebook Google Colab untuk grid search + Optuna hyperparameter tuning (batch size, imgsz, learning rate, optimizer).
- `analisis_sensitivitas.ipynb`
  - Notebook Kaggle untuk analisis sensitivitas model terhadap variasi parameter deteksi/tracking.
- `analysis/comparison/`
  - Runner dan modul analisis perbandingan:
    - Tracking vs YOLO-only
    - YOLOv5-StrongSORT vs YOLOv8-StrongSORT
- `evaluation/mot/`
  - Evaluasi metrik MOT (MOTA, IDF1, dll).
- `benchmarks/`
  - Benchmark performa pipeline.
- `pipelines/detection/`
  - Testing deteksi YOLO (tanpa tracking) dan export video anotasi.
- `tools/data_prep/`
  - Utility persiapan data (extract frame, convert annotation CSV ke format MOT).
- `configs/experiments/`
  - Konfigurasi eksperimen berbasis JSON.
- `outputs/reports/`
  - Output laporan CSV hasil analisis.

## 2) Struktur Dataset

```
dataset/
├── videos/              # Video input (ramai.mp4, sepi.mp4, sangat sepi.mp4)
└── tracker_gt/          # Ground truth MOT per skenario
    ├── ramai/
    │   ├── gt.txt       # Anotasi MOT (frame,id,x,y,w,h,conf,class,visibility)
    │   └── labels.txt   # Daftar kelas (no_gloves, no_hairnet, no_mask)
    ├── sepi/
    └── sangat-sepi/
```

## 3) Struktur Model

```
models/
├── 1 - baseline/        # Model baseline (yolov5.pt, yolov8.pt)
├── 2 - epochs/          # Variasi epochs (100–500.pt)
├── 3 - batch size/      # Variasi batch size (8, 16, 32, 64.pt)
├── 4 - imgsz/           # Variasi image size (640, 832, 1088.pt)
├── 5 - learning rate/   # Variasi lr0 (1e-2, 1e-3, 1e-4.pt)
├── 6 - optimized/       # Model hasil optimasi Optuna
└── 7 - tuned/           # Model terbaik final (yolov5.pt, yolov8.pt)
```

> File `.pt` dan `.mp4` dikelola dengan **Git LFS**.

## 4) Setup Environment

Gunakan virtual environment proyek:

```bash
cd /home/kopet/skripsi
source venv/bin/activate
pip install -r requirements.txt
```

## 5) Menjalankan Tracking Utama

Jalankan:

```bash
python main.py
```

Catatan penting:

- `main.py` saat ini masih menggunakan beberapa path hardcoded, terutama:
  - model di fungsi `load_model()`
  - video input pada blok `if __name__ == "__main__"`
  - output prediksi MOT di variabel `pred_file`
- Silakan sesuaikan path tersebut dengan data/eksperimen Anda.

## 6) Analisis Perbandingan Utama

### 4.1 YOLOv5-StrongSORT vs YOLOv8-StrongSORT

```bash
python analysis/comparison/run_v5_vs_v8_strongsort.py --strict-frame-range
```

Opsional:

```bash
python analysis/comparison/run_v5_vs_v8_strongsort.py \
  --config configs/experiments/v5_vs_v8_strongsort.json \
  --output outputs/reports/v5_vs_v8_strongsort_metrics.csv \
  --summary-output outputs/reports/v5_vs_v8_strongsort_summary.csv
```

### 4.2 Tracking vs YOLO-only

```bash
python analysis/comparison/run_tracking_vs_yolo.py
```

Jika file YOLO-only sudah tersedia dan tidak ingin generate ulang:

```bash
python analysis/comparison/run_tracking_vs_yolo.py --skip-generate-yolo-only
```

## 7) Evaluasi MOT Satu Pasangan File

```bash
python evaluation/mot/run_mot_validation.py \
  --gt dataset/tracker-val/vid3-gt.txt \
  --pred dataset/tracker-results/iterasi-1/v8-medium-it2-vid3-pred.txt \
  --iou 0.7
```

Output tabel MOT menampilkan metrik lengkap (termasuk MOTA, IDF1, FP, FN, IDSW, IDTP, IDFP, IDFN, IDP, IDR, dll).

## 7.1) Tutorial Lengkap: Validasi Semua Metrik Tracking + Speed

Langkah ini mencakup metrik tracking (IDF1, MOTA, FN, FP, IDSW, IDTP, IDFP, IDFN) dan performa (FPS + inference speed).

### A) Pastikan file GT & prediksi dalam format MOT

- GT: `dataset/tracker-val/*.txt`
- Pred: `dataset/tracker-results/*-pred.txt`

Format baris MOT (9 kolom):

```
frame,id,x,y,w,h,conf,class,visibility
```

### B) Jalankan evaluasi MOT (semua metrik tracking)

```bash
python evaluation/mot/run_mot_validation.py \
  --gt dataset/tracker-val/vid3-gt.txt \
  --pred dataset/tracker-results/iterasi-1/v8-medium-it2-vid3-pred.txt \
  --iou 0.7 \
  --name V8-VID3
```

Tabel output MOT sudah memuat metrik yang Anda butuhkan: IDF1, MOTA, FP, FN, IDSW, IDTP, IDFP, IDFN.

### C) Ukur FPS dan inference speed

Ada dua opsi:

1. **Benchmark pipeline langsung** (paling sederhana)

```bash
python benchmarks/run_pipeline_benchmark.py \
  --video dataset/videos/vid_3.mp4 \
  --model models/yolov8/iterasi-2/v8-medium.pt \
  --conf 0.25 \
  --iou 0.3 \
  --imgsz 1088 \
  --max-frames 200
```

Output menampilkan `Average FPS` dan `Avg Frame Time (ms)`.

2. **Sekaligus di laporan evaluasi** (untuk perbandingan multi-eksperimen)

```bash
python analysis/comparison/run_v5_vs_v8_strongsort.py --speed-frames 200
```

atau

```bash
python analysis/comparison/run_tracking_vs_yolo.py --speed-frames 200
```

CSV hasilnya akan memiliki kolom `avg_fps` dan `avg_inference_ms` untuk tracking.

## 8) Benchmark Pipeline

```bash
python benchmarks/run_pipeline_benchmark.py \
  --video dataset/videos/vid_3.mp4 \
  --max-frames 200
```

## 9) Detection-Only Utilities

### 9.1 Quick Detection Test (display + latency log di terminal)

```bash
python pipelines/detection/test_detection.py \
  --model models/yolov8/iterasi-2/v8-medium.pt \
  --video dataset/videos/vid_3.mp4
```

### 9.2 Export Video Detection-Only (tanpa tracking)

```bash
python pipelines/detection/yolo_detect_only.py \
  --model models/yolov8/iterasi-2/v8-medium.pt \
  --source dataset/videos/vid_3.mp4 \
  --output video-results/detection-only/vid3_detect.mp4
```

## 10) Data Preparation Tools

### 10.1 Extract Frame dari Video

```bash
python tools/data_prep/extract_frames.py \
  --video dataset/videos/vid_3.mp4 \
  --output-dir dataset/frames/vid3 \
  --interval-sec 1
```

Atau berbasis jumlah frame:

```bash
python tools/data_prep/extract_frames.py \
  --video dataset/videos/vid_3.mp4 \
  --output-dir dataset/frames/vid3 \
  --interval-frames 10
```

### 10.2 Convert CSV Annotation ke MOT GT

```bash
python tools/data_prep/mot_val_convert.py \
  --csv path/to/annotation.csv \
  --output dataset/tracker-val/vidX-gt.txt \
  --video-width 1920 \
  --video-height 1080 \
  --track-id-col object_id
```

## 11) Analisis File Log Inferensi

Jika Anda punya file log inferensi sendiri, analisis dengan:

```bash
python analysis/inference_logs/analyze_inference.py \
  --log path/to/inference_log.txt \
  --output outputs/reports/inference_latency_summary.csv
```

Catatan: pipeline saat ini tidak lagi otomatis membuat `inference_log.txt` ke file.

## 12) Konfigurasi Eksperimen

Konfigurasi default perbandingan ada di:

- `configs/experiments/v5_vs_v8_strongsort.json`

Anda bisa mengubah:

- `settings` (`iou_threshold`, `conf_threshold`, `imgsz`)
- daftar `experiments` (path GT, prediksi tracking, video, model)

## 13) Lokasi Output Penting

- Detail + summary perbandingan model:
  - `outputs/reports/v5_vs_v8_strongsort_metrics.csv`
  - `outputs/reports/v5_vs_v8_strongsort_summary.csv`
- Detail + summary tracking vs YOLO-only:
  - `outputs/reports/tracking_vs_yolo_metrics.csv`
  - `outputs/reports/tracking_vs_yolo_summary.csv`
- Ringkasan latency inferensi:
  - `outputs/reports/inference_latency_summary.csv`

## 14) Troubleshooting Singkat

- Jika import gagal, pastikan Anda menjalankan command dari root proyek:
  - `/home/kopet/skripsi`
- Jika model/video tidak ditemukan, cek ulang path di command atau di config JSON.
- Jika metrik MOT gagal karena frame mismatch, jalankan tanpa `--strict-frame-range` atau sesuaikan file prediksi agar range frame konsisten.
