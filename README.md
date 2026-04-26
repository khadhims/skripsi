# Skripsi Tracking Pipeline (YOLO + StrongSORT)

Dokumen ini menjelaskan cara menggunakan file utama setelah restrukturisasi proyek.

## 1) Ringkasan Folder Utama

- `main.py`
  - Runner utama tracking realtime/video (YOLO + StrongSORT) untuk generate prediksi MOT.
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

## 2) Setup Environment

Gunakan virtual environment proyek:

```bash
cd /home/kopet/skripsi
source venv/bin/activate
pip install -r requirements.txt
```

## 3) Menjalankan Tracking Utama

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

## 4) Analisis Perbandingan Utama

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

## 5) Evaluasi MOT Satu Pasangan File

```bash
python evaluation/mot/run_mot_validation.py \
  --gt dataset/tracker-val/vid3-gt.txt \
  --pred dataset/tracker-results/iterasi-1/v8-medium-it2-vid3-pred.txt \
  --iou 0.7
```

## 6) Benchmark Pipeline

```bash
python benchmarks/run_pipeline_benchmark.py \
  --video dataset/videos/vid_3.mp4 \
  --max-frames 200
```

## 7) Detection-Only Utilities

### 7.1 Quick Detection Test (display + latency log di terminal)

```bash
python pipelines/detection/test_detection.py \
  --model models/yolov8/iterasi-2/v8-medium.pt \
  --video dataset/videos/vid_3.mp4
```

### 7.2 Export Video Detection-Only (tanpa tracking)

```bash
python pipelines/detection/yolo_detect_only.py \
  --model models/yolov8/iterasi-2/v8-medium.pt \
  --source dataset/videos/vid_3.mp4 \
  --output video-results/detection-only/vid3_detect.mp4
```

## 8) Data Preparation Tools

### 8.1 Extract Frame dari Video

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

### 8.2 Convert CSV Annotation ke MOT GT

```bash
python tools/data_prep/mot_val_convert.py \
  --csv path/to/annotation.csv \
  --output dataset/tracker-val/vidX-gt.txt \
  --video-width 1920 \
  --video-height 1080 \
  --track-id-col object_id
```

## 9) Analisis File Log Inferensi

Jika Anda punya file log inferensi sendiri, analisis dengan:

```bash
python analysis/inference_logs/analyze_inference.py \
  --log path/to/inference_log.txt \
  --output outputs/reports/inference_latency_summary.csv
```

Catatan: pipeline saat ini tidak lagi otomatis membuat `inference_log.txt` ke file.

## 10) Konfigurasi Eksperimen

Konfigurasi default perbandingan ada di:

- `configs/experiments/v5_vs_v8_strongsort.json`

Anda bisa mengubah:

- `settings` (`iou_threshold`, `conf_threshold`, `imgsz`)
- daftar `experiments` (path GT, prediksi tracking, video, model)

## 11) Lokasi Output Penting

- Detail + summary perbandingan model:
  - `outputs/reports/v5_vs_v8_strongsort_metrics.csv`
  - `outputs/reports/v5_vs_v8_strongsort_summary.csv`
- Detail + summary tracking vs YOLO-only:
  - `outputs/reports/tracking_vs_yolo_metrics.csv`
  - `outputs/reports/tracking_vs_yolo_summary.csv`
- Ringkasan latency inferensi:
  - `outputs/reports/inference_latency_summary.csv`

## 12) Troubleshooting Singkat

- Jika import gagal, pastikan Anda menjalankan command dari root proyek:
  - `/home/kopet/skripsi`
- Jika model/video tidak ditemukan, cek ulang path di command atau di config JSON.
- Jika metrik MOT gagal karena frame mismatch, jalankan tanpa `--strict-frame-range` atau sesuaikan file prediksi agar range frame konsisten.
