# Panduan Lengkap Metrik Evaluasi: YOLOv5-StrongSORT vs YOLOv8-StrongSORT

Dokumen ini menjelaskan **seluruh** kolom yang terdapat dalam file laporan `baseline_metrics.csv` dan `tuned_metrics.csv`. Penjelasan ini disusun untuk mempermudah analisis komparatif dalam skripsi.

---

## 1. Informasi Metadata & Konfigurasi

Kolom-kolom ini mendefinisikan konteks dari eksperimen yang dijalankan.

| Header Kolom                            | Penjelasan                                                                                                                                            |
| :-------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **experiment**                          | Nama unik skenario uji (contoh: `v5-sepi`, `v8-ramai`).                                                                                               |
| **model_family**                        | Arsitektur detektor yang digunakan (`yolov5-strongsort` atau `yolov8-strongsort`).                                                                    |
| **class_agnostic**                      | Jika `True`, sistem menganggap semua objek sebagai kategori yang sama (misal: semua kendaraan dihitung sebagai "objek" tanpa membedakan motor/mobil). |
| **gt_path**                             | Lokasi file _Ground Truth_ (data referensi yang dianggap benar).                                                                                      |
| **pred_path**                           | Lokasi file hasil prediksi/tracking sistem.                                                                                                           |
| **gt_min_frame** / **gt_max_frame**     | Rentang frame awal dan akhir yang tersedia pada data _Ground Truth_.                                                                                  |
| **pred_min_frame** / **pred_max_frame** | Rentang frame awal dan akhir di mana sistem berhasil memberikan deteksi.                                                                              |

---

## 2. Metrik Efisiensi (Speed)

Mengukur performa komputasi dari integrasi YOLO dan StrongSORT.

| Header Kolom         | Penjelasan                                                                                                            |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **avg_fps**          | _Frames Per Second_ rata-rata. Menunjukkan berapa frame yang bisa diproses dalam 1 detik.                             |
| **avg_inference_ms** | Waktu inferensi rata-rata dalam milidetik (ms) per frame. Mencakup waktu deteksi YOLO dan proses asosiasi StrongSORT. |

---

## 3. Metrik Statistik Dasar (Basic Metrics)

Metrik ini dihitung secara manual (heuristik) untuk memberikan gambaran kasar sebelum masuk ke perhitungan MOT yang kompleks.

| Header Kolom                       | Penjelasan                                                                                  |
| :--------------------------------- | :------------------------------------------------------------------------------------------ |
| **basic_precision**                | Akurasi deteksi dasar: `TP / (TP + FP)`. Fokus pada ketepatan prediksi.                     |
| **basic_recall**                   | Daya tangkap dasar: `TP / (TP + FN)`. Fokus pada seberapa banyak objek GT yang tertangkap.  |
| **basic_avg_frame_count_error**    | Rata-rata selisih jumlah objek per frame antara prediksi dan GT.                            |
| **basic_total_unique_count_error** | Selisih antara jumlah total ID unik yang ditemukan sistem dengan jumlah ID unik asli di GT. |
| **basic_total_unique_gt**          | Jumlah total individu/objek unik yang ada dalam data asli (GT).                             |
| **basic_total_unique_pred**        | Jumlah total individu/objek unik yang didaftarkan oleh sistem pelacak.                      |
| **basic_total_tp**                 | _True Positives_: Total deteksi yang dianggap benar (berhasil mencocokkan objek).           |
| **basic_total_fp**                 | _False Positives_: Total deteksi palsu (sistem mendeteksi objek di tempat kosong).          |
| **basic_total_fn**                 | _False Negatives_: Total objek yang gagal terdeteksi oleh sistem.                           |

---

## 4. Metrik Standar MOT (Multi-Object Tracking)

Metrik berbasis library `motmetrics` yang mengikuti standar internasional.

| Header Kolom            | Penjelasan                                                                                                                          |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| **mota**                | _Multi-Object Tracking Accuracy_. Akurasi keseluruhan yang menggabungkan kesalahan FP, FN, dan ID Switch.                           |
| **idf1**                | _Identification F1-Score_. Keseimbangan antara ID Precision dan ID Recall. Menunjukkan seberapa stabil sistem menjaga ID yang sama. |
| **idp**                 | _Identification Precision_. Seberapa akurat ID yang diberikan (apakah ID tersebut benar-benar milik objek itu).                     |
| **idr**                 | _Identification Recall_. Seberapa banyak bagian dari lintasan objek yang berhasil diberi ID yang benar.                             |
| **precision**           | Tingkat ketepatan deteksi objek dalam konteks pelacakan.                                                                            |
| **recall**              | Tingkat keberhasilan menangkap objek dalam konteks pelacakan.                                                                       |
| **num_switches**        | Jumlah kejadian di mana sebuah objek berganti ID di tengah jalan (_ID Switch_).                                                     |
| **num_false_positives** | Jumlah deteksi salah yang tercatat dalam perhitungan MOT resmi.                                                                     |
| **num_misses**          | Jumlah objek yang terlewatkan dalam perhitungan MOT resmi.                                                                          |
| **num_fragmentations**  | Berapa kali lintasan pelacakan objek terputus atau terpecah menjadi beberapa bagian.                                                |
| **num_matches**         | Jumlah total pasangan (prediksi, GT) yang berhasil dicocokkan berdasarkan ambang batas jarak (IoU).                                 |
| **num_objects**         | Total jumlah kemunculan objek di seluruh frame dalam Ground Truth (akumulasi).                                                      |
| **num_detections**      | Total jumlah deteksi yang dihasilkan sistem di seluruh frame.                                                                       |

---

## 5. Metrik Identitas Detail (Identity Counters)

Digunakan untuk menghitung nilai IDF1 secara presisi.

| Header Kolom | Penjelasan                                                                                                              |
| :----------- | :---------------------------------------------------------------------------------------------------------------------- |
| **idtp**     | _Identity True Positives_. Jumlah frame di mana objek dilacak dengan ID yang benar sesuai pemetaan terbaik.             |
| **idfp**     | _Identity False Positives_. Jumlah frame di mana sistem memberikan ID yang salah pada posisi tersebut.                  |
| **idfn**     | _Identity False Negatives_. Jumlah frame di mana objek asli tidak mendapatkan ID yang sesuai dalam lintasan yang tepat. |

---

## 6. Perbandingan Metrik yang Mirip (Key Differences)

Sering kali terdapat kebingungan antara metrik "Basic" dengan metrik "MOT/Identity". Berikut adalah perbedaannya:

### **A. basic_precision vs precision**
*   **basic_precision:** Dihitung secara sederhana per frame. Jika sistem mendeteksi kotak di posisi yang benar, ia dianggap benar (TP), tanpa memedulikan apakah ID-nya konsisten atau tidak.
*   **precision:** Dalam konteks MOT, ini dihitung setelah proses asosiasi pelacakan. Ia mencerminkan akurasi deteksi yang sudah difilter oleh algoritma pelacakan (StrongSORT).

### **B. basic_recall vs recall**
*   **basic_recall:** Seberapa banyak objek asli yang berhasil "disentuh" oleh kotak prediksi di setiap frame secara mentah.
*   **recall:** Seberapa banyak objek asli yang berhasil dilacak secara konsisten. Jika sebuah objek terdeteksi tapi tidak bisa diasosiasikan ke sebuah lintasan (track), maka ia mungkin tidak terhitung secara penuh di sini.

### **C. basic_total_fp vs num_false_positives vs idfp**
*   **basic_total_fp:** Jumlah deteksi palsu mentah (sistem mendeteksi sesuatu di tempat kosong).
*   **num_false_positives:** FP yang dihitung setelah penyelarasan lintasan.
*   **idfp (Identity FP):** FP yang dihitung khusus untuk identitas. Ini terjadi jika sistem memberikan ID pada posisi yang tidak ada objeknya, atau jika satu objek diberikan dua ID berbeda secara bersamaan.

### **D. basic_total_fn vs num_misses vs idfn**
*   **basic_total_fn:** Objek asli yang benar-benar tidak terdeteksi sama sekali oleh YOLO.
*   **num_misses:** Objek yang "hilang" dari pantauan sistem pelacak (bisa karena detektor gagal, atau pelacak gagal menyambungkan ID).
*   **idfn (Identity FN):** Objek asli yang tidak berhasil dipasangkan dengan ID yang benar dalam pemetaan lintasan global.

### **E. idp vs precision**
*   **precision:** Fokus pada **lokasi** (apakah kotak pembatasnya benar?).
*   **idp (Identification Precision):** Fokus pada **identitas** (apakah ID yang diberikan benar?). Sebuah deteksi bisa saja memiliki lokasi yang benar (Precision tinggi) tapi ID-nya salah karena tertukar (IDP rendah).

---

## 7. Cara Membaca Hasil untuk Analisis Skripsi


1.  **Jika `num_switches` tinggi:** Artinya StrongSORT sering "kebingungan" dan mengganti nomor ID objek, biasanya terjadi saat objek saling berdempetan atau terhalang (oklusi).
2.  **Jika `basic_total_unique_pred` jauh lebih besar dari `basic_total_unique_gt`:** Artinya sistem terlalu sensitif atau sering membuat ID baru untuk objek yang sama (fragmentasi tinggi).
3.  **Perbandingan YOLOv5 vs YOLOv8:**
    - Lihat **avg_inference_ms** untuk membandingkan kecepatan.
    - Lihat **MOTA** untuk membandingkan akurasi deteksi secara umum.
    - Lihat **IDF1** untuk membandingkan ketangguhan algoritma StrongSORT dalam menjaga identitas pada masing-masing model.
