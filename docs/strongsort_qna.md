# StrongSORT — Kumpulan Tanya-Jawab Pemahaman Kode (Persiapan Sidang)

Dokumen ini merangkum seluruh pertanyaan cek-pemahaman yang dibahas selama
walkthrough kode `strongsort_custom`, lengkap dengan jawaban yang diberikan
dan penjelasan detail/rinci sebagai bahan belajar untuk sidang. Setiap poin
disertai referensi `file:line` supaya bisa langsung dicek ulang di kode.

Urutan mengikuti alur eksekusi per-frame: filtering deteksi → CMC → ReID →
cost matrix/gating → matching cascade → parameter/inisialisasi → verifikasi
dead code → lifecycle track.

---

## 1. Efek menaikkan `min_conf` terhadap ID switch

**Pertanyaan:** Kalau `min_conf` (default `0.1`) dinaikkan mendekati atau
menyamai `conf_threshold` YOLO (`0.25`) — jadi filter deteksi StrongSORT
lebih ketat — apa efeknya terhadap jumlah *ID switch*?

**Jawaban awal:** ID switch akan **menurun**.

**Evaluasi:** ❌ Terbalik — jawaban yang benar adalah ID switch (dan
fragmentasi) justru **meningkat**.

**Penjelasan detail:**
`min_conf` adalah filter pertama yang dijalankan di `StrongSort.update()`:

```python
remain_inds = self.detection_layout.confidences(dets) >= self.min_conf
dets = dets[remain_inds]
```
(`strongsort.py:95-96`)

Rantai akibatnya kalau `min_conf` dinaikkan:
1. Lebih banyak deteksi dengan confidence rendah (biasanya muncul saat objek
   **sebagian terhalang/occluded**, motion blur, atau di tepi frame)
   dibuang sebelum sempat masuk ke tracker.
2. Track yang objeknya sedang mengalami occlusion parsial **tidak
   mendapat deteksi yang lolos filter** di frame itu → `_match()` tidak
   punya kandidat buat track tersebut → track jatuh ke
   `unmatched_tracks` → `mark_missed()` dipanggil (`tracker.py:92-93`).
3. `time_since_update` track itu terus bertambah tiap frame occlusion
   berlanjut. Kalau track masih `Tentative`, **sekali miss saja sudah
   `Deleted`** (`track.py:193-194`). Kalau `Confirmed`, dia baru mati
   setelah `time_since_update > max_age` (`track.py:195-196`).
4. Begitu objek muncul lagi (occlusion selesai) dan deteksinya lolos filter
   lagi, track lama sudah mati (atau nyaris mati lalu keburu dianggap
   unrelated) → `_initiate_track()` membuat track **ID baru**
   (`tracker.py:161-171`) untuk objek yang sebenarnya sama.
5. Hasil akhirnya: **ID switch dan fragmentation naik**, bukan turun.

Ini menjelaskan kenapa StrongSORT sengaja set `min_conf=0.1` — jauh lebih
longgar dari `conf_threshold=0.25` milik YOLO sendiri. Filter longgar ini
"mengorbankan" beberapa deteksi berkualitas rendah supaya masuk ke
matching cascade (dengan risiko dihitung, karena `min_cost_matching` tetap
punya `max_distance` sebagai gerbang penolakan kedua), demi menjaga
kontinuitas track selama occlusion singkat. Konsekuensinya: lebih banyak
komputasi ReID per frame (lebih banyak crop yang diproses model), tapi
ID switch/fragmentasi lebih rendah — trade-off akurasi identitas vs.
kecepatan (FPS), bukan trade-off akurasi deteksi.

---

## 2. CMC pada video statis/tripod

**Pertanyaan:** Kalau video direkam dari kamera statis/tripod (tidak ada
gerakan kamera sama sekali), apa yang diharapkan terjadi pada output CMC
(`warp_matrix`) dan posisi track?

**Jawaban:** "Jika pada video statis/tripod seharusnya tidak banyak berubah
karena minim sekali kompensasi gerakan."

**Evaluasi:** ✅ Benar.

**Penjelasan detail:**
CMC (`strongsort_custom/motion/cmc/ecc.py`) memakai `cv2.findTransformECC`
untuk mengestimasi transformasi affine 2×3 antara frame grayscale
konsekutif. Kalau kamera benar-benar diam, secara ideal transformasi yang
diestimasi adalah **matriks identitas** (tidak ada rotasi/translasi/scale),
sehingga `track.camera_update(warp_matrix)` (`track.py:140-149`) hampir
tidak mengubah `self.mean[:4]` track — cukup no-op secara efektif.

Nuansa tambahan yang relevan untuk sidang: dalam praktik, ECC tidak akan
menghasilkan identitas **sempurna** karena noise sensor kamera, kompresi
video, dan sedikit getaran (bahkan tripod pun tidak 100% rigid). Jadi
`warp_matrix` akan **mendekati** identitas dengan deviasi kecil, bukan
identitas eksak. Ini juga kenapa fallback identitas dipakai secara eksplisit
saat ECC **gagal konvergen** (`cv2.error: StsNoConv`) atau di frame
pertama (belum ada frame sebelumnya untuk dibandingkan) — kasus tersebut
berbeda dari "kamera diam" (yang tetap menjalankan estimasi, hasilnya
kebetulan dekat identitas), vs. fallback (yang secara eksplisit tidak
menjalankan estimasi sama sekali).

---

## 3. Biaya komputasi ReID saat `min_conf` dinaikkan

**Pertanyaan:** Kalau `min_conf` dinaikkan, apa efeknya terhadap beban
komputasi ekstraksi fitur ReID (`self.model.get_features`) per frame?

**Jawaban:** "Iya, dikarenakan jumlah object detection yang diterima
semakin sedikit karena tingginya threshold untuk syarat deteksi."

**Evaluasi:** ✅ Benar.

**Penjelasan detail:**
`get_features(xyxy, img)` (`base_backend.py:190-199`) menjalankan crop →
resize → normalize → **inference batch** untuk **setiap deteksi** yang
lolos filter `min_conf` (`strongsort.py:109-112`, dipanggil setelah baris
95-96). Jumlah baris di `xyxy` = jumlah deteksi yang lolos filter. Semakin
tinggi `min_conf`, semakin sedikit deteksi yang lolos, semakin kecil batch
yang diproses model ReID → **FPS naik** (komputasi lebih ringan), tapi ini
berbanding terbalik dengan poin #1: makin sedikit deteksi occlusion yang
"diselamatkan", makin tinggi risiko ID switch. Ini konkretnya trade-off
`min_conf`: **kecepatan (FPS) vs. kontinuitas identitas (ID switch/
fragmentation)** — dua metrik yang saling tarik-menarik lewat satu
parameter yang sama.

---

## 4. Bobot `mc_lambda` dalam cost matrix

**Pertanyaan:** Kalau `mc_lambda` dibuat sangat mendekati `1` (bobot
appearance cost hampir 100%, gating distance hampir diabaikan) vs. sangat
mendekati `0` (kebalikannya) — dalam skenario dua objek yang saling
berdekatan secara posisi tapi tampilannya (appearance) berbeda jelas, mode
mana yang lebih rentan salah assign?

**Jawaban:** "Dua objek berdekatan tapi tampilannya beda karena bobot lebih
condong ke prediksi posisi" (yaitu `mc_lambda` rendah → gating distance
dominan → rentan salah assign meski appearance-nya jelas beda).

**Evaluasi:** ✅ Benar, dengan nuansa tambahan.

**Penjelasan detail:**
Rumus di `gate_cost_matrix()` (`linear_assignment.py:195-197`):
```python
cost_matrix[row] = mc_lambda * cost_matrix[row] + (1 - mc_lambda) * gating_distance
```
- `mc_lambda` **tinggi** (mendekati 1) → cost didominasi **appearance
  (cosine distance ke gallery EMA)**. Baik untuk kasus objek berdekatan
  posisi tapi beda identitas (mis. dua orang berpapasan) — appearance
  yang membedakan mereka.
- `mc_lambda` **rendah** (mendekati 0) → cost didominasi **gating distance
  (Mahalanobis, berbasis posisi/Kalman)**. Rentan salah assign saat dua
  objek posisinya berdekatan meski appearance-nya jelas beda — karena
  appearance nyaris tidak dipertimbangkan.

Nuansa tambahan (poin yang saya tambahkan saat itu): meskipun `mc_lambda`
tinggi umumnya lebih aman untuk kasus "posisi dekat tapi tampilan beda",
tetap ada risiko sebaliknya di kasus **gating ambiguity** — kalau dua objek
posisinya sangat berdekatan **dan** salah satunya sedang mengalami
degradasi appearance (motion blur, pencahayaan berubah, sebagian
terhalang), `gate_cost_matrix()` tetap **membuang** (set ke `gated_cost =
INFTY_COST`, baris 194) kandidat yang gating distance-nya melebihi
`chi2inv95[4]` **sebelum** blending `mc_lambda` diterapkan. Jadi gating
tetap berfungsi sebagai **hard filter** duluan, baru sisa kandidat yang
lolos gating diberi bobot appearance vs. posisi sesuai `mc_lambda`. Default
StrongSORT `mc_lambda=0.98` (dari `strongsort.py:59`, bukan `0.995` di
`Tracker` — lihat §6) sengaja condong berat ke appearance, konsisten dengan
filosofi StrongSORT yang menonjolkan ReID dibanding motion-only seperti
SORT/ByteTrack.

---

## 5. Prioritas dalam matching cascade: `time_since_update=1` vs. `=5`

**Pertanyaan:** Karena `matching_cascade()` di kode ini ternyata **flat**
(satu Hungarian assignment untuk semua confirmed tracks, `cascade_depth`
tidak dipakai — lihat §5 di `docs/strongsort_flowchart.md`), track
`time_since_update=1` dan `time_since_update=5` diperlakukan setara. Kalau
cascade **asli** (bertingkat) diterapkan, track mana yang akan lebih
diuntungkan, dan kenapa itu relevan dengan risiko ID switch saat dua track
"berebut" satu deteksi yang sama?

**Jawaban:** "Track yang `time_since_update=1` karena untuk
`time_since_update=5` sudah tercampur dengan fitur embedding frame
lama-nya sehingga bisa dianggap banyak noise, lalu jika misal salah track
maka ada kemungkinan ID drift yang berakibat ID Switch juga."

**Evaluasi:** ✅ Benar (track `time_since_update=1` yang diuntungkan),
dengan mekanisme tambahan yang lebih "tekstual" sesuai paper.

**Penjelasan detail:**
Alasan yang diberikan (fitur appearance kian "basi"/noisy pada track yang
lama tidak ter-update) valid — karena gallery EMA (`Track.features`,
`track.py:179-184`) hanya diperbarui saat `update()` dipanggil, jadi track
yang sudah lama miss membawa representasi appearance dari kondisi objek
**yang sudah berubah** (pencahayaan, pose, dll berbeda dari saat ini).

Mekanisme **tekstual/teoretis** (dari paper DeepSORT/StrongSORT, yang jadi
alasan utama desain cascade) berbasis **pertumbuhan covariance Kalman**,
bukan cuma staleness fitur:
- Setiap kali `predict()` dipanggil tanpa `update()` mengoreksinya,
  `motion_cov` di Kalman filter terus **membesar**
  (`kalman_filters/base.py`, langkah `predict()`) — makin lama track
  tidak ter-*update*, makin besar ketidakpastian posisinya.
- Covariance besar → *gating region* (`gating_distance` di
  `gate_cost_matrix`, dibandingkan `chi2inv95[4]`) jadi **lebih lebar/
  permisif** — track lama bisa "menjangkau" deteksi yang secara geometris
  agak jauh dan tetap lolos gating.
- Dalam skema **flat** (yang dipakai kode ini), track lama dengan gating
  lebar ini ikut bersaing setara dengan track segar dalam satu Hungarian
  assignment global. Kalau totalnya kebetulan menghasilkan cost lebih
  rendah, track lama bisa **menang** merebut deteksi yang sebenarnya lebih
  tepat untuk track segar — walau secara identitas itu salah.
- Cascade **asli** (bertingkat, berbasis `cascade_depth`/usia) mencegah ini
  dengan memberi **hak pilih pertama** ke track paling segar
  (`time_since_update` terkecil), baru sisa deteksi yang belum terpakai
  diperebutkan oleh track yang lebih tua. Ini melindungi track segar dari
  "direbut" oleh track tua yang gating-nya kebetulan lebih longgar.

Kedua efek ini (fitur appearance basi + covariance melebar) **saling
melengkapi**, bukan saling meniadakan — keduanya sama-sama menjelaskan
kenapa cascade bertingkat lebih aman daripada flat matching, dan keduanya
bermuara ke risiko yang sama: ID switch/drift saat track lama salah
merebut deteksi milik track segar.

---

## 6. Parameter mana yang benar-benar berlaku: `StrongSort.mc_lambda` vs. `Tracker.mc_lambda`

**Pertanyaan:** `StrongSort.__init__` punya default `mc_lambda=0.98`
sedangkan `Tracker.__init__` punya default `mc_lambda=0.995`. Nilai mana
yang sebenarnya dipakai saat tracker berjalan?

**Jawaban:** "StrongSORT karena jika ada nilai `mc_lambda` yang
dideklarasikan, maka nilai default akan ter-override dengan nilai yang
dideklarasikan."

**Evaluasi:** ✅ Benar untuk kesimpulan, tapi mekanismenya perlu
diperjelas.

**Penjelasan detail:**
`StrongSort.__init__` **selalu** memanggil `Tracker(...)` dengan
`mc_lambda=mc_lambda` secara eksplisit (`strongsort.py:76-83`):
```python
self.tracker = Tracker(
    ...,
    mc_lambda=mc_lambda,
    ema_alpha=ema_alpha,
)
```
Ini berlaku **bahkan kalau user StrongSORT tidak meng-override apa pun** —
karena parameter Python selalu mengutamakan **argumen eksplisit** di atas
**default parameter** si callee. Jadi:
- Kalau user set `mc_lambda=X` saat instansiasi `StrongSort(...)` → `X`
  yang dipakai (override eksplisit, sesuai jawaban).
- Kalau user **tidak** set apa-apa → yang dipakai tetap `0.98` (default
  `StrongSort`), **bukan** `0.995` (default `Tracker`) — karena
  `StrongSort` tetap mem-forward nilai `0.98` miliknya sendiri secara
  eksplisit ke `Tracker()`.

Konsekuensinya: `mc_lambda=0.995` di `tracker.py:50` adalah **dead
default** — tidak akan pernah tercapai selama satu-satunya jalur
instansiasi `Tracker` di codebase ini adalah lewat `StrongSort`. Pelajaran
umum untuk sidang: kalau ditanya "nilai default parameter X itu berapa",
selalu telusuri ke **titik instansiasi terluar yang benar-benar dipanggil**
(via config eksperimen/YAML/JSON), bukan cuma baca `__init__` paling
dalam — karena default di layer dalam bisa jadi tidak pernah relevan.

---

## 7. Membuktikan `Tracker.cmc` adalah dead code

**Pertanyaan:** Selain membaca kode dan tidak menemukan pemanggilan
`Tracker.cmc`, bagaimana cara membuktikan klaim ini secara lebih
meyakinkan/empiris?

**Jawaban:** "Kita berikan conditional logging, jika memang terpanggil
maka akan menambahkan log bahwa `Tracker.cmc` memang dipanggil dan jika
tidak maka akan menampilkan log bahwa hanya `self.cmc` yang dipakai aktif
di `update()`."

**Evaluasi:** ✅ Valid, ditambah satu metode pelengkap yang lebih
sederhana.

**Penjelasan detail:**
Logging kondisional adalah cara **positif** untuk membuktikan pemanggilan
(menunjukkan berapa kali suatu baris tereksekusi). Cara pelengkap yang
lebih sederhana untuk bukti **negatif** (tidak pernah dipakai): hapus/
comment baris `self.cmc = get_cmc_method("ecc")()` di `Tracker.__init__`
(`tracker.py:62`), lalu jalankan **full pipeline** video sampai selesai.
Kalau tidak muncul `AttributeError: 'Tracker' object has no attribute
'cmc'` di titik manapun, itu bukti langsung bahwa atribut tersebut memang
tidak pernah diakses di seluruh execution path yang benar-benar dilewati.

Kombinasi tiga lapis verifikasi paling meyakinkan untuk sidang:
1. **Statis** — `grep -rn "\.cmc" strongsort_custom/trackers/strongsort/sort/` →
   tidak ada pemanggilan `self.cmc.` di dalam `tracker.py` (hanya
   assignment di `__init__`).
2. **Dinamis-negatif** — hapus baris assignment, jalankan pipeline, tidak
   ada `AttributeError`.
3. **Dinamis-positif** (opsional, kalau ingin lebih yakin) — logging
   kondisional/`print` di titik yang dicurigai untuk konfirmasi eksplisit
   nol pemanggilan.

Catatan: ini berbeda dengan `StrongSort.cmc` (`strongsort.py:86`) yang
**aktif dipakai** di `update()` baris 104 — dua atribut bernama sama
(`cmc`) di dua kelas berbeda, satu hidup satu mati. Kesalahan umum saat
membaca cepat adalah menyamakan keduanya karena namanya identik.

---

## 8. Apakah track baru "kehilangan" kompensasi CMC kalau CMC sempat di-skip?

**Pertanyaan:** CMC di-skip total kalau `len(self.tracker.tracks) == 0`
(`strongsort.py:103`). Kalau di frame berikutnya deteksi pertama masuk dan
track baru dibuat, apakah track baru itu "kehilangan" kompensasi kamera
untuk pergerakan yang terjadi sebelum dia lahir? Apakah ini masalah?

**Jawaban:** "Seharusnya track baru tidak 'kehilangan' kompensasi karena
CMC di-*process* untuk setiap frame, jika frame selanjutnya terdapat
deteksi dan track dibuat, maka akan dihitung kompensasinya."

**Evaluasi:** ✅ Kesimpulan benar (tidak masalah), tapi alasan perlu
dikoreksi/diperdalam.

**Penjelasan detail:**
Alasan yang diberikan ("CMC diproses tiap frame jadi nanti kehitung juga")
kurang tepat sebagai penjelasan **inti**, karena sebenarnya track baru
**tidak pernah butuh kompensasi retroaktif sama sekali** — bukan soal
"nanti kehitung di frame berikutnya". Penjelasan yang lebih presisi:

`_initiate_track()` (`tracker.py:161-171`) membuat `Track` baru langsung
dari `detection.to_xyah()` — koordinat mentah deteksi **frame saat ini**
(lihat `Track.__init__`, `track.py:82`: `self.bbox = detection.to_xyah()`).
Track baru **tidak punya riwayat state dari frame sebelumnya** yang perlu
"digeser" mengikuti pergerakan kamera — dia lahir sudah berada di sistem
koordinat frame sekarang, apa adanya. `camera_update()` (`track.py:140-149`)
hanya relevan untuk **menggeser `mean` yang sudah ada** (hasil prediksi
dari state frame sebelumnya) supaya tetap valid di frame sekarang.

Jadi CMC yang di-skip di frame sebelumnya (karena belum ada track sama
sekali) memang benar-benar **tidak relevan** — bukan karena "nanti akan
dihitung juga di frame berikutnya", tapi karena track yang lahir di
frame berikutnya tidak pernah memiliki "utang" kompensasi untuk dibayar
mundur. CMC hanya bekerja pada arah "maju" (state lama → disesuaikan untuk
frame baru), tidak pernah perlu bekerja mundur pada track yang belum
eksis.

---

## 9. Apakah ID tetap sama setelah track miss lalu ter-*match* lagi?

**Pertanyaan:** Track *confirmed* yang miss 1 frame (`time_since_update`
jadi `1`, tidak masuk output), lalu di frame berikutnya kembali
ter-*match* (`time_since_update` balik ke `0`) — apakah track ini akan
punya ID yang sama saat muncul lagi di output, atau dianggap track baru?

**Jawaban:** "ID-nya sama, karena seluruh hasil deteksi/track disimpan
dalam list terlebih dahulu asalkan `time_since_update` tidak melebihi
`max_age`."

**Evaluasi:** ✅ Benar, dengan mekanisme presisi yang perlu ditambahkan.

**Penjelasan detail:**
Alasan kuncinya bukan sekadar "disimpan dalam list", tapi **identitas
objek Python yang tidak pernah diganti**:

`self.tracks` di `Tracker` **tidak pernah dibuat ulang**, hanya
**dimutasi** dan **difilter**:
```python
for track_idx, detection_idx in matches:
    self.tracks[track_idx].update(detections[detection_idx])   # tracker.py:90-91
for track_idx in unmatched_tracks:
    self.tracks[track_idx].mark_missed()                        # tracker.py:92-93
...
self.tracks = [t for t in self.tracks if not t.is_deleted()]    # tracker.py:96 — filter, bukan rebuild
```
Track yang miss di frame N (`mark_missed()` dipanggil pada objek `Track`
yang sama) lalu match lagi di frame N+1 adalah **objek Python yang
identik** — `.update()` maupun `.mark_missed()` **tidak pernah menyentuh
`self.id`** (`track.py:163-196`). Jadi ID pasti sama, selama objeknya
tidak sempat difilter keluar oleh `is_deleted()` di antara kedua frame
tersebut.

Syarat presisi supaya track **tidak** ikut terhapus saat miss
(`mark_missed()`, `track.py:191-196`):
```python
def mark_missed(self):
    if self.state == TrackState.Tentative:
        self.state = TrackState.Deleted
    elif self.time_since_update > self._max_age:
        self.state = TrackState.Deleted
```
- Track **Tentative**: **nol toleransi** — sekali miss saja sebelum lolos
  `n_init`, langsung `Deleted`.
- Track **Confirmed**: boleh miss berkali-kali, mati hanya kalau
  `time_since_update > max_age` (**strict greater than**, bukan `>=`).
  Contoh: `max_age=30` → track masih hidup sampai miss ke-30, baru mati di
  miss ke-31.

Jadi jawaban "asalkan tidak melebihi `max_age`" sudah menangkap kondisi
untuk track **Confirmed**, tapi perlu ditambah: track **Tentative** punya
aturan yang jauh lebih ketat (nol toleransi miss), tidak tunduk pada
`max_age` sama sekali.

---

## Ringkasan temuan "dead code" / hal tersembunyi (sering ditanya di sidang)

| Temuan | Lokasi | Status |
|---|---|---|
| `Tracker.GATING_THRESHOLD = np.sqrt(chi2inv95[4])` | `tracker.py:40` | Didefinisikan, **tidak pernah dipakai** — `gate_cost_matrix()` pakai variabel lokalnya sendiri `chi2inv95[4]` (`linear_assignment.py:187`) |
| `cascade_depth` di `matching_cascade()` | `linear_assignment.py:82-90` | Parameter diterima, **tidak pernah dipakai** untuk membagi track per level usia — cuma satu `min_cost_matching` flat |
| `Tracker.cmc` | `tracker.py:62` | Dibuat di `__init__`, **tidak pernah dipanggil** di manapun dalam `tracker.py` (beda dengan `StrongSort.cmc` yang aktif) |
| `Tracker.mc_lambda=0.995` (default) | `tracker.py:50` | **Dead default** — selalu di-override eksplisit oleh `StrongSort.__init__` (`mc_lambda=0.98`) |
| CI bypass `TrackState.Confirmed` langsung | `track.py:91-99` | Aktif hanya kalau env var `GITHUB_ACTIONS=true` — tidak relevan di lingkungan eksperimen normal |
