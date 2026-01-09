import pandas as pd

# ===== CONFIG =====
CSV_PATH = "C:/Users/KHADHI MUSAID SYAH/Downloads/export_219349_project-219349-at-2025-12-30-07-21-9a199096.csv"
OUTPUT_GT = "gt.txt"

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# ==================

df = pd.read_csv(CSV_PATH)

# Ganti nama kolom jika berbeda
# Pastikan kolom ini ada di CSV kamu
df = df.rename(columns={
    "frame": "frame",
    "x": "x",
    "y": "y",
    "width": "w",
    "height": "h",
    "object_id": "track_id"  # atau annotation_id
})

# Jika track_id bukan numerik → mapping ke integer
df["track_id"] = df["track_id"].astype("category").cat.codes + 1

# Convert persen → pixel
df["x"] = (df["x"] / 100 * VIDEO_WIDTH).astype(int)
df["y"] = (df["y"] / 100 * VIDEO_HEIGHT).astype(int)
df["w"] = (df["w"] / 100 * VIDEO_WIDTH).astype(int)
df["h"] = (df["h"] / 100 * VIDEO_HEIGHT).astype(int)

# Tambah kolom MOT
df["conf"] = 1
df["class"] = 1
df["visibility"] = 1

# Urutan kolom MOT
df = df[["frame", "track_id", "x", "y", "w", "h", "conf", "class", "visibility"]]

# Sort wajib
df = df.sort_values(["frame", "track_id"])

# Simpan TANPA header
df.to_csv(OUTPUT_GT, index=False, header=False)

print("✅ gt.txt berhasil dibuat")
