import motmetrics as mm
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
GT_PATH = "./dataset/tracker-val/vid3-gt.txt"
PRED_PATH = "./dataset/tracker-results/iterasi-1/v8-medium-it2-vid3-pred.txt"
IOU_THRESHOLD = 0.7


# ==============================
# LOAD DATA
# ==============================

def load_mot_file(path):
    """
    Load MOT format:
    frame, id, x, y, w, h, conf, class, visibility
    """
    cols = [
        "frame",
        "id",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "class",
        "visibility",
    ]
    df = pd.read_csv(path, header=None)
    df.columns = cols
    return df


gt_df = load_mot_file(GT_PATH)
pred_df = load_mot_file(PRED_PATH)

# ==============================
# FRAME RANGE CHECK
# ==============================

gt_min = gt_df["frame"].min()
gt_max = gt_df["frame"].max()

pred_min = pred_df["frame"].min()
pred_max = pred_df["frame"].max()

print("\n===== FRAME RANGE CHECK =====")
print(f"GT   : {gt_min} → {gt_max}")
print(f"Pred : {pred_min} → {pred_max}")

if pred_max != gt_max:
    raise ValueError("Frame range tidak sama! Evaluasi dihentikan.")

if pred_max > gt_max:
    print("\n⚠ WARNING: Pred frame melebihi GT frame!")
elif pred_max < gt_max:
    print("\n⚠ WARNING: Pred frame lebih pendek dari GT!")
else:
    print("\n✓ Frame range aman (GT max == Pred max)")


# ==============================
# AGNOSTIC MODE
# ==============================
# Abaikan class → tidak difilter per class
# Semua object dianggap 1 kategori

# Convert ke format motmetrics
gt = mm.io.loadtxt(GT_PATH, fmt="mot15-2D")
pred = mm.io.loadtxt(PRED_PATH, fmt="mot15-2D")

# ==============================
# EVALUATION
# ==============================

acc = mm.utils.compare_to_groundtruth(
    gt,
    pred,
    dist="iou",
    distth=IOU_THRESHOLD
)

mh = mm.metrics.create()

summary = mh.compute(
    acc,
    metrics=mm.metrics.motchallenge_metrics,
    name="Agnostic"
)

# ==============================
# PRINT RESULT
# ==============================

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)

print("\n===== MOT Evaluation (Class-Agnostic) =====\n")
print(strsummary)

# ==============================
# EXTRA METRICS DETAIL
# ==============================

mota = summary.loc["Agnostic"]["mota"]
idf1 = summary.loc["Agnostic"]["idf1"]

print("\n===== Key Metrics =====")
print(f"MOTA  : {mota:.4f}")
print(f"IDF1  : {idf1:.4f}")
