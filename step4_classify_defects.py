import os
import csv
import shutil

# Base directory (ic_hackathon)
BASE_DIR = os.path.abspath("..")

# CSV with extracted features
FEATURES_CSV = os.path.join(BASE_DIR, "defect_features.csv")

# Source abnormal images
SRC_IMG_DIR = os.path.join(
    BASE_DIR, "raw_images", "Anomaly_test", "Anomaly_test", "abnormal_img"
)

# Destination dataset folder
DST_BASE = os.path.join(BASE_DIR, "dataset")

# Defect classes
classes = ["crack", "bridge", "open", "short", "scratch", "other"]
for c in classes:
    os.makedirs(os.path.join(DST_BASE, c), exist_ok=True)


def classify(row):
    area = float(row["area"])
    aspect = float(row["aspect_ratio"])
    solidity = float(row["solidity"])
    touches = int(row["touches_boundary"])

    if aspect > 4:
        return "crack"
    if area > 2500 and solidity > 0.8:
        return "bridge"
    if touches == 1:
        return "open"
    if area < 400:
        return "short"
    if solidity < 0.6:
        return "scratch"
    return "other"


count = 0

with open(FEATURES_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cls = classify(row)

        # Convert mask filename → image filename
        mask_name = row["filename"]
        img_name = mask_name.replace("_mask", "")

        src = os.path.join(SRC_IMG_DIR, img_name)
        dst = os.path.join(DST_BASE, cls, img_name)

        if not os.path.exists(src):
            print("Missing:", img_name)
            continue

        shutil.copy(src, dst)
        count += 1

print(f"Classified {count} defect images into 6 classes")

