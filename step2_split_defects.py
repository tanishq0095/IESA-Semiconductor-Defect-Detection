import os
import cv2
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_DIR = os.path.join(
    BASE_DIR, "raw_images", "Anomaly_test", "Anomaly_test", "abnormal_img"
)

MASK_DIR = os.path.join(
    BASE_DIR, "raw_images", "Anomaly_test", "Anomaly_test", "abnormal_mask"
)

OUT_DIR = os.path.join(BASE_DIR, "dataset")

def classify(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "other"

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / (h + 1e-5)

    if area < 150:
        return "via"
    if aspect > 5:
        return "scratch"
    if aspect > 2:
        return "crack"
    if area > 3000:
        return "cmp"
    if area > 800:
        return "bridge"
    return "short"

count = 0

for img in os.listdir(IMG_DIR):
    if not img.endswith(".jpg"):
        continue

    mask_name = img.replace(".jpg", "_mask.jpg")
    mask_path = os.path.join(MASK_DIR, mask_name)

    if not os.path.exists(mask_path):
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cls = classify(mask)

    shutil.copy(
        os.path.join(IMG_DIR, img),
        os.path.join(OUT_DIR, cls, img)
    )

    count += 1

print("Processed", count, "defect images")
