import cv2
import numpy as np
import os
import csv

# Paths
MASK_DIR = "../raw_images/Anomaly_test/Anomaly_test/abnormal_mask"
OUTPUT_CSV = "../defect_features.csv"

# Prepare CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "area",
        "perimeter",
        "aspect_ratio",
        "solidity",
        "touches_boundary"
    ])

    for file in os.listdir(MASK_DIR):
        if not file.endswith(".jpg"):
            continue

        path = os.path.join(MASK_DIR, file)

        # Read mask (grayscale)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Binary mask
        _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        cnt = max(contours, key=cv2.contourArea)

        # Area
        area = cv2.contourArea(cnt)

        # Perimeter
        perimeter = cv2.arcLength(cnt, True)

        # Bounding box → aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        # Solidity = area / convex hull area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        # Boundary contact
        h_img, w_img = mask.shape
        touches_boundary = (
            x == 0 or
            y == 0 or
            x + w >= w_img - 1 or
            y + h >= h_img - 1
        )

        # Write to CSV
        writer.writerow([
            file,
            int(area),
            float(perimeter),
            round(aspect_ratio, 3),
            round(solidity, 3),
            int(touches_boundary)
        ])

print("Feature extraction complete.")
print("Saved to:", OUTPUT_CSV)
