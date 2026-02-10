import os
import shutil

# Step 1: Find base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Step 2: Correct path to clean images (nested folder!)
RAW_DIR = os.path.join(
    BASE_DIR,
    "raw_images",
    "Anomaly_test",
    "Anomaly_test",
    "normal_img"
)

# Step 3: Output folder
OUT_DIR = os.path.join(BASE_DIR, "dataset", "clean")

# Step 4: Create output folder if missing
os.makedirs(OUT_DIR, exist_ok=True)

# Step 5: Copy images
count = 0
for file in os.listdir(RAW_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        shutil.copy(
            os.path.join(RAW_DIR, file),
            os.path.join(OUT_DIR, file)
        )
        count += 1

print("Copied", count, "clean images")
