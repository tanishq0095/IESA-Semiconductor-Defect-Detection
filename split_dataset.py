import os
import shutil
import random

# Configuration
base_path = r'C:\Users\mange\OneDrive\Desktop\ic_hackathon\dataset'
classes = ['bridge', 'clean', 'crack', 'malformed_via', 'open', 'other', 'scratch', 'short']
split_ratio = {'Train': 0.75, 'Validation': 0.15, 'Test': 0.10}

# 1. Create the new folder structure
for split in split_ratio.keys():
    for cls in classes:
        os.makedirs(os.path.join(base_path, split, cls), exist_ok=True)

# 2. Move images and force delete original folders
for cls in classes:
    src_folder = os.path.join(base_path, cls)
    
    # Skip if it's not a folder or already one of the split folders
    if not os.path.isdir(src_folder) or cls in split_ratio.keys():
        continue
        
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(files)
    
    train_idx = int(len(files) * split_ratio['Train'])
    val_idx = train_idx + int(len(files) * split_ratio['Validation'])
    
    for i, f in enumerate(files):
        src = os.path.join(src_folder, f)
        if i < train_idx:
            dest = os.path.join(base_path, 'Train', cls, f)
        elif i < val_idx:
            dest = os.path.join(base_path, 'Validation', cls, f)
        else:
            dest = os.path.join(base_path, 'Test', cls, f)
        shutil.move(src, dest)
    
    # FORCE DELETE the original folder even if it has hidden files
    shutil.rmtree(src_folder)

print("SUCCESS: Your dataset is now organized into Train, Validation, and Test folders!")
    