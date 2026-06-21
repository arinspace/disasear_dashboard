"""
Dataset Verification and Debug Script
======================================
Validates dataset structure and checks if image-mask pairs exist correctly.
Lists first 5 images and verifies corresponding mask files are present.

Directories:
    - Images: D:/geoai_train/geoai_train/datasets/train
    - Masks: D:/geoai_train/geoai_train/datasets/train_labels
"""

import os

img_dir = "D:/geoai_train/geoai_train/datasets/train"
mask_dir = "D:/geoai_train/geoai_train/datasets/train_labels"

images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
print("จำนวนภาพ:", len(images))

for i in images[:5]:
    mask_name = os.path.splitext(i)[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_name)
    print(i, "->", mask_path, os.path.exists(mask_path))
