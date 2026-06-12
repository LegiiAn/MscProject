import os
import cv2
import numpy as np

dataset_path = "data/cleargrasp_dataset/cleargrasp-dataset-train/square-plastic-bottle-train"
rgb_dir = os.path.join(dataset_path, "rgb-imgs")
mask_dir = os.path.join(dataset_path, "segmentation-masks")
output_dir = "data/raw_objects/square_bottle"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(rgb_dir):
    if not filename.endswith('.png'): continue
    
    # 1. Load RGB image and its matching mask
    rgb = cv2.imread(os.path.join(rgb_dir, filename))
    mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
    
    # 2. Convert RGB to BGRA (adds an Alpha transparency channel)
    bgra = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
    
    # 3. Set alpha channel to 0 everywhere outside the mask
    bgra[:, :, 3] = mask 
    
    # 4. Save clean transparent object asset
    cv2.imwrite(os.path.join(output_dir, f"trans_{filename}"), bgra)

print("Extraction complete. Foreground assets ready for compositing.")