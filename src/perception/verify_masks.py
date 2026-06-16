import os
# Keep OpenEXR enabled 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
from pathlib import Path

def verify_mask_folders():
    BASE_DIR = Path(r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset\cleargrasp-dataset-train\square-plastic-bottle-train")
    
    # Define suspected paths
    mask_dir = BASE_DIR / "segmentation-masks"
    
    print("[DIAGNOSTIC] Validating mask directories:")
    print(f"  Segmentation Mask Folder Exists: {mask_dir.exists()}")
    
    if mask_dir.exists():
        files = list(mask_dir.iterdir())
        if len(files) > 0:
            print(f"  Sample mask file found: {files[0].name}")
            img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"  Successfully read mask matrix. Shape: {img.shape} | Dtype: {img.dtype}")
            else:
                print("  [ERROR] Failed to read mask file matrix.")
        else:
            print("  [WARNING] Segmentation mask folder is empty.")

if __name__ == "__main__":
    verify_mask_folders()