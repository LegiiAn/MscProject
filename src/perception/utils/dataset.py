import os
# Force enable OpenEXR decoding capability globally in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class ClearGraspPerceptionDataset(Dataset):
    """
    Multi-modal Dataset for loading RGB images, Ground-Truth Surface Normals,
    and Binary Segmentation Masks for transparent object perception processing.
    """
    def __init__(self, base_dir: str, transform=None):
        self.base_dir = Path(base_dir)
        self.rgb_dir = self.base_dir / "rgb-imgs"
        self.normals_dir = self.base_dir / "camera-normals"
        self.masks_dir = self.base_dir / "segmentation-masks"
        self.transform = transform
        
        if not all([self.rgb_dir.exists(), self.normals_dir.exists(), self.masks_dir.exists()]):
            raise FileNotFoundError(f"One or more required directories are missing in {base_dir}")
            
        # Lock onto the JPG files you verified
        self.rgb_files = sorted(list(self.rgb_dir.glob("*.jpg")) + list(self.rgb_dir.glob("*.jpeg")))
        self.dataset_triplets = []
        
        for rgb_path in self.rgb_files:
            ext = rgb_path.suffix
            prefix = rgb_path.name.replace(f"-rgb{ext}", "")
            
            # Map out paths for the matching modalities
            normal_path = self.normals_dir / f"{prefix}-cameraNormals.exr"
            mask_path = self.masks_dir / f"{prefix}-segmentation-mask.png"
            
            if normal_path.exists() and mask_path.exists():
                self.dataset_triplets.append((rgb_path, normal_path, mask_path))
                
        if len(self.dataset_triplets) == 0:
            raise RuntimeError(f"Zero complete data triplets matched inside {base_dir}")

    def __len__(self):
        return len(self.dataset_triplets)

    def __getitem__(self, idx):
        rgb_path, normal_path, mask_path = self.dataset_triplets[idx]
        
        # 1. Ingest and convert RGB
        rgb_img = cv2.imread(str(rgb_path))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 2. Ingest and convert EXR Surface Normals
        normal_img = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        # 3. Ingest and convert Segmentation Mask (Grayscale)
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # --- Resolution Downsampling for Memory Safety ---
        rgb_img = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_AREA)
        normal_img = cv2.resize(normal_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_img = cv2.resize(mask_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # --- Value Normalization ---
        rgb_data = rgb_img.astype(np.float32) / 255.0
        normal_data = normal_img.astype(np.float32)  # Already mapped to [-1, 1] range
        
        # Binarize mask values to absolute 0.0 (background) or 1.0 (transparent object)
        mask_data = (mask_img > 0).astype(np.float32)
        
        # --- PyTorch Tensor Packing ---
        rgb_tensor = torch.from_numpy(rgb_data).permute(2, 0, 1)      # [3, H, W]
        normal_tensor = torch.from_numpy(normal_data).permute(2, 0, 1)  # [3, H, W]
        mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)         # [1, H, W] (Adds channel dim)
        
        return rgb_tensor, normal_tensor, mask_tensor

if __name__ == "__main__":
    TEST_PATH = r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset\cleargrasp-dataset-train\square-plastic-bottle-train"
    try:
        dataset = ClearGraspPerceptionDataset(base_dir=TEST_PATH)
        print(f"[SUCCESS] Multi-modal pipeline loaded with {len(dataset)} complete triplets.")
        
        rgb, normal, mask = dataset[0]
        print(f"RGB Tensor Shape:     {rgb.shape}")
        print(f"Normal Tensor Shape:  {normal.shape}")
        print(f"Mask Tensor Shape:    {mask.shape} | Unique Values: {torch.unique(mask).tolist()}")
    except Exception as e:
        print(f"[FAILURE] Triplet assembly crashed: {str(e)}")