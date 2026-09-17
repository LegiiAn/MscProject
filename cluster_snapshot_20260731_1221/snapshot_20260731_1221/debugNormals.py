"""
Quick diagnostic: inspect the ClearGrasp normal map format.
Checks channel ranges, signs, and whether BGR/RGB swap is needed.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from pathlib import Path

DATA_DIR = (
    Path.home() / "MscProject" / "data" / "cleargrasp_dataset"
    / "cleargrasp-dataset-train" / "square-plastic-bottle-train"
)

normal_dir = DATA_DIR / "camera-normals"
mask_dir = DATA_DIR / "segmentation-masks"
depth_dir = DATA_DIR / "depth-imgs-rectified"

# Find first available file
normal_files = sorted(normal_dir.glob("*.exr"))
print(f"Found {len(normal_files)} normal files")

if not normal_files:
    print("No EXR normal files found!")
    exit()

# Load first normal map
npath = normal_files[0]
print(f"\nInspecting: {npath.name}")

n_img = cv2.imread(str(npath), cv2.IMREAD_UNCHANGED)
print(f"Shape: {n_img.shape}")
print(f"Dtype: {n_img.dtype}")
print(f"Overall range: [{n_img.min():.4f}, {n_img.max():.4f}]")

# Check per-channel (OpenCV loads as BGR)
for i, label in enumerate(["Channel 0 (B in BGR)", "Channel 1 (G in BGR)", "Channel 2 (R in BGR)"]):
    ch = n_img[:, :, i]
    print(f"  {label}: [{ch.min():.4f}, {ch.max():.4f}], mean={ch.mean():.4f}")

if n_img.shape[2] > 3:
    print(f"  Channel 3 (Alpha): [{n_img[:,:,3].min():.4f}, {n_img[:,:,3].max():.4f}]")

# Load corresponding mask to isolate object pixels
stem = npath.stem.replace("-cameraNormals", "")
mask_candidates = list(mask_dir.glob(f"{stem}*"))
if mask_candidates:
    mask = cv2.imread(str(mask_candidates[0]), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    obj_pixels = mask > 0
    
    print(f"\n--- Object pixels only ({np.sum(obj_pixels)} pixels) ---")
    for i, label in enumerate(["Ch0 (B)", "Ch1 (G)", "Ch2 (R)"]):
        ch = n_img[:, :, i][obj_pixels]
        print(f"  {label}: [{ch.min():.4f}, {ch.max():.4f}], mean={ch.mean():.4f}, std={ch.std():.4f}")

    # Check: are these already unit vectors?
    normals_bgr = n_img[:, :, :3].astype(np.float32)
    magnitudes = np.linalg.norm(normals_bgr, axis=2)
    obj_mags = magnitudes[obj_pixels]
    print(f"\n  Vector magnitudes on object: mean={obj_mags.mean():.4f}, std={obj_mags.std():.4f}")
    print(f"  (Should be ~1.0 if true unit normals)")

    # Check if normals are in [0,1] range (need remapping to [-1,1]) 
    # or already in [-1,1] range
    if n_img[:, :, :3].min() >= -0.01:
        print(f"\n  ⚠️  All values >= 0. Normals are likely in [0,1] range.")
        print(f"  Need to remap: normal = image * 2.0 - 1.0")
    else:
        print(f"\n  ✓ Values include negatives. Normals likely already in [-1,1] range.")

# Also check depth format
print(f"\n--- Depth format check ---")
depth_files = sorted(depth_dir.glob("*.exr"))
if depth_files:
    d = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    print(f"Shape: {d.shape}, Dtype: {d.dtype}")
    if len(d.shape) == 3:
        d = d[:, :, 0]
    print(f"Range: [{d.min():.4f}, {d.max():.4f}] metres")
    print(f"Mean: {d.mean():.4f}, Median: {np.median(d):.4f}")
    valid = d[d > 0]
    if len(valid) > 0:
        print(f"Valid (>0) range: [{valid.min():.4f}, {valid.max():.4f}]")

print("\n--- Filename patterns ---")
print(f"Normal example: {normal_files[0].name}")
depth_files = sorted(depth_dir.glob("*.exr"))
if depth_files:
    print(f"Depth example:  {depth_files[0].name}")
mask_files = sorted(mask_dir.glob("*.png"))
if mask_files:
    print(f"Mask example:   {mask_files[0].name}")
rgb_files = sorted((DATA_DIR / "rgb-imgs").glob("*.jpg")) + sorted((DATA_DIR / "rgb-imgs").glob("*.png"))
if rgb_files:
    print(f"RGB example:    {rgb_files[0].name}")
