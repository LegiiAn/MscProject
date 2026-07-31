import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2
from pathlib import Path
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


def reconstruct_depth_poisson(raw_depth, predicted_normals, predicted_mask, lambda_anchor=100):
    """
    Reconstructs 3D depth by integrating predicted surface normals via
    a sparse least-squares system (Poisson-style depth reconstruction).
    
    Enforces:
      - X-gradient constraints: Z(y,x+1) - Z(y,x) = p(y,x)
      - Y-gradient constraints: Z(y+1,x) - Z(y,x) = q(y,x) 
      - Anchor constraints: Z(y,x) = raw_depth(y,x) for valid background pixels
      
    Uses fully vectorized numpy operations instead of nested Python loops
    for practical runtime on full-resolution images.
    """
    H, W = raw_depth.shape
    num_pixels = H * W
    
    # Pixel index grid for mapping (y, x) -> flat index
    pixel_indices = np.arange(num_pixels).reshape(H, W)
    
    # 1. Compute target surface gradients from predicted normals
    Nx = predicted_normals[:, :, 0]
    Ny = predicted_normals[:, :, 1]
    Nz = predicted_normals[:, :, 2]
    Nz_safe = np.where(np.abs(Nz) < 1e-5, 1e-5, Nz)
    p = (-Nx / Nz_safe) / fx   # dZ/dx
    q = (-Ny / Nz_safe) / fy   # dZ/dy
    
    # 2. Identify anchor vs hole regions
    hole_mask = (predicted_mask > 0) | (raw_depth <= 0)
    anchor_mask = ~hole_mask & (raw_depth > 0)
    
    # ── Vectorized constraint construction ──────────────────────────────
    
    # --- Constraint Set 1: X-gradient equations ---
    # Z(y, x+1) - Z(y, x) = p(y, x) for all (y, x) where x < W-1
    x_current = pixel_indices[:, :-1].ravel()   # flat indices of (y, x)
    x_next = pixel_indices[:, 1:].ravel()       # flat indices of (y, x+1)
    n_xgrad = len(x_current)
    eq_x = np.arange(n_xgrad)
    
    rows_x = np.concatenate([eq_x, eq_x])
    cols_x = np.concatenate([x_next, x_current])
    data_x = np.concatenate([np.ones(n_xgrad), -np.ones(n_xgrad)])
    b_x = p[:, :-1].ravel()
    
    # --- Constraint Set 2: Y-gradient equations ---
    # Z(y+1, x) - Z(y, x) = q(y, x) for all (y, x) where y < H-1
    y_current = pixel_indices[:-1, :].ravel()
    y_below = pixel_indices[1:, :].ravel()
    n_ygrad = len(y_current)
    eq_y = np.arange(n_ygrad) + n_xgrad  # offset equation indices
    
    rows_y = np.concatenate([eq_y, eq_y])
    cols_y = np.concatenate([y_below, y_current])
    data_y = np.concatenate([np.ones(n_ygrad), -np.ones(n_ygrad)])
    b_y = q[:-1, :].ravel()
    
    # --- Constraint Set 3: Anchor equations ---
    # lambda * Z(y, x) = lambda * raw_depth(y, x) for valid background pixels
    anchor_ys, anchor_xs = np.where(anchor_mask)
    anchor_flat = pixel_indices[anchor_ys, anchor_xs]
    n_anchors = len(anchor_flat)
    eq_a = np.arange(n_anchors) + n_xgrad + n_ygrad
    
    rows_a = eq_a
    cols_a = anchor_flat
    data_a = np.full(n_anchors, lambda_anchor)
    b_a = lambda_anchor * raw_depth[anchor_ys, anchor_xs]
    
    # --- Assemble the full sparse system ---
    total_eqs = n_xgrad + n_ygrad + n_anchors
    
    all_rows = np.concatenate([rows_x, rows_y, rows_a])
    all_cols = np.concatenate([cols_x, cols_y, cols_a])
    all_data = np.concatenate([data_x, data_y, data_a])
    all_b = np.concatenate([b_x, b_y, b_a]).astype(np.float64)
    
    A = sparse.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(total_eqs, num_pixels),
    )
    
    print(f"[SOLVER] {total_eqs:,} equations, {num_pixels:,} unknowns, "
          f"{len(all_data):,} non-zeros, {n_anchors:,} anchors")
    
    # 3. Solve via sparse LSQR
    result = splinalg.lsqr(A, all_b, iter_lim=500)
    x_solution = result[0]
    istop = result[1]
    itn = result[2]
    print(f"[SOLVER] Converged: stop_reason={istop}, iterations={itn}")
    
    reconstructed = x_solution.reshape(H, W).astype(np.float32)
    return np.clip(reconstructed, 0, 2.0)


def main():
    """Validation test using synthetic hemispherical normals on a real depth map."""
    BASE_DIR = Path(r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset"
                    r"\cleargrasp-dataset-train\square-plastic-bottle-train")
    depth_dir = BASE_DIR / "depth-imgs-rectified"
    
    if not depth_dir.exists():
        print(f"[ERROR] Directory not found: {depth_dir}")
        return
    
    try:
        sample_path = next(depth_dir.glob("*.exr"))
    except StopIteration:
        print("[ERROR] No .exr files found.")
        return
    
    raw = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print("[ERROR] Failed to load depth file.")
        return
    
    raw_depth = (raw[:, :, 0] if len(raw.shape) == 3 else raw).astype(np.float32)
    H, W = raw_depth.shape
    
    # Downsample for fast test
    scale = 0.25
    H_s, W_s = int(H * scale), int(W * scale)
    depth_small = cv2.resize(raw_depth, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
    
    # Synthetic hemisphere normals for testing
    normals = np.zeros((H_s, W_s, 3), dtype=np.float32)
    yy, xx = np.mgrid[:H_s, :W_s]
    cx, cy = W_s // 2, H_s // 2
    r_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    radius = min(H_s, W_s) // 3
    sphere = r_sq < radius ** 2
    
    zn = np.sqrt(np.maximum(0, radius ** 2 - r_sq)) / radius
    normals[:, :, 0] = np.where(sphere, (xx - cx) / radius, 0.0)
    normals[:, :, 1] = np.where(sphere, (yy - cy) / radius, 0.0)
    normals[:, :, 2] = np.where(sphere, zn, 1.0)
    
    print(f"[TEST] Running on {H_s}x{W_s} downsampled image...")
    result = reconstruct_depth_poisson(depth_small, normals, sphere.astype(np.uint8))
    
    print(f"[OK] Input range: [{depth_small.min():.4f}, {depth_small.max():.4f}]m")
    print(f"[OK] Output range: [{result.min():.4f}, {result.max():.4f}]m")


if __name__ == "__main__":
    main()