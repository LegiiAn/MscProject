"""
Lambda_anchor Sensitivity Analysis
==================================
Defends the hardcoded lambda_anchor=100 in the Poisson solver by showing
reconstruction RMSE is stable across a range of values. Preempts the
examiner question "did you tune that number to look good?"

Reruns Method C (Poisson with GT normals) at lambda in {10, 50, 100, 500, 1000}
on a subset of images and reports RMSE for each.

Requires GPU only if using predicted normals; with GT normals it's CPU-only.
This version uses GT normals (CPU-friendly).

Usage:
    python src/evaluation/lambda_sensitivity.py
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

FX = 918.0 * 0.5
FY = 918.0 * 0.5
DEPTH_CLIP = 2.5


def bgr_to_xyz(n):
    return n[:, :, ::-1].copy()


def poisson_reconstruct(raw_depth, normals_xyz, mask, fx, fy, lambda_anchor):
    H, W = raw_depth.shape
    num_px = H * W
    idx = np.arange(num_px).reshape(H, W)
    Nx, Ny, Nz = normals_xyz[:, :, 0], normals_xyz[:, :, 1], normals_xyz[:, :, 2]
    Nz_safe = np.where(np.abs(Nz) < 1e-5, 1e-5, Nz)
    hole = (mask > 0) | (raw_depth <= 0)
    anchor = ~hole & (raw_depth > 0)
    if not np.any(anchor) or not np.any(hole):
        return raw_depth.copy()
    kernel = np.ones((3, 3), np.uint8)
    boundary = (cv2.dilate(hole.astype(np.uint8), kernel, 2) > 0) & anchor
    Z_approx = np.mean(raw_depth[boundary]) if np.any(boundary) else np.mean(raw_depth[anchor])
    p = -Nx * Z_approx / (Nz_safe * fx)
    q = -Ny * Z_approx / (Nz_safe * fy)
    hole_x = hole[:, :-1] | hole[:, 1:]
    xr, xc = np.where(hole_x)
    n_x = len(xr)
    eq_x = np.arange(n_x)
    rows_x = np.concatenate([eq_x, eq_x])
    cols_x = np.concatenate([idx[xr, xc + 1], idx[xr, xc]])
    data_x = np.concatenate([np.ones(n_x), -np.ones(n_x)])
    b_x = p[xr, xc]
    hole_y = hole[:-1, :] | hole[1:, :]
    yr, yc = np.where(hole_y)
    n_y = len(yr)
    eq_y = np.arange(n_y) + n_x
    rows_y = np.concatenate([eq_y, eq_y])
    cols_y = np.concatenate([idx[yr + 1, yc], idx[yr, yc]])
    data_y = np.concatenate([np.ones(n_y), -np.ones(n_y)])
    b_y = q[yr, yc]
    ay, ax = np.where(anchor)
    af = idx[ay, ax]
    n_a = len(af)
    eq_a = np.arange(n_a) + n_x + n_y
    A = sparse.csr_matrix(
        (np.concatenate([data_x, data_y, np.full(n_a, lambda_anchor)]),
         (np.concatenate([rows_x, rows_y, eq_a]),
          np.concatenate([cols_x, cols_y, af]))),
        shape=(n_x + n_y + n_a, num_px))
    b = np.concatenate([b_x, b_y, lambda_anchor * raw_depth[ay, ax]]).astype(np.float64)
    x_sol = splinalg.lsqr(A, b, iter_lim=500)[0]
    return np.clip(x_sol.reshape(H, W).astype(np.float32), 0, DEPTH_CLIP)


def compute_rmse(pred, gt, mask):
    valid = (mask > 0) & np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if not np.any(valid):
        return np.nan
    diff = pred[valid] - gt[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


def main():
    DATA_DIR = (Path.home() / "MscProject" / "data" / "cleargrasp_dataset"
                / "cleargrasp-dataset-train" / "square-plastic-bottle-train")
    rgb_dir = DATA_DIR / "rgb-imgs"
    depth_dir = DATA_DIR / "depth-imgs-rectified"
    mask_dir = DATA_DIR / "segmentation-masks"
    normal_dir = DATA_DIR / "camera-normals"

    SCALE = 0.5
    N_IMAGES = 30
    LAMBDAS = [10, 50, 100, 500, 1000]

    rgb_files = sorted(rgb_dir.glob("*.jpg"))[:N_IMAGES]
    results = {lam: [] for lam in LAMBDAS}

    for rgb_path in tqdm(rgb_files, desc="Lambda sweep"):
        stem = rgb_path.stem.replace("-rgb", "")
        depth_path = depth_dir / f"{stem}-depth-rectified.exr"
        mask_path = mask_dir / f"{stem}-segmentation-mask.png"
        normal_path = normal_dir / f"{stem}-cameraNormals.exr"
        if not all(p.exists() for p in [depth_path, mask_path, normal_path]):
            continue

        raw_d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if len(raw_d.shape) == 3:
            raw_d = raw_d[:, :, 0]
        raw_d = raw_d.astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        n_img = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
        if n_img is None or len(n_img.shape) != 3:
            continue
        normals = bgr_to_xyz(n_img[:, :, :3].astype(np.float32))

        H_s = int(raw_d.shape[0] * SCALE)
        W_s = int(raw_d.shape[1] * SCALE)
        depth_s = cv2.resize(raw_d, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
        mask_s = cv2.resize(mask, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
        _, mask_s = cv2.threshold(mask_s, 127, 255, cv2.THRESH_BINARY)
        normals_s = cv2.resize(normals, (W_s, H_s))
        nrm = np.linalg.norm(normals_s, axis=2, keepdims=True)
        normals_s = normals_s / np.where(nrm < 1e-6, 1.0, nrm)

        if np.sum(mask_s > 0) < 50:
            continue

        corrupted = depth_s.copy()
        corrupted[mask_s > 0] = 0.0

        for lam in LAMBDAS:
            recon = poisson_reconstruct(corrupted, normals_s, mask_s, FX, FY, lam)
            rmse = compute_rmse(recon, depth_s, mask_s)
            if not np.isnan(rmse):
                results[lam].append(rmse)

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("LAMBDA_ANCHOR SENSITIVITY ANALYSIS")
    print("=" * 50)
    print(f"{'Lambda':>10} {'Mean RMSE (m)':>15} {'Std':>10} {'N':>6}")
    print("-" * 50)
    means = []
    stds = []
    for lam in LAMBDAS:
        data = results[lam]
        if data:
            m, s = np.mean(data), np.std(data)
            means.append(m)
            stds.append(s)
            print(f"{lam:>10} {m:>15.4f} {s:>10.4f} {len(data):>6}")
        else:
            means.append(np.nan)
            stds.append(np.nan)
    print("=" * 50)

    # Variation across lambdas
    valid_means = [m for m in means if not np.isnan(m)]
    if valid_means:
        variation = (max(valid_means) - min(valid_means)) / np.mean(valid_means) * 100
        print(f"RMSE variation across lambda range: {variation:.1f}%")
        if variation < 10:
            print(">> STABLE: Choice of lambda has minimal impact on results.")
        else:
            print(">> SENSITIVE: Lambda choice matters; report the tuning.")

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.errorbar(LAMBDAS, means, yerr=stds, marker="o", markersize=8,
                linewidth=2, capsize=5, color="#1f77b4")
    ax.axvline(100, color="red", linestyle="--", alpha=0.6, label="Chosen (λ=100)")
    ax.set_xscale("log")
    ax.set_xlabel("λ_anchor (log scale)")
    ax.set_ylabel("Mean RMSE (metres)")
    ax.set_title("Depth Reconstruction Sensitivity to Anchor Weight", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lambda_sensitivity.pdf", format="pdf", bbox_inches="tight")
    print("[SAVED] lambda_sensitivity.pdf")


if __name__ == "__main__":
    main()