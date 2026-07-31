"""
Ablation Study: Depth Reconstruction for Transparent Objects
============================================================
Compares depth recovery strategies on ClearGrasp data:

    A) Raw corrupted depth  — sensor with transparent regions zeroed
    B) cv2.inpaint baseline — texture interpolation (no 3D awareness)
    C) Poisson reconstruction with GT normals   — upper bound
    D) Poisson reconstruction with PREDICTED normals — realistic pipeline

Metrics computed ONLY over the transparent object mask region.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.models.cleargrasp_net import ClearGraspDualNet

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

DEPTH_CLIP = 2.5

# ClearGrasp uses Intel RealSense D415 intrinsics at 1920x1080
# fx = fy ≈ 918 pixels at full resolution
FX_FULL = 918.0
FY_FULL = 918.0


def bgr_to_xyz_normals(normals_bgr):
    """ClearGrasp EXR via cv2: BGR=[Nz,Ny,Nx] -> XYZ=[Nx,Ny,Nz]"""
    return normals_bgr[:, :, ::-1].copy()


def poisson_reconstruct(raw_depth, normals_xyz, mask, fx, fy, lambda_anchor=100.0):
    """
    Boundary-anchored Poisson depth integration with perspective camera model.
    
    The depth gradient in pixel space relates to surface normals as:
        dZ/du = -Nx * Z / (Nz * fx)
        dZ/dv = -Ny * Z / (Nz * fy)
    
    Since Z is unknown inside the hole, we approximate with the mean
    boundary depth (valid pixels adjacent to the hole).
    """
    H, W = raw_depth.shape
    num_px = H * W
    idx = np.arange(num_px).reshape(H, W)

    Nx = normals_xyz[:, :, 0]
    Ny = normals_xyz[:, :, 1]
    Nz = normals_xyz[:, :, 2]
    Nz_safe = np.where(np.abs(Nz) < 1e-5, 1e-5, Nz)

    hole = (mask > 0) | (raw_depth <= 0)
    anchor = ~hole & (raw_depth > 0)

    if not np.any(anchor) or not np.any(hole):
        return raw_depth.copy()

    # Estimate Z for the depth-dependent gradient term using boundary pixels
    # (pixels adjacent to the hole that have valid depth)
    kernel = np.ones((3, 3), np.uint8)
    dilated_hole = cv2.dilate(hole.astype(np.uint8), kernel, iterations=2)
    boundary = (dilated_hole > 0) & anchor
    if np.any(boundary):
        Z_approx = np.mean(raw_depth[boundary])
    else:
        Z_approx = np.mean(raw_depth[anchor])

    # Perspective-correct gradients in pixel space
    # dZ/du ≈ -Nx * Z_approx / (Nz * fx)
    # dZ/dv ≈ -Ny * Z_approx / (Nz * fy)
    p = -Nx * Z_approx / (Nz_safe * fx)   # dZ/du in metres per pixel
    q = -Ny * Z_approx / (Nz_safe * fy)   # dZ/dv in metres per pixel

    # X-gradient equations: only inside or touching the hole
    hole_x = hole[:, :-1] | hole[:, 1:]
    x_rows, x_cols_local = np.where(hole_x)
    n_x = len(x_rows)
    if n_x > 0:
        eq_x = np.arange(n_x)
        idx_curr_x = idx[x_rows, x_cols_local]
        idx_next_x = idx[x_rows, x_cols_local + 1]
        rows_x = np.concatenate([eq_x, eq_x])
        cols_x = np.concatenate([idx_next_x, idx_curr_x])
        data_x = np.concatenate([np.ones(n_x), -np.ones(n_x)])
        b_x = p[x_rows, x_cols_local]
    else:
        rows_x = cols_x = np.array([], dtype=int)
        data_x = b_x = np.array([], dtype=float)

    # Y-gradient equations: only inside or touching the hole
    hole_y = hole[:-1, :] | hole[1:, :]
    y_rows_local, y_cols = np.where(hole_y)
    n_y = len(y_rows_local)
    eq_offset = n_x
    if n_y > 0:
        eq_y = np.arange(n_y) + eq_offset
        idx_curr_y = idx[y_rows_local, y_cols]
        idx_below_y = idx[y_rows_local + 1, y_cols]
        rows_y = np.concatenate([eq_y, eq_y])
        cols_y = np.concatenate([idx_below_y, idx_curr_y])
        data_y = np.concatenate([np.ones(n_y), -np.ones(n_y)])
        b_y = q[y_rows_local, y_cols]
    else:
        rows_y = cols_y = np.array([], dtype=int)
        data_y = b_y = np.array([], dtype=float)

    # Anchor equations: lock background with strong weight
    ay, ax_c = np.where(anchor)
    af = idx[ay, ax_c]
    n_a = len(af)
    eq_a = np.arange(n_a) + n_x + n_y

    total_eq = n_x + n_y + n_a
    A = sparse.csr_matrix(
        (np.concatenate([data_x, data_y, np.full(n_a, lambda_anchor)]),
         (np.concatenate([rows_x, rows_y, eq_a]),
          np.concatenate([cols_x, cols_y, af]))),
        shape=(total_eq, num_px),
    )
    b = np.concatenate([b_x, b_y, lambda_anchor * raw_depth[ay, ax_c]]).astype(np.float64)

    x_sol = splinalg.lsqr(A, b, iter_lim=500)[0]
    return np.clip(x_sol.reshape(H, W).astype(np.float32), 0, DEPTH_CLIP)


def inpaint_depth(raw_depth, mask):
    hole = ((mask > 0) | (raw_depth <= 0)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(hole, kernel, iterations=1)
    return cv2.inpaint(raw_depth.astype(np.float32), dilated, inpaintRadius=5, flags=cv2.INPAINT_NS)


def compute_metrics(pred_depth, gt_depth, eval_mask):
    valid = eval_mask > 0
    if not np.any(valid):
        return {"rmse": np.nan, "mae": np.nan, "rel": np.nan, "n_pixels": 0}
    pred = pred_depth[valid]
    gt = gt_depth[valid]
    finite = np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if not np.any(finite):
        return {"rmse": np.nan, "mae": np.nan, "rel": np.nan, "n_pixels": 0}
    pred, gt = pred[finite], gt[finite]
    diff = np.abs(pred - gt)
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(diff)),
        "rel": float(np.mean(diff / gt)),
        "n_pixels": int(np.sum(finite)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = (
        Path.home() / "MscProject" / "data" / "cleargrasp_dataset"
        / "cleargrasp-dataset-train" / "square-plastic-bottle-train"
    )
    CHECKPOINT = Path.home() / "MscProject" / "checkpoints" / "cleargrasp_dualnet_epoch_10.pth"

    rgb_dir = DATA_DIR / "rgb-imgs"
    depth_dir = DATA_DIR / "depth-imgs-rectified"
    mask_dir = DATA_DIR / "segmentation-masks"
    normal_dir = DATA_DIR / "camera-normals"

    SCALE = 0.5
    MAX_IMAGES = 100

    # Scale focal length to working resolution
    fx_scaled = FX_FULL * SCALE
    fy_scaled = FY_FULL * SCALE

    print("[INIT] Loading perception model...")
    model = ClearGraspDualNet().to(device)
    if CHECKPOINT.exists():
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("[OK] Loaded checkpoint")
    else:
        print(f"[WARN] No checkpoint at {CHECKPOINT}")
    model.eval()

    rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    if MAX_IMAGES:
        rgb_files = rgb_files[:MAX_IMAGES]
    print(f"[DATA] Evaluating {len(rgb_files)} images at scale={SCALE}")
    print(f"[CAMERA] fx={fx_scaled:.1f}, fy={fy_scaled:.1f} at scale={SCALE}")

    methods = ["raw_corrupted", "inpaint_baseline", "poisson_gt_normals", "poisson_pred_normals"]
    all_metrics = {m: [] for m in methods}
    qualitative_samples = []
    skipped = 0

    for rgb_path in tqdm(rgb_files, desc="Ablation"):
        stem = rgb_path.stem
        base_stem = stem.replace("-rgb", "") if "-rgb" in stem else stem

        depth_path = depth_dir / f"{base_stem}-depth-rectified.exr"
        mask_path = mask_dir / f"{base_stem}-segmentation-mask.png"
        normal_path = normal_dir / f"{base_stem}-cameraNormals.exr"

        if not depth_path.exists() or not mask_path.exists():
            skipped += 1
            continue

        rgb_img = cv2.imread(str(rgb_path))
        raw_depth_full = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        gt_mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or raw_depth_full is None or gt_mask_full is None:
            skipped += 1
            continue

        if len(raw_depth_full.shape) == 3:
            raw_depth_full = raw_depth_full[:, :, 0]
        raw_depth_full = raw_depth_full.astype(np.float32)
        _, gt_mask_full = cv2.threshold(gt_mask_full, 10, 255, cv2.THRESH_BINARY)

        gt_normals_full = None
        if normal_path.exists():
            n_img = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
            if n_img is not None and len(n_img.shape) == 3 and n_img.shape[2] >= 3:
                gt_normals_full = bgr_to_xyz_normals(n_img[:, :, :3].astype(np.float32))

        H_s = int(raw_depth_full.shape[0] * SCALE)
        W_s = int(raw_depth_full.shape[1] * SCALE)

        gt_depth = cv2.resize(raw_depth_full, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask_full, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
        _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
        rgb_small = cv2.resize(rgb_img, (W_s, H_s))

        gt_normals = None
        if gt_normals_full is not None:
            gt_normals = cv2.resize(gt_normals_full, (W_s, H_s))
            norms = np.linalg.norm(gt_normals, axis=2, keepdims=True)
            norms = np.where(norms < 1e-6, 1.0, norms)
            gt_normals = gt_normals / norms

        if np.sum(gt_mask > 0) < 50:
            continue

        corrupted_depth = gt_depth.copy()
        corrupted_depth[gt_mask > 0] = 0.0

        # Method A: Raw corrupted
        all_metrics["raw_corrupted"].append(compute_metrics(corrupted_depth, gt_depth, gt_mask))

        # Method B: cv2.inpaint
        inpainted = inpaint_depth(corrupted_depth, gt_mask)
        all_metrics["inpaint_baseline"].append(compute_metrics(inpainted, gt_depth, gt_mask))

        # Method C: Poisson with GT normals
        if gt_normals is not None:
            poisson_gt = poisson_reconstruct(corrupted_depth, gt_normals, gt_mask, fx_scaled, fy_scaled)
            all_metrics["poisson_gt_normals"].append(compute_metrics(poisson_gt, gt_depth, gt_mask))
        else:
            poisson_gt = None
            all_metrics["poisson_gt_normals"].append(
                {"rmse": np.nan, "mae": np.nan, "rel": np.nan, "n_pixels": 0})

        # Method D: Poisson with predicted normals
        rgb_tensor = torch.from_numpy(
            cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0
        rgb_tensor = F.interpolate(rgb_tensor, size=(256, 256), mode="bilinear", align_corners=False)
        rgb_tensor = rgb_tensor.to(device)

        with torch.no_grad():
            pred_normals_tensor, _ = model(rgb_tensor)

        pred_normals_up = F.interpolate(
            pred_normals_tensor, size=(H_s, W_s), mode="bilinear", align_corners=False
        )
        pred_normals_np = pred_normals_up[0].cpu().numpy().transpose(1, 2, 0)
        pred_normals_np = bgr_to_xyz_normals(pred_normals_np)

        poisson_pred = poisson_reconstruct(corrupted_depth, pred_normals_np, gt_mask, fx_scaled, fy_scaled)
        all_metrics["poisson_pred_normals"].append(compute_metrics(poisson_pred, gt_depth, gt_mask))

        if len(qualitative_samples) < 4:
            qualitative_samples.append({
                "rgb": cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB),
                "gt_depth": gt_depth, "corrupted": corrupted_depth,
                "inpainted": inpainted, "poisson_gt": poisson_gt,
                "poisson_pred": poisson_pred, "mask": gt_mask,
            })

    print(f"\nSkipped {skipped} images")
    print("\n" + "=" * 70)
    print("ABLATION RESULTS: Depth Reconstruction for Transparent Objects")
    print("=" * 70)
    print(f"{'Method':<30} {'RMSE (m)':>10} {'MAE (m)':>10} {'Rel Err':>10} {'N':>6}")
    print("-" * 70)

    summary = {}
    for method in methods:
        valid = [m for m in all_metrics[method] if not np.isnan(m["rmse"])]
        if not valid:
            print(f"{method:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>6}")
            continue
        rmses = [m["rmse"] for m in valid]
        maes = [m["mae"] for m in valid]
        rels = [m["rel"] for m in valid]
        summary[method] = {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "rel_mean": float(np.mean(rels)), "n_images": len(valid),
            "rmses": rmses, "maes": maes,
        }
        print(f"{method:<30} {np.mean(rmses):>10.4f} {np.mean(maes):>10.4f} "
              f"{np.mean(rels):>10.4f} {len(valid):>6}")

    print("=" * 70)
    if "inpaint_baseline" in summary and "poisson_gt_normals" in summary:
        imp = (1 - summary["poisson_gt_normals"]["rmse_mean"] / summary["inpaint_baseline"]["rmse_mean"]) * 100
        print(f"\nPoisson (GT normals) vs Inpaint:   {imp:+.1f}% RMSE change")
    if "inpaint_baseline" in summary and "poisson_pred_normals" in summary:
        imp = (1 - summary["poisson_pred_normals"]["rmse_mean"] / summary["inpaint_baseline"]["rmse_mean"]) * 100
        print(f"Poisson (Pred normals) vs Inpaint: {imp:+.1f}% RMSE change")

    results_out = {}
    for method in methods:
        if method in summary:
            s = summary[method]
            results_out[method] = {
                "rmse_mean": s["rmse_mean"], "rmse_std": s["rmse_std"],
                "mae_mean": s["mae_mean"], "rel_mean": s["rel_mean"],
                "n_images": s["n_images"],
            }
    # Dump per-image RMSE arrays for statistical testing
    per_image = {m: summary[m]["rmses"] for m in methods if m in summary}
    with open("ablation_per_image.json", "w") as f:
        json.dump(per_image, f)
    print("[SAVED] ablation_per_image.json")
    generate_thesis_figure(qualitative_samples, summary)

    


def generate_thesis_figure(samples, summary):
    if not samples or not summary:
        print("[WARN] Not enough data for figure.")
        return
    n_rows = min(len(samples), 3)
    fig = plt.figure(figsize=(16, 4 * n_rows + 4), dpi=300)
    gs = gridspec.GridSpec(n_rows + 1, 6, hspace=0.35, wspace=0.15)
    all_d = []
    for s in samples[:n_rows]:
        v = s["gt_depth"][s["gt_depth"] > 0]
        if len(v):
            all_d.extend(v.tolist())
    vmin = np.percentile(all_d, 2) if all_d else 0
    vmax = np.percentile(all_d, 98) if all_d else 1
    titles = ["RGB Input", "GT Depth", "Corrupted\n(Sensor)", "Inpainted\n(cv2)",
              "Poisson\n(GT Normals)", "Poisson\n(Predicted)"]
    for row in range(n_rows):
        s = samples[row]
        panels = [("rgb", s["rgb"]), ("depth", s["gt_depth"]), ("depth", s["corrupted"]),
                  ("depth", s["inpainted"]), ("depth", s["poisson_gt"]), ("depth", s["poisson_pred"])]
        for col, (ptype, data) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            if data is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_facecolor("#f0f0f0")
            elif ptype == "rgb":
                ax.imshow(data)
            else:
                ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
                contours, _ = cv2.findContours(
                    s["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    c = cnt.squeeze()
                    if c.ndim == 2 and len(c) > 2:
                        ax.plot(c[:, 0], c[:, 1], color="red", linewidth=0.8, alpha=0.7)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=9, fontweight="bold")
    ax_box = fig.add_subplot(gs[n_rows, :3])
    plot_methods = ["raw_corrupted", "inpaint_baseline", "poisson_gt_normals", "poisson_pred_normals"]
    plot_labels = ["Raw\nCorrupted", "cv2.inpaint\nBaseline", "Poisson\n(GT Normals)", "Poisson\n(Predicted)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    box_data = [summary[m]["rmses"] if m in summary else [] for m in plot_methods]
    bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.6)
    ax_box.set_xticklabels(plot_labels, fontsize=8)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax_box.set_ylabel("RMSE (metres)")
    ax_box.set_title("Depth Reconstruction Error", fontsize=11, fontweight="bold")
    ax_box.grid(True, axis="y", alpha=0.3)
    ax_bar = fig.add_subplot(gs[n_rows, 3:])
    x_pos = np.arange(len(plot_methods))
    means = [summary[m]["rmse_mean"] if m in summary else 0 for m in plot_methods]
    stds = [summary[m]["rmse_std"] if m in summary else 0 for m in plot_methods]
    bars = ax_bar.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(plot_labels, fontsize=8)
    ax_bar.set_ylabel("Mean RMSE (metres)")
    ax_bar.set_title("Depth Recovery Comparison", fontsize=11, fontweight="bold")
    ax_bar.grid(True, axis="y", alpha=0.3)
    for bar, mv in zip(bars, means):
        if mv > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{mv:.4f}m", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.savefig("ablation_depth_reconstruction.pdf", format="pdf", bbox_inches="tight")
    print("[SAVED] ablation_depth_reconstruction.pdf")


if __name__ == "__main__":
    main()