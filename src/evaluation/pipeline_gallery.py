"""
End-to-End Pipeline Gallery
===========================
The single figure examiners remember: for several sample images, show the
full pipeline in one row —

    RGB → Predicted Normals → Reconstructed Depth → VLA Grasp Overlay

Ties together every component of the thesis in one visual.

Requires GPU (perception + VLA inference). Run on cluster.

Usage:
    python src/evaluation/pipeline_gallery.py
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import re
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.models.cleargrasp_net import ClearGraspDualNet
from src.training.vla_dataset import OpenVLADataset

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

FX = 918.0 * 0.5
FY = 918.0 * 0.5
DEPTH_CLIP = 2.5


def bgr_to_xyz(n):
    return n[:, :, ::-1].copy()


def poisson_reconstruct(raw_depth, normals_xyz, mask, fx, fy, lambda_anchor=100.0):
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


def extract_action_values(text):
    out_pos = text.rfind("Out:")
    after = text[out_pos:] if out_pos != -1 else text
    full = re.findall(r"<action_(\d+)>", after)
    if len(full) >= 7:
        return [int(v) for v in full[:7]]
    flex = re.findall(r"(\d+)>", after)
    if len(flex) >= 7:
        vals = [int(v) for v in flex if 0 <= int(v) <= 255]
        if len(vals) >= 7:
            return vals[:7]
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CG_DIR = (Path.home() / "MscProject" / "data" / "cleargrasp_dataset"
              / "cleargrasp-dataset-train" / "square-plastic-bottle-train")
    PERCEPTION_CKPT = Path.home() / "MscProject" / "checkpoints" / "cleargrasp_dualnet_epoch_10.pth"
    VLA_CKPT = Path("vla_checkpoints/lora_epoch_2_step_13000")
    if not VLA_CKPT.exists():
        VLA_CKPT = Path("vla_checkpoints/lora_best")

    N_SAMPLES = 4
    SCALE = 0.5

    print("[INIT] Loading perception model...")
    perc_model = ClearGraspDualNet().to(device)
    ckpt = torch.load(PERCEPTION_CKPT, map_location=device)
    perc_model.load_state_dict(ckpt["model_state_dict"])
    perc_model.eval()

    print("[INIT] Loading VLA...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True)
    vla_model = PeftModel.from_pretrained(base, VLA_CKPT)
    vla_model.eval()

    # Sample from ClearGrasp for perception+depth, and processed for VLA
    rgb_files = sorted((CG_DIR / "rgb-imgs").glob("*.jpg"))[:N_SAMPLES]
    depth_dir = CG_DIR / "depth-imgs-rectified"
    mask_dir = CG_DIR / "segmentation-masks"

    rows = []
    for rgb_path in rgb_files:
        stem = rgb_path.stem.replace("-rgb", "")
        depth_path = depth_dir / f"{stem}-depth-rectified.exr"
        mask_path = mask_dir / f"{stem}-segmentation-mask.png"
        if not depth_path.exists() or not mask_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        raw_d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if len(raw_d.shape) == 3:
            raw_d = raw_d[:, :, 0]
        raw_d = raw_d.astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        H_s = int(rgb.shape[0] * SCALE)
        W_s = int(rgb.shape[1] * SCALE)
        rgb_s = cv2.resize(rgb, (W_s, H_s))
        depth_s = cv2.resize(raw_d, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
        mask_s = cv2.resize(mask, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
        _, mask_s = cv2.threshold(mask_s, 127, 255, cv2.THRESH_BINARY)

        # Perception
        rgb_t = torch.from_numpy(cv2.cvtColor(rgb_s, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        rgb_t = F.interpolate(rgb_t, size=(256, 256), mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            pred_n, pred_m = perc_model(rgb_t)
        pred_n_up = F.interpolate(pred_n, size=(H_s, W_s), mode="bilinear", align_corners=False)
        pred_n_np = pred_n_up[0].cpu().numpy().transpose(1, 2, 0)
        pred_n_xyz = bgr_to_xyz(pred_n_np)

        # Depth reconstruction
        corrupted = depth_s.copy()
        corrupted[mask_s > 0] = 0.0
        recon = poisson_reconstruct(corrupted, pred_n_xyz, mask_s, FX, FY)

        # VLA grasp prediction (using the composite pipeline prompt)
        # VLA grasp prediction (using the composite pipeline prompt)
        pil_rgb = cv2.cvtColor(rgb_s, cv2.COLOR_BGR2RGB) # Numpy array for the plot
        pil_img = Image.fromarray(pil_rgb)               # PIL Image for the processor

        inputs = processor(
            text="In: What action should the robot take to pick up the transparent plastic bottle.\nOut: ",
            images=pil_img,
            return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = vla_model.generate(input_ids=input_ids, pixel_values=pixel_values,
                                     max_new_tokens=30, do_sample=False)
        gen = processor.tokenizer.decode(out[0], skip_special_tokens=False)
        actions = extract_action_values(gen)
        if actions:
            gx = int((actions[0] / 255.0) * W_s)
            gy = int((actions[1] / 255.0) * H_s)
        else:
            gx, gy = W_s // 2, H_s // 2

        rows.append({
            "rgb": pil_rgb,
            "normals": np.clip((pred_n_np + 1) / 2, 0, 1),
            "depth": recon,
            "grasp": (gx, gy),
            "mask": mask_s,
        })

    # ── Figure ───────────────────────────────────────────────────────────
    n = len(rows)
    if n == 0:
        print("[ERROR] No samples processed.")
        return

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n), dpi=300)
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["RGB Input", "Predicted Normals", "Reconstructed Depth", "VLA Grasp Target"]

    d_all = np.concatenate([r["depth"][r["depth"] > 0].ravel() for r in rows])
    vmin, vmax = np.percentile(d_all, 2), np.percentile(d_all, 98)

    for i, r in enumerate(rows):
        axes[i, 0].imshow(r["rgb"])
        axes[i, 1].imshow(r["normals"])
        axes[i, 2].imshow(r["depth"], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 3].imshow(r["rgb"])
        # Grasp overlay
        contours, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            c = cnt.squeeze()
            if c.ndim == 2 and len(c) > 2:
                axes[i, 3].plot(c[:, 0], c[:, 1], color="cyan", linewidth=1)
        axes[i, 3].plot(r["grasp"][0], r["grasp"][1], "x", color="red",
                        markersize=14, markeredgewidth=3)

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("pipeline_gallery.pdf", format="pdf", bbox_inches="tight")
    print("[SAVED] pipeline_gallery.pdf")


if __name__ == "__main__":
    main()