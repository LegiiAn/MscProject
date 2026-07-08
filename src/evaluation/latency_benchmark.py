"""
Pipeline Latency Benchmark
=========================
Measures per-frame inference time for each stage:
    1. Perception forward pass (normals + mask)
    2. Poisson depth reconstruction (sparse solve)
    3. VLA autoregressive generation

Supports honest systems claims: "X seconds per frame, dominated by VLA
decoding" and sets up "real-time is future work" as a defensible caveat.

Requires GPU. Run on cluster.

Usage:
    python src/evaluation/latency_benchmark.py
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.models.cleargrasp_net import ClearGraspDualNet
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

FX = 918.0 * 0.5
FY = 918.0 * 0.5


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
    return x_sol.reshape(H, W)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CG_DIR = (Path.home() / "MscProject" / "data" / "cleargrasp_dataset"
              / "cleargrasp-dataset-train" / "square-plastic-bottle-train")
    PERCEPTION_CKPT = Path.home() / "MscProject" / "checkpoints" / "cleargrasp_dualnet_epoch_10.pth"
    VLA_CKPT = Path("vla_checkpoints/lora_epoch_2_step_13000")
    if not VLA_CKPT.exists():
        VLA_CKPT = Path("vla_checkpoints/lora_best")

    N_RUNS = 20
    SCALE = 0.5

    print("[INIT] Loading models...")
    perc_model = ClearGraspDualNet().to(device)
    ckpt = torch.load(PERCEPTION_CKPT, map_location=device)
    perc_model.load_state_dict(ckpt["model_state_dict"])
    perc_model.eval()

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True)
    vla_model = PeftModel.from_pretrained(base, VLA_CKPT)
    vla_model.eval()

    # Prepare one sample
    rgb_path = sorted((CG_DIR / "rgb-imgs").glob("*.jpg"))[0]
    stem = rgb_path.stem.replace("-rgb", "")
    depth_path = CG_DIR / "depth-imgs-rectified" / f"{stem}-depth-rectified.exr"
    mask_path = CG_DIR / "segmentation-masks" / f"{stem}-segmentation-mask.png"

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
    corrupted = depth_s.copy()
    corrupted[mask_s > 0] = 0.0

    rgb_t = torch.from_numpy(cv2.cvtColor(rgb_s, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    rgb_t = F.interpolate(rgb_t, size=(256, 256), mode="bilinear", align_corners=False).to(device)

    # ── Warmup ───────────────────────────────────────────────────────────
    print("[WARMUP] Running warmup passes...")
    for _ in range(3):
        with torch.no_grad():
            pred_n, _ = perc_model(rgb_t)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # ── Benchmark perception ─────────────────────────────────────────────
    print(f"[BENCH] Perception ({N_RUNS} runs)...")
    perc_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_n, pred_m = perc_model(rgb_t)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        perc_times.append(time.perf_counter() - t0)

    pred_n_up = F.interpolate(pred_n, size=(H_s, W_s), mode="bilinear", align_corners=False)
    pred_n_xyz = bgr_to_xyz(pred_n_up[0].cpu().numpy().transpose(1, 2, 0))

    # ── Benchmark depth reconstruction ───────────────────────────────────
    print(f"[BENCH] Depth reconstruction ({N_RUNS} runs)...")
    depth_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = poisson_reconstruct(corrupted, pred_n_xyz, mask_s, FX, FY)
        depth_times.append(time.perf_counter() - t0)

    # ── Benchmark VLA ────────────────────────────────────────────────────
    print(f"[BENCH] VLA generation ({N_RUNS} runs)...")
    
    # FIX: Convert the OpenCV numpy array to a PIL Image before passing it in
    pil_img = Image.fromarray(cv2.cvtColor(rgb_s, cv2.COLOR_BGR2RGB))
    
    inputs = processor(
        text="In: What action should the robot take to pick up the transparent plastic bottle.\nOut: ",
        images=pil_img,  # Pass the PIL Image here
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

    # Warmup VLA
    with torch.no_grad():
        _ = vla_model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=7, do_sample=False)

    vla_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = vla_model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=7, do_sample=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        vla_times.append(time.perf_counter() - t0)


    # ── Report ───────────────────────────────────────────────────────────
    def stats(times):
        arr = np.array(times) * 1000  # ms
        return np.mean(arr), np.std(arr)

    p_m, p_s = stats(perc_times)
    d_m, d_s = stats(depth_times)
    v_m, v_s = stats(vla_times)
    total = p_m + d_m + v_m

    print("\n" + "=" * 55)
    print("PIPELINE LATENCY BENCHMARK")
    print("=" * 55)
    print(f"{'Stage':<30} {'Time (ms)':>12} {'% total':>10}")
    print("-" * 55)
    print(f"{'1. Perception (normals+mask)':<30} {p_m:>8.1f}±{p_s:.0f} {p_m/total*100:>9.1f}%")
    print(f"{'2. Depth reconstruction':<30} {d_m:>8.1f}±{d_s:.0f} {d_m/total*100:>9.1f}%")
    print(f"{'3. VLA generation':<30} {v_m:>8.1f}±{v_s:.0f} {v_m/total*100:>9.1f}%")
    print("-" * 55)
    print(f"{'TOTAL per frame':<30} {total:>8.1f} ms")
    print(f"{'Throughput':<30} {1000/total:>8.2f} FPS")
    print("=" * 55)

    import json
    with open("latency_benchmark.json", "w") as f:
        json.dump({
            "perception_ms": {"mean": p_m, "std": p_s},
            "depth_ms": {"mean": d_m, "std": d_s},
            "vla_ms": {"mean": v_m, "std": v_s},
            "total_ms": total,
            "fps": 1000 / total,
        }, f, indent=2)
    print("[SAVED] latency_benchmark.json")


if __name__ == "__main__":
    main()