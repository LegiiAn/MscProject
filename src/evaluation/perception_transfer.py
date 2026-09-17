"""
Perception-Transfer Evaluation on ZeroWaste-f  (WP2 front-end, zero-shot)
========================================================================
QUESTION: on real materials-recovery conveyor images, does the PERCEPTION
front-end degrade gracefully while the POLICY collapses into gibberish?

This scores the SAME 572 val frames the VLA saw (zerowaste_eval.py), so the
"front-end vs policy on identical images" contrast is valid.

THE rigid_plastic CAVEAT (encoded, not footnoted)
-------------------------------------------------
WP2 was trained to segment TRANSPARENT objects. ZeroWaste's closest class,
rigid_plastic (id 1), also contains OPAQUE rigid plastics (coloured tubs,
bottles). So a low IoU vs rigid_plastic is ambiguous:
  - perception genuinely fails on real images        (a real negative), OR
  - perception correctly ignores opaque rigid items  (class mismatch, NOT a failure).

To separate these, we DO NOT lead with IoU. We report three things:
  1. IoU vs rigid_plastic (primary, but caveated)      — comparability number
  2. IoU vs any-object (reference upper bound)
  3. HIT-PRECISION: of the pixels perception fires on, what fraction land on
     ANY annotated object vs background. This is class-mismatch-INDEPENDENT and
     answers the question that matters: is perception producing coherent object
     masks, or hallucinating on background?
  + qualitative overlays (real image + predicted mask) so you can SEE it.

CLAIM THIS SUPPORTS (narrow, defensible):
  "On real conveyor images, the perception front-end produces spatially coherent
   masks that fire predominantly on real objects, while the policy emits invalid
   output — the front-end degrades gracefully where the policy collapses."
It does NOT claim high IoU or accurate real-image perception. Only graceful
degradation vs collapse.

Preprocessing matches src/perception/utils/dataset.py EXACTLY:
  256x256, RGB, /255.0, NO ImageNet normalisation.

Run on cluster:
    python src/evaluation/perception_transfer.py --n 572
    python src/evaluation/perception_transfer.py --n 40 --overlays 8   # smoke + figure
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.perception.models.cleargrasp_net import ClearGraspDualNet

VAL = "data/zerowaste/data/splits_final_deblurred/val"
CLASS_NAMES = {0: "bg", 1: "rigid_plastic", 2: "cardboard", 3: "metal", 4: "soft_plastic"}


def iou(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    return float(inter / union) if union else np.nan


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=572)
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--ckpt", default="checkpoints/cleargrasp_dualnet_epoch_10.pth")
    ap.add_argument("--out", default="perception_transfer_results.json")
    ap.add_argument("--overlays", type=int, default=8, help="how many qualitative panels to save")
    ap.add_argument("--thresh", type=float, default=0.5, help="sigmoid threshold for the mask")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = Path(args.val)
    rgb_dir, seg_dir = val / "data", val / "sem_seg"
    stems = sorted(p.stem for p in rgb_dir.glob("*.PNG")) or \
            sorted(p.stem for p in rgb_dir.glob("*.png"))
    stems = stems[:args.n]
    print(f"[DATA] {len(stems)} val frames")

    print(f"[INIT] loading perception net {args.ckpt}...")
    model = ClearGraspDualNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
    model.eval()

    iou_rigid, iou_any = [], []
    hit_hits, hit_total = 0, 0          # hit-precision numerator/denominator (pixels)
    frames_fired = 0                    # frames where perception fired on >0.1% of pixels
    coverage = []                       # fraction of frame predicted as object
    overlays = []

    print("[EVAL] scoring perception on real frames...")
    with torch.no_grad():
        for fi, stem in enumerate(stems):
            bgr = cv2.imread(str(rgb_dir / f"{stem}.PNG"), cv2.IMREAD_COLOR)
            seg = cv2.imread(str(seg_dir / f"{stem}.PNG"), cv2.IMREAD_UNCHANGED)
            if bgr is None or seg is None:
                continue
            Hf, Wf = seg.shape[:2]

            # preprocessing IDENTICAL to dataset.py: RGB, 256, /255.0, no norm
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb256 = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(rgb256.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            _, seg_logits = model(x)
            pm = (torch.sigmoid(seg_logits[0, 0]).cpu().numpy() > args.thresh)
            pm_full = cv2.resize(pm.astype(np.uint8), (Wf, Hf),
                                 interpolation=cv2.INTER_NEAREST) > 0

            gt_rigid = (seg == 1)
            gt_any = (seg > 0)

            if gt_rigid.any():
                iou_rigid.append(iou(pm_full, gt_rigid))
            iou_any.append(iou(pm_full, gt_any))

            # hit-precision: of fired pixels, how many land on ANY object?
            fired = int(pm_full.sum())
            if fired > 0:
                hit_hits += int(np.logical_and(pm_full, gt_any).sum())
                hit_total += fired
            cov = fired / (Hf * Wf)
            coverage.append(cov)
            if cov > 0.001:
                frames_fired += 1

            # qualitative overlays
            if len(overlays) < args.overlays and gt_rigid.any():
                small = cv2.resize(rgb, (480, 270))
                pm_s = cv2.resize(pm_full.astype(np.uint8), (480, 270),
                                  interpolation=cv2.INTER_NEAREST) > 0
                overlays.append({"stem": stem, "rgb": small, "pred": pm_s})

            if (fi + 1) % 50 == 0:
                print(f"  {fi+1}/{len(stems)}")

    n_any = len(iou_any)
    ir = np.nanmean(iou_rigid) * 100 if iou_rigid else float("nan")
    ia = np.nanmean(iou_any) * 100 if iou_any else float("nan")
    hit_prec = 100 * hit_hits / hit_total if hit_total else float("nan")
    hlo, hhi = wilson_ci(hit_hits, hit_total)
    fired_pct = 100 * frames_fired / max(n_any, 1)
    mean_cov = 100 * float(np.mean(coverage)) if coverage else 0.0

    print("\n" + "=" * 80)
    print("PERCEPTION-TRANSFER ON ZEROWASTE-f  (WP2 front-end, zero-shot)")
    print("=" * 80)
    print(f"  frames scored                 : {n_any}")
    print(f"  frames where perception fired : {frames_fired}/{n_any} ({fired_pct:.0f}%)")
    print(f"  mean object-coverage of frame : {mean_cov:.1f}%")
    print("-" * 80)
    print("  PRIMARY (caveated) — segmentation IoU:")
    print(f"    IoU vs rigid_plastic        : {ir:5.1f}%   (n={len(iou_rigid)})  <- CAVEAT below")
    print(f"    IoU vs any-object (ref)     : {ia:5.1f}%   (n={n_any})")
    print("-" * 80)
    print("  CLASS-MISMATCH-INDEPENDENT — hit-precision:")
    print(f"    fired pixels on ANY object  : {hit_prec:5.1f}%  [{100*hlo:.0f},{100*hhi:.0f}]")
    print(f"    (of every pixel perception predicted as object, this fraction")
    print(f"     landed on a real annotated object rather than background)")
    print("=" * 80)
    print("CAVEAT: WP2 was trained on TRANSPARENT objects; ZeroWaste rigid_plastic")
    print("  includes OPAQUE rigid plastics, so IoU-vs-rigid_plastic understates")
    print("  perception if it correctly ignores opaque items. Hit-precision is the")
    print("  class-mismatch-independent read: high hit-precision = coherent masks on")
    print("  real objects (graceful degradation); low = hallucinating on background.")
    print("\nCONTRAST FOR SYNTHESIS:")
    print("  policy on these SAME frames: 0% on-target, ~100% gibberish output.")
    print("  if hit-precision is well above 0, the front-end degrades gracefully")
    print("  where the policy collapses -> 'perception transfers, policy does not'.")

    json.dump({
        "n_frames": n_any,
        "frames_fired": frames_fired, "frames_fired_pct": fired_pct,
        "mean_coverage_pct": mean_cov,
        "iou_rigid_plastic_pct": ir, "n_rigid": len(iou_rigid),
        "iou_any_object_pct": ia,
        "hit_precision_pct": hit_prec, "hit_precision_ci": [100*hlo, 100*hhi],
        "hit_pixels": hit_hits, "fired_pixels": hit_total,
        "caveat": "WP2 trained on transparent objects; rigid_plastic includes opaque items. "
                  "Lead with hit-precision + qualitative overlays, not IoU.",
    }, open(args.out, "w"), indent=2)
    print(f"\n[SAVED] {args.out}")

    # qualitative figure: real image with predicted mask outline
    if overlays:
        n = len(overlays)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 5.2))
        axes = np.array(axes).reshape(-1)
        for ax, ov in zip(axes, overlays):
            ax.imshow(ov["rgb"])
            # red outline of predicted mask
            cnts, _ = cv2.findContours(ov["pred"].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cc = c.squeeze()
                if cc.ndim == 2 and len(cc) > 2:
                    ax.plot(cc[:, 0], cc[:, 1], color="#E23", linewidth=1.4)
            ax.set_title(ov["stem"], fontsize=9)
            ax.axis("off")
        for ax in axes[len(overlays):]:
            ax.axis("off")
        fig.suptitle("WP2 perception on real ZeroWaste frames — predicted transparent-mask (red)",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig("perception_transfer_overlays.pdf", bbox_inches="tight")
        fig.savefig("perception_transfer_overlays.png", dpi=200, bbox_inches="tight")
        print("[SAVED] perception_transfer_overlays.pdf/.png  <- the slide visual")


if __name__ == "__main__":
    main()