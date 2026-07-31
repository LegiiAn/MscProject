#!/usr/bin/env python3
"""
fig_augmentation_honest.py
==========================
Replaces the misleading `fig_augmentation_strip` (which grabbed 6 unrelated
finished composites and slapped process labels on them).

This version is HONEST BY CONSTRUCTION: it takes ONE real ClearGrasp bottle and
applies each transform INDEPENDENTLY to that same object, using the EXACT
parameter ranges from generatedata.py. An examiner could diff this against the
generator and the operations would match.

Panels (independent — each transform applied alone to the original):
  original · scale · rotation · flip · photometric · noise
plus one final "composited on background" panel to show the end product.

Transform ranges copied verbatim from generatedata.py:
  scale        random.uniform(0.25, 0.75)
  rotation     random.uniform(-15, 15)   BORDER_REFLECT (fg) / CONSTANT 0 (mask)
  flip         cv2.flip(., 1)
  photometric  contrast a~U(0.6,1.4), brightness b~U(-40,40),
               hue +U(-10,10), sat *U(0.7,1.3)
  noise        gaussian, std ~U(1,5)

Usage:
    python fig_augmentation_honest.py --root ~/MscProject --out slides_figs
    python fig_augmentation_honest.py --source 000000042-rgb.jpg   # force a bottle
"""

import argparse
import random
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the shared slide styling if available; otherwise fall back gracefully
try:
    from graphgenerate.slide_style import apply_style, save, C
    HAVE_STYLE = True
except Exception:
    HAVE_STYLE = False
    C = {"muted": "#6B6B6B", "ink": "#1A1A1A", "native": "#2A9D5C"}

IMG_SIZE = (224, 224)   # generatedata.py native resolution


# ---- transforms copied from generatedata.py (same ranges) -----------------
def photometric_jitter(img, rng):
    a = rng.uniform(0.6, 1.4)
    b = rng.integers(-40, 41)
    img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_noise(img, rng):
    noise = rng.normal(0, rng.uniform(1, 5), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---- source selection: a LARGE, central bottle ----------------------------
def pick_large_bottle(rgb_dir, mask_dir, rng, forced=None, tries=60):
    rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    if forced:
        rgb_files = [rgb_dir / forced] + rgb_files

    best = None
    best_area = 0
    checked = 0
    for rgb_path in ([rgb_dir / forced] if forced else rng.permutation(rgb_files)[:tries]):
        rgb_path = Path(rgb_path)
        if not rgb_path.exists():
            continue
        stem = rgb_path.stem
        mstem = stem.replace("-rgb", "-segmentation-mask")
        mask_path = mask_dir / f"{mstem}.png"
        if not mask_path.exists():
            continue
        fg = cv2.imread(str(rgb_path))
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if fg is None or m is None:
            continue
        _, mb = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
        area = int((mb > 0).sum())
        frac = area / (mb.shape[0] * mb.shape[1])
        checked += 1
        # want a clearly visible object: >8% of frame, single dominant blob
        if frac > 0.08 and area > best_area:
            best_area, best = area, (fg, mb, stem, frac)
        if forced:
            break
    return best, checked


def crop_to_object(fg, mask, pad=0.9):
    """Tight-ish crop around the object so the bottle is large in every panel."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return fg, mask
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(fg.shape[1], x1 + px), min(fg.shape[0], y1 + py)
    return fg[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="slides_figs")
    ap.add_argument("--source", default=None, help="force a specific -rgb.jpg filename")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    cat = root / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train" / "square-plastic-bottle-train"
    rgb_dir, mask_dir = cat / "rgb-imgs", cat / "segmentation-masks"
    bg_dir = root / "data" / "raw_backgrounds"
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if HAVE_STYLE:
        apply_style(base_fontsize=16)

    rng = np.random.default_rng(args.seed)
    picked, checked = pick_large_bottle(rgb_dir, mask_dir, rng, forced=args.source)
    if picked is None:
        print(f"[ERROR] no suitably large bottle found (checked {checked}). "
              f"Try --source <filename> or a different --seed.")
        return
    fg0, mask0, stem, frac = picked
    print(f"[OK] using {stem}  (object = {100*frac:.1f}% of frame)")

    # crop so the object is prominent, then resize to working size
    fg0, mask0 = crop_to_object(fg0, mask0)
    fg0 = cv2.resize(fg0, IMG_SIZE)
    mask0 = cv2.resize(mask0, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    _, mask0 = cv2.threshold(mask0, 127, 255, cv2.THRESH_BINARY)

    H, W = IMG_SIZE[1], IMG_SIZE[0]

    # ---- build each panel INDEPENDENTLY from the original --------------------
    panels = []

    # original
    panels.append(("original", to_rgb(fg0)))

    # scale: shrink the object within the frame (place smaller version on grey pad)
    s = 0.45  # representative of U(0.25,0.75)
    sw, sh = int(W * s), int(H * s)
    small = cv2.resize(fg0, (sw, sh))
    canvas = np.full_like(fg0, 200)  # light grey pad to show the object got smaller
    oy, ox = (H - sh) // 2, (W - sw) // 2
    canvas[oy:oy+sh, ox:ox+sw] = small
    panels.append(("scale  ×0.45", to_rgb(canvas)))

    # rotation: +15 deg, BORDER_REFLECT like the generator
    rot = cv2.getRotationMatrix2D((W // 2, H // 2), 15, 1.0)
    rotated = cv2.warpAffine(fg0, rot, (W, H), borderMode=cv2.BORDER_REFLECT)
    panels.append(("rotation  +15°", to_rgb(rotated)))

    # flip
    panels.append(("flip (horizontal)", to_rgb(cv2.flip(fg0, 1))))

    # photometric
    panels.append(("photometric jitter", to_rgb(photometric_jitter(fg0, rng))))

    # noise
    panels.append(("gaussian noise", to_rgb(add_noise(fg0, rng))))

    # final: composited on a real background (the end product)
    bgs = sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.png"))
    if bgs:
        bg = cv2.resize(cv2.imread(str(rng.choice(bgs))), IMG_SIZE)
        soft = cv2.GaussianBlur(mask0.astype(np.float32), (7, 7), 0)[:, :, None] / 255.0
        comp = (fg0 * soft + bg * (1 - soft)).astype(np.uint8)
        comp = add_noise(photometric_jitter(comp, rng), rng)
        panels.append(("→ final composite", to_rgb(comp)))

    # ---- render -------------------------------------------------------------
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.1))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img)
        ax.axis("off")
        colour = C.get("native", "#2A9D5C") if "final" in label else C.get("muted", "#6B6B6B")
        ax.set_title(label, fontsize=14, fontweight="bold", color=colour, pad=8)
    fig.suptitle("Domain randomisation — each transform applied to the same bottle",
                 fontsize=18, fontweight="bold", y=1.08)
    fig.tight_layout()

    if HAVE_STYLE:
        save(fig, out, "fig_augmentation_honest")
    else:
        p = out / "fig_augmentation_honest.png"
        fig.savefig(p, dpi=220, bbox_inches="tight", transparent=True)
        fig.savefig(out / "fig_augmentation_honest.pdf", bbox_inches="tight")
        print(f"  wrote {p}")
    print("Done. If the object is still small or awkward, try a different --seed "
          "or pass --source <a-rgb.jpg you like>.")


if __name__ == "__main__":
    main()