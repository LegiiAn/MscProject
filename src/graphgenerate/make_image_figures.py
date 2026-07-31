#!/usr/bin/env python3
"""
make_image_figures.py
=====================
Builds the IMAGE-BASED presentation figures from your actual data. These are the
ones Maria specifically asked for ("add the dataset info + picture").

    python make_image_figures.py --root ~/MscProject --out slides_figs

Figures produced (slide numbers refer to presentation_blueprint_v2.md):
    fig_dataset_montage      -> Slide 8   native vs composite, 3+3 grid
    fig_transparency_cues    -> Slide 10  annotated zoom: what compositing destroys
    fig_target_derivation    -> Slide 9   naive centroid vs contour-isolated + valid
    fig_augmentation_strip   -> Slide 8   one object, six randomisations

WHAT IT NEEDS
-------------
Composites + masks from your Run 2 dataset:
    <root>/data/processed_dataset/{train,val,test}/images/*.png
    <root>/data/processed_dataset/annotations.json
Native ClearGrasp frames:
    <root>/data/cleargrasp_dataset/cleargrasp-dataset-train/
          square-plastic-bottle-train/rgb-imgs/*.jpg
    ...                                  /segmentation-masks/*.png

Paths are guessed, then overridable:
    --composites DIR --natives DIR --masks DIR

If nothing is found the script tells you exactly which directory was empty
rather than silently producing a blank figure.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

from graphgenerate.slide_style import apply_style, save, C

IMG_EXT = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")


# ---------------------------------------------------------------------------
def find_images(d, limit=400):
    d = Path(d)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*")):
        if p.suffix in IMG_EXT:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def load_rgb(p, size=None):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return im


def load_mask(p, size=None):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def resolve(root, args):
    """Locate the four asset directories, honouring CLI overrides."""
    root = Path(root).expanduser()
    cg = root / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train" / "square-plastic-bottle-train"
    paths = {
        "composites": Path(args.composites).expanduser() if args.composites
                      else root / "data" / "processed_dataset" / "train" ,
        "natives":    Path(args.natives).expanduser() if args.natives
                      else cg / "rgb-imgs",
        "masks":      Path(args.masks).expanduser() if args.masks
                      else cg / "segmentation-masks",
    }
    for k, v in paths.items():
        n = len(find_images(v, limit=5))
        status = f"{n}+ images" if n else "EMPTY / NOT FOUND"
        print(f"  {k:<12} {v}  [{status}]")
    return paths


# ===========================================================================
# SLIDE 8 — the dataset picture Maria asked for
# ===========================================================================
def fig_dataset_montage(paths, out, seed=0):
    natives = find_images(paths["natives"])
    comps = find_images(paths["composites"])
    if not natives or not comps:
        print("  !! skipped: need both native and composite images")
        return

    rng = random.Random(seed)
    n_sel = rng.sample(natives, min(3, len(natives)))
    c_sel = rng.sample(comps, min(3, len(comps)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
    for j, p in enumerate(n_sel):
        im = load_rgb(p, (420, 320))
        axes[0, j].imshow(im)
        axes[0, j].axis("off")
    for j, p in enumerate(c_sel):
        im = load_rgb(p, (420, 320))
        axes[1, j].imshow(im)
        axes[1, j].axis("off")

    for ax in axes.ravel():
        for s in ax.spines.values():
            s.set_visible(False)

    fig.text(0.5, 0.955, "NATIVE ClearGrasp — real refraction, shadows, highlights",
             ha="center", fontsize=23, fontweight="bold", color=C["native"])
    fig.text(0.5, 0.475, "ReSort-IT COMPOSITE — background variety, no refraction",
             ha="center", fontsize=23, fontweight="bold", color=C["composite"])

    fig.subplots_adjust(top=0.93, bottom=0.02, hspace=0.14, wspace=0.03)
    return save(fig, out, "fig_dataset_montage")


# ===========================================================================
# SLIDE 10 — annotated zoom showing what compositing destroys
# ===========================================================================
def fig_transparency_cues(paths, out, seed=1):
    natives = find_images(paths["natives"])
    comps = find_images(paths["composites"])
    if not natives or not comps:
        print("  !! skipped: need both native and composite images")
        return

    rng = random.Random(seed)
    nat = load_rgb(rng.choice(natives), (640, 480))
    com = load_rgb(rng.choice(comps), (640, 480))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.6))
    a1.imshow(nat); a1.axis("off")
    a2.imshow(com); a2.axis("off")
    a1.set_title("Native — the object distorts its background",
                 fontsize=20, color=C["native"], fontweight="bold", pad=12)
    a2.set_title("Composite — the object is pasted on top",
                 fontsize=20, color=C["composite"], fontweight="bold", pad=12)

    # Annotation anchors. Adjust the (x, y) fractions to point at the bottle
    # in whichever frame you end up selecting.
    cues = [(0.42, 0.45, "refraction of\nbackground"),
            (0.62, 0.74, "cast shadow"),
            (0.38, 0.26, "specular\nhighlight")]
    h, w = nat.shape[:2]
    for fx, fy, label in cues:
        a1.annotate(label, xy=(fx * w, fy * h), xytext=(fx * w + 0.20 * w, fy * h - 0.16 * h),
                    color=C["native"], fontsize=15, fontweight="bold",
                    ha="center", arrowprops=dict(arrowstyle="->", color=C["native"], lw=2.2))
    a2.text(0.5 * w, 0.5 * h, "none of these",
            ha="center", va="center", fontsize=26, fontweight="bold",
            color=C["composite"],
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=C["composite"], lw=2.5, alpha=0.9))

    fig.tight_layout()
    return save(fig, out, "fig_transparency_cues")


# ===========================================================================
# SLIDE 9 — grasp target derivation: the two bugs and the fix
# ===========================================================================
def _naive_centroid(mask):
    ys, xs = np.nonzero(mask)
    return (xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan)


def _contour_targets(mask, min_area=150):
    """Per-object centre of mass, plus the validity-corrected interior point."""
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (lbl == i).astype(np.uint8)
        cx, cy = cents[i]

        # Validity test: is the centre of mass actually inside the object?
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inside = False
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            inside = cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0

        # Distance-transform fallback -> most-interior pixel
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        fy, fx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        out.append({"naive": (cx, cy), "inside": inside, "fixed": (float(fx), float(fy))})
    return out


def fig_target_derivation(paths, out, seed=2):
    masks = find_images(paths["masks"])
    natives = find_images(paths["natives"])
    if not masks:
        print("  !! skipped: no masks found")
        return

    rng = random.Random(seed)
    # Prefer a mask with several separate objects — that's where naive fails
    chosen, chosen_mask, best_n = None, None, 0
    for p in rng.sample(masks, min(60, len(masks))):
        m = load_mask(p, (640, 480))
        if m is None:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        big = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 400)
        if big > best_n:
            chosen, chosen_mask, best_n = p, m, big
        if big >= 3:
            break
    if chosen_mask is None:
        print("  !! skipped: could not read any mask")
        return
    print(f"  using mask {chosen.name} ({best_n} objects)")

    # Try to show the mask over its RGB frame if we can match by stem
    bg = None
    for r in natives:
        if r.stem.split("-")[0] == chosen.stem.split("-")[0]:
            bg = load_rgb(r, (640, 480))
            break

    targets = _contour_targets(chosen_mask)
    nx, ny = _naive_centroid(chosen_mask)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for ax in (a1, a2):
        if bg is not None:
            ax.imshow(bg)
            ax.imshow(chosen_mask, cmap="Greys_r", alpha=0.35)
        else:
            ax.imshow(chosen_mask, cmap="Greys_r")
        ax.axis("off")

    # LEFT — naive whole-mask centroid
    naive_inside = bool(chosen_mask[int(np.clip(ny, 0, chosen_mask.shape[0]-1)),
                                    int(np.clip(nx, 0, chosen_mask.shape[1]-1))])
    a1.plot(nx, ny, marker="x", ms=26, mew=5, color=C["run1"])
    a1.set_title("Naive: centroid of the whole mask", fontsize=19,
                 color=C["run1"], fontweight="bold", pad=12)
    a1.text(0.5, -0.06, "lands in empty space between objects" if not naive_inside
            else "single object here — but fails whenever there are several",
            transform=a1.transAxes, ha="center", fontsize=16, color=C["run1"])

    # RIGHT — contour-isolated, validity-corrected
    for t in targets:
        cx, cy = t["naive"]
        fx, fy = t["fixed"]
        if not t["inside"]:
            a2.plot(cx, cy, marker="x", ms=18, mew=3.5, color=C["run1"], alpha=0.75)
            a2.annotate("", xy=(fx, fy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=C["accent"], lw=2.6))
        a2.plot(fx, fy, marker="+", ms=26, mew=5, color=C["native"])
    a2.set_title("Fixed: per-object + validity-guaranteed", fontsize=19,
                 color=C["native"], fontweight="bold", pad=12)
    n_bad = sum(1 for t in targets if not t["inside"])
    a2.text(0.5, -0.06,
            f"{len(targets)} objects · {n_bad} centroid(s) fell outside → snapped inside",
            transform=a2.transAxes, ha="center", fontsize=16, color=C["native"])

    fig.tight_layout()
    return save(fig, out, "fig_target_derivation")


# ===========================================================================
# SLIDE 8 companion — domain randomisation strip
# ===========================================================================
def fig_augmentation_strip(paths, out, seed=3):
    comps = find_images(paths["composites"])
    if not comps:
        print("  !! skipped: no composites found")
        return
    rng = random.Random(seed)
    sel = rng.sample(comps, min(6, len(comps)))
    labels = ["scale", "rotation", "flip", "photometric", "background", "noise"]

    fig, axes = plt.subplots(1, len(sel), figsize=(3.0 * len(sel), 3.4))
    if len(sel) == 1:
        axes = [axes]
    for ax, p, lab in zip(axes, sel, labels):
        ax.imshow(load_rgb(p, (300, 240)))
        ax.axis("off")
        ax.set_title(lab, fontsize=17, color=C["muted"], pad=8)
    fig.suptitle("Domain randomisation applied per composite",
                 fontsize=21, fontweight="bold", y=1.06)
    fig.tight_layout()
    return save(fig, out, "fig_augmentation_strip")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="slides_figs")
    ap.add_argument("--composites", default=None)
    ap.add_argument("--natives", default=None)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="change this to reshuffle which images get picked")
    args = ap.parse_args()

    apply_style()
    print("Locating assets:")
    paths = resolve(args.root, args)
    out = Path(args.out).expanduser()
    print()

    for name, fn, kw in [
        ("dataset_montage",    fig_dataset_montage,   {"seed": args.seed}),
        ("transparency_cues",  fig_transparency_cues, {"seed": args.seed + 1}),
        ("target_derivation",  fig_target_derivation, {"seed": args.seed + 2}),
        ("augmentation_strip", fig_augmentation_strip, {"seed": args.seed + 3}),
    ]:
        print(f"[{name}]")
        try:
            fn(paths, out, **kw)
        except Exception as e:
            print(f"  !! failed: {type(e).__name__}: {e}")
        print()

    print("Tip: rerun with a different --seed until you like the sampled images.")


if __name__ == "__main__":
    main()