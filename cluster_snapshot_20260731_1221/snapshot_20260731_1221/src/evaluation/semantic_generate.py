"""
Semantic Reasoning Test — Step 1: Create Two-Object Test Scenes
===============================================================
Creates test images with TWO transparent objects from DIFFERENT
ClearGrasp categories placed on left/right halves of the frame.

For each scene we save:
  - The composite image
  - Per-object masks with known centroids
  - Category labels and spatial positions

Step 2 (semantic_evaluate.py) then runs the VLA on each scene with
prompts targeting each object alternately, measuring whether the
prediction shifts toward the named object.

Usage:
    python src/evaluation/semantic_generate.py
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

BASE = Path.home() / "MscProject" / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train"
BG_DIR = Path.home() / "MscProject" / "data" / "raw_backgrounds"
OUT_DIR = Path.home() / "MscProject" / "data" / "semantic_test"

CATEGORIES = {
    "bottle": "square-plastic-bottle-train",
    "cup":    "cup-with-waves-train",
    "glass":  "stemless-plastic-champagne-glass-train",
}

# Natural-language names for prompting
CATEGORY_PROMPTS = {
    "bottle": "transparent plastic bottle",
    "cup":    "transparent plastic cup",
    "glass":  "transparent plastic champagne glass",
}

IMG_SIZE = (224, 224)
N_PAIRS = 40       # per category combination
MIN_AREA = 80

random.seed(123)
np.random.seed(123)


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, b = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    return b


def find_mask_path(rgb_path, mask_dir):
    """Handle ClearGrasp naming: 000000-rgb.jpg -> 000000-segmentation-mask.png"""
    stem = rgb_path.stem
    # Try the standard convention
    ms = stem.replace("-rgb", "-segmentation-mask") if "-rgb" in stem else stem
    mp = mask_dir / f"{ms}.png"
    if mp.exists():
        return mp
    # Fallback: try without suffix change
    mp2 = mask_dir / f"{stem}.png"
    if mp2.exists():
        return mp2
    return None


def safe_centroid(mask):
    """Return (cx, cy, area) guaranteed inside mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) < 0:
        filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        cx, cy = int(cx), int(cy)
    return cx, cy, float(cv2.contourArea(c))


def place_on_side(fg, mask, h, w, side):
    """
    Resize object to 35-55% of frame width (bigger than before to ensure
    visibility at 224x224), place on specified half.
    Returns (full_frame_mask, cx, cy, area) or None.
    """
    scale = random.uniform(0.35, 0.55)
    new_w = max(20, int(w * scale))
    new_h = max(20, int(fg.shape[0] * new_w / fg.shape[1]))
    new_h = min(new_h, h - 10)
    new_w = min(new_w, w // 2 - 10)  # must fit in half

    fg_r = cv2.resize(fg, (new_w, new_h))
    mask_r = cv2.resize(mask, (new_w, new_h))
    _, mask_r = cv2.threshold(mask_r, 127, 255, cv2.THRESH_BINARY)

    half = w // 2
    if side == "left":
        x = random.randint(2, max(2, half - new_w - 2))
    else:
        x = random.randint(half + 2, max(half + 2, w - new_w - 2))
    y = random.randint(2, max(2, h - new_h - 2))

    hc = min(new_h, h - y)
    wc = min(new_w, w - x)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y:y+hc, x:x+wc] = mask_r[:hc, :wc]

    tgt = safe_centroid(full_mask)
    if tgt is None:
        return None

    return fg_r, mask_r, full_mask, x, y, hc, wc, tgt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)
    (OUT_DIR / "masks").mkdir(exist_ok=True)

    backgrounds = list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.png"))
    print(f"Loaded {len(backgrounds)} backgrounds")

    # Load sources per category
    sources = {}
    for short, folder in CATEGORIES.items():
        rgb_dir = BASE / folder / "rgb-imgs"
        mask_dir = BASE / folder / "segmentation-masks"
        if not rgb_dir.exists():
            print(f"[SKIP] {folder} not found — transfer it first")
            continue
        pairs = []
        for rp in sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png")):
            mp = find_mask_path(rp, mask_dir)
            if mp:
                pairs.append((rp, mp))
        random.shuffle(pairs)
        sources[short] = pairs[:150]  # cap to keep it fast
        print(f"  {short}: {len(sources[short])} source pairs")

    cats = list(sources.keys())
    if len(cats) < 2:
        print(f"[ERROR] Need >= 2 categories, found {cats}. Transfer more data.")
        return

    records = []
    pid = 0

    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            ca, cb = cats[i], cats[j]
            print(f"\n{ca} + {cb}...")

            for _ in tqdm(range(N_PAIRS), desc=f"{ca}+{cb}"):
                bg = cv2.imread(str(random.choice(backgrounds)))
                if bg is None:
                    continue
                bg = cv2.resize(bg, IMG_SIZE)
                h, w = bg.shape[:2]

                sa = random.choice(sources[ca])
                sb = random.choice(sources[cb])
                fg_a, mask_a = cv2.imread(str(sa[0])), load_mask(sa[1])
                fg_b, mask_b = cv2.imread(str(sb[0])), load_mask(sb[1])
                if any(x is None for x in [fg_a, mask_a, fg_b, mask_b]):
                    continue

                # Random left/right assignment
                if random.random() > 0.5:
                    side_a, side_b = "left", "right"
                else:
                    side_a, side_b = "right", "left"

                ra = place_on_side(fg_a, mask_a, h, w, side_a)
                rb = place_on_side(fg_b, mask_b, h, w, side_b)
                if ra is None or rb is None:
                    continue

                fg_ar, mask_ar, fm_a, xa, ya, ha, wa, tgt_a = ra
                fg_br, mask_br, fm_b, xb, yb, hb, wb, tgt_b = rb

                # Check minimal overlap
                if np.sum(cv2.bitwise_and(fm_a, fm_b) > 0) > 30:
                    continue

                # Composite
                comp = bg.copy()
                sa_soft = cv2.GaussianBlur(mask_ar.astype(np.float32), (5, 5), 0)
                mr_a = sa_soft[:ha, :wa, None] / 255.0
                comp[ya:ya+ha, xa:xa+wa] = (
                    fg_ar[:ha, :wa] * mr_a + comp[ya:ya+ha, xa:xa+wa] * (1 - mr_a)
                ).astype(np.uint8)

                sb_soft = cv2.GaussianBlur(mask_br.astype(np.float32), (5, 5), 0)
                mr_b = sb_soft[:hb, :wb, None] / 255.0
                comp[yb:yb+hb, xb:xb+wb] = (
                    fg_br[:hb, :wb] * mr_b + comp[yb:yb+hb, xb:xb+wb] * (1 - mr_b)
                ).astype(np.uint8)

                noise = np.random.normal(0, 2, comp.shape).astype(np.float32)
                comp = np.clip(comp + noise, 0, 255).astype(np.uint8)

                name = f"pair_{pid:04d}_{ca}_{cb}.png"
                cv2.imwrite(str(OUT_DIR / "images" / name), comp)
                cv2.imwrite(str(OUT_DIR / "masks" / f"a_{name}"), fm_a)
                cv2.imwrite(str(OUT_DIR / "masks" / f"b_{name}"), fm_b)

                records.append({
                    "image": f"images/{name}",
                    "mask_a": f"masks/a_{name}",
                    "mask_b": f"masks/b_{name}",
                    "cat_a": ca,
                    "cat_b": cb,
                    "prompt_a": CATEGORY_PROMPTS[ca],
                    "prompt_b": CATEGORY_PROMPTS[cb],
                    "side_a": side_a,
                    "side_b": side_b,
                    "target_a": [tgt_a[0], tgt_a[1]],
                    "target_b": [tgt_b[0], tgt_b[1]],
                })
                pid += 1

    with open(OUT_DIR / "test_pairs.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[DONE] {len(records)} two-object test scenes saved to {OUT_DIR}")


if __name__ == "__main__":
    main()