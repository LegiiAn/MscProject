"""
ReSort-IT dataset generator — v4 (Run 2)
========================================
Root-cause fixes over the Run-1 generator:

  FIX 1 — Label validity. Targets are GUARANTEED to lie on object pixels.
          Centroids are checked with cv2.pointPolygonTest; if the centroid
          falls outside the contour (crescent / crushed shapes), the target
          falls back to the contour's most-interior point via a distance
          transform. Validity is 100% by construction.

  FIX 2 — Native resolution. Samples are generated at 224x224, OpenVLA's
          encoder input size, removing the destructive 640->224 double
          downsampling of Run 1. At 224px, one action bin = 0.875px.

  FIX 3 — Real-data mixing. Each source image contributes 2 composites AND
          2 lightly-augmented REAL ClearGrasp frames (which contain true
          refraction, shadows and specular cues that composites lack),
          giving a 50/50 composite/real training mix.

Also saved per sample: full and target object masks (for exact grasp
evaluation), sample_type ("composite"|"real"), and target_area_px
(for per-size accuracy breakdown).
"""

import cv2
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

# ── Cluster Paths ──────────────────────────────────────────────────────────
BASE        = Path.home() / "MscProject" / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train"
BG_DIR      = Path.home() / "MscProject" / "data" / "raw_backgrounds"
OUT_DIR     = Path.home() / "MscProject" / "data" / "processed_dataset"      # Run-1 dataset must be moved aside first!
ANNOTATIONS = OUT_DIR / "annotations.json"

CATEGORIES  = [
    "square-plastic-bottle-train",
]

CATEGORY_INSTRUCTIONS = {
    "cup-with-waves-train":                    "Pick up the transparent plastic cup.",
    "flower-bath-bomb-train":                  "Pick up the transparent flower-shaped plastic object.",
    "heart-bath-bomb-train":                   "Pick up the transparent heart-shaped plastic object.",
    "square-plastic-bottle-train":             "Pick up the transparent plastic bottle.",
    "stemless-plastic-champagne-glass-train":  "Pick up the transparent plastic champagne glass.",
}

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE               = (224, 224)   # FIX 2: OpenVLA-native (width, height)
COMPOSITES_PER_IMAGE   = 2
REAL_VARIANTS_PER_IMAGE = 2           # FIX 3: 50/50 real/composite
TRAIN_RATIO            = 0.70
VAL_RATIO              = 0.15
MIN_TARGET_AREA_PX     = 25           # skip degenerate specks at 224x224

random.seed(42)
np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────────────
def load_binary_mask(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, b = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    return b


def safe_target_point(contour, canvas_shape):
    """
    FIX 1: return a target point GUARANTEED inside the contour.
    Try the centroid first; if it falls outside (concave/crescent shapes),
    fall back to the most-interior point of the filled contour, found as
    the argmax of the L2 distance transform.
    Returns (x, y) or None if the contour is degenerate.
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0:
        return cx, cy

    # Centroid outside the shape -> most-interior point
    h, w = canvas_shape
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    if filled[y, x] == 0:
        return None
    return int(x), int(y)


def pick_target(mask):
    """
    Select a random object contour from a binary mask and derive a
    validity-guaranteed target point plus the target-only filled mask.
    Returns (cx, cy, target_mask, area) or None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_TARGET_AREA_PX]
    if not contours:
        return None
    target_contour = random.choice(contours)

    pt = safe_target_point(target_contour, mask.shape)
    if pt is None:
        return None
    cx, cy = pt

    tgt_mask = np.zeros_like(mask)
    cv2.drawContours(tgt_mask, [target_contour], -1, 255, thickness=-1)

    # Belt-and-braces: assert validity by construction
    if tgt_mask[cy, cx] == 0:
        return None

    return cx, cy, tgt_mask, float(cv2.contourArea(target_contour))


def photometric_jitter(img):
    """Contrast/brightness + hue/saturation jitter (shared by both paths)."""
    a = random.uniform(0.6, 1.4)
    b = random.randint(-40, 40)
    img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_noise(img):
    noise = np.random.normal(0, random.uniform(1, 5), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ── Sample builders ────────────────────────────────────────────────────────
def make_composite(fg, mask, bg):
    """
    Composite path (as Run 1, at native resolution, with FIX 1 targets).
    Returns (image, cx, cy, full_mask, tgt_mask, area) or None.
    """
    h, w = bg.shape[:2]

    scale = random.uniform(0.25, 0.75)
    new_w = max(10, int(w * scale))
    new_h = max(10, int(fg.shape[0] * new_w / fg.shape[1]))
    new_h = min(new_h, h)  # keep within frame vertically
    fg   = cv2.resize(fg,   (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))

    if random.random() > 0.5:
        fg, mask = cv2.flip(fg, 1), cv2.flip(mask, 1)

    angle = random.uniform(-15, 15)
    rot = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    fg   = cv2.warpAffine(fg,   rot, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    picked = pick_target(mask)
    if picked is None:
        return None
    lcx, lcy, local_tgt, area = picked

    x = random.randint(0, max(0, w - new_w))
    y = random.randint(0, max(0, h - new_h))

    fg = photometric_jitter(fg)

    soft = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)
    composite = bg.copy()
    roi = composite[y:y+new_h, x:x+new_w]
    mr = soft[:new_h, :new_w, None] / 255.0
    composite[y:y+new_h, x:x+new_w] = (
        fg[:new_h, :new_w] * mr + roi * (1 - mr)
    ).astype(np.uint8)
    composite = add_noise(composite)

    full_canvas = np.zeros((h, w), dtype=np.uint8)
    full_canvas[y:y+new_h, x:x+new_w] = mask[:new_h, :new_w]
    tgt_canvas = np.zeros((h, w), dtype=np.uint8)
    tgt_canvas[y:y+new_h, x:x+new_w] = local_tgt[:new_h, :new_w]

    return composite, x + lcx, y + lcy, full_canvas, tgt_canvas, area


def make_real(rgb, mask):
    """
    FIX 3: real-ClearGrasp path. The native render (true refraction,
    shadows, speculars) is resized to working resolution, lightly
    augmented (flip + photometric + noise; NO rotation, to preserve
    authentic geometry), and labelled from its own segmentation mask
    with FIX-1 target derivation.
    """
    img = cv2.resize(rgb, IMG_SIZE)
    m = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    if random.random() > 0.5:
        img, m = cv2.flip(img, 1), cv2.flip(m, 1)

    picked = pick_target(m)
    if picked is None:
        return None
    cx, cy, tgt_mask, area = picked

    img = add_noise(photometric_jitter(img))
    return img, cx, cy, m, tgt_mask, area


def map_to_action_tokens(cx, cy, img_w, img_h):
    norm = [cx / img_w, cy / img_h, 0.5, 0.5, 0.5, 0.5, 1.0]
    return [int(np.clip(round(a * 255), 0, 255)) for a in norm]


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(exist_ok=True)
        (OUT_DIR / split / "masks").mkdir(exist_ok=True)

    backgrounds = list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.png"))
    random.shuffle(backgrounds)
    print(f"Loaded {len(backgrounds)} backgrounds")

    n_bg = len(backgrounds)
    bg_pools = {
        "train": backgrounds[: int(n_bg * TRAIN_RATIO)],
        "val":   backgrounds[int(n_bg * TRAIN_RATIO): int(n_bg * (TRAIN_RATIO + VAL_RATIO))],
        "test":  backgrounds[int(n_bg * (TRAIN_RATIO + VAL_RATIO)):],
    }
    for k, p in bg_pools.items():
        if not p:
            bg_pools[k] = bg_pools["train"]
    print(f"Background split: train {len(bg_pools['train'])} / "
          f"val {len(bg_pools['val'])} / test {len(bg_pools['test'])}")

    annotations = {"train": [], "val": [], "test": []}
    counters = {"train": {"composite": 0, "real": 0},
                "val":   {"composite": 0, "real": 0},
                "test":  {"composite": 0, "real": 0}}
    validity_failures = 0  # should stay 0 by construction

    for cat_name in CATEGORIES:
        cat_path = BASE / cat_name
        rgb_dir, mask_dir = cat_path / "rgb-imgs", cat_path / "segmentation-masks"

        rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        random.shuffle(rgb_files)

        n = len(rgb_files)
        n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
        splits = (
            [("train", f) for f in rgb_files[:n_train]] +
            [("val",   f) for f in rgb_files[n_train:n_train + n_val]] +
            [("test",  f) for f in rgb_files[n_train + n_val:]]
        )

        instruction = CATEGORY_INSTRUCTIONS.get(cat_name, "Pick up the transparent plastic object.")
        print(f"\n{cat_name}: {n} images -> train {n_train} / val {n_val} / test {n - n_train - n_val}")

        for split, rgb_path in tqdm(splits, desc=cat_name):
            stem = rgb_path.stem
            mask_stem = stem.replace("-rgb", "-segmentation-mask") if "-rgb" in stem else stem
            mask_path = mask_dir / f"{mask_stem}.png"
            if not mask_path.exists():
                continue

            fg = cv2.imread(str(rgb_path))
            mask = load_binary_mask(mask_path)
            if fg is None or mask is None:
                continue

            jobs = ([("composite", i) for i in range(COMPOSITES_PER_IMAGE)] +
                    [("real", i) for i in range(REAL_VARIANTS_PER_IMAGE)])

            for sample_type, i in jobs:
                if sample_type == "composite":
                    bg_path = random.choice(bg_pools[split])
                    raw_bg = cv2.imread(str(bg_path))
                    if raw_bg is None:
                        continue
                    bg = cv2.resize(raw_bg, IMG_SIZE)
                    out = make_composite(fg, mask, bg)
                    bg_name = bg_path.name
                else:
                    out = make_real(fg, mask)
                    bg_name = "native"

                if out is None:
                    continue
                img, cx, cy, full_m, tgt_m, area = out

                # runtime validity assertion (FIX 1 guarantee)
                if tgt_m[int(np.clip(cy, 0, tgt_m.shape[0]-1)),
                         int(np.clip(cx, 0, tgt_m.shape[1]-1))] == 0:
                    validity_failures += 1
                    continue

                tokens = map_to_action_tokens(cx, cy, IMG_SIZE[0], IMG_SIZE[1])

                out_name = f"{cat_name}_{stem}_{sample_type}{i}.png"
                cv2.imwrite(str(OUT_DIR / split / out_name), img)
                cv2.imwrite(str(OUT_DIR / split / "masks" / f"full_{out_name}"), full_m)
                cv2.imwrite(str(OUT_DIR / split / "masks" / f"target_{out_name}"), tgt_m)

                annotations[split].append({
                    "image":         f"{split}/{out_name}",
                    "category":      cat_name,
                    "instruction":   instruction,
                    "sample_type":   sample_type,
                    "source_file":   rgb_path.name,
                    "background":    bg_name,
                    "click_2d":      [int(cx), int(cy)],
                    "action_tokens": tokens,
                    "target_area_px": area,
                    "mask_full":     f"{split}/masks/full_{out_name}",
                    "mask_target":   f"{split}/masks/target_{out_name}",
                })
                counters[split][sample_type] += 1

    with open(ANNOTATIONS, "w") as f:
        json.dump(annotations, f, indent=2)

    print("\n── Done ──────────────────────────────────────")
    for split, c in counters.items():
        print(f"  {split:5s}: {c['composite']:,} composite + {c['real']:,} real "
              f"= {c['composite'] + c['real']:,}")
    print(f"  Validity failures (must be 0): {validity_failures}")
    print(f"  Annotations saved to {ANNOTATIONS}")


if __name__ == "__main__":
    main()