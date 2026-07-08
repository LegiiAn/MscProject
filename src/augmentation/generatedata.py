import cv2
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
#BASE        = Path(r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset\cleargrasp-dataset-train")
#BG_DIR      = Path(r"C:\Users\m_vit\Documents\MscProject\data\raw_backgrounds")
#OUT_DIR     = Path(r"C:\Users\m_vit\Documents\MscProject\data\processed_dataset")
# ── Cluster Paths ──
BASE    = Path.home() / "MscProject/data/cleargrasp_dataset/cleargrasp-dataset-train"
BG_DIR  = Path.home() / "MscProject/data/raw_backgrounds"
OUT_DIR = Path.home() / "MscProject/data/processed_dataset"

ANNOTATIONS = OUT_DIR / "annotations.json"

CATEGORIES  = [
    #"cup-with-waves-train",
    #"flower-bath-bomb-train",
    #"heart-bath-bomb-train",
    "square-plastic-bottle-train",
    #"stemless-plastic-champagne-glass-train",
]

# Map category folder names to natural language instructions for VLA prompts
CATEGORY_INSTRUCTIONS = {
    "cup-with-waves-train":                    "Pick up the transparent plastic cup.",
    "flower-bath-bomb-train":                  "Pick up the transparent flower-shaped plastic object.",
    "heart-bath-bomb-train":                   "Pick up the transparent heart-shaped plastic object.",
    "square-plastic-bottle-train":             "Pick up the transparent plastic bottle.",
    "stemless-plastic-champagne-glass-train":  "Pick up the transparent plastic champagne glass.",
}

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE             = (640, 480)   # (width, height)
COMPOSITES_PER_IMAGE = 3           # augmented versions per source image
TRAIN_RATIO          = 0.70
VAL_RATIO            = 0.15
# remaining 0.15 → test

random.seed(42)
np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────────────
def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return binary


def soften_mask_edges(mask, blur_radius=5):
    """Apply Gaussian blur to mask edges to destroy sharp compositing seams.
    This prevents the model from using the hard alpha boundary as a localization shortcut."""
    # Blur the binary mask so edges become soft gradients
    softened = cv2.GaussianBlur(mask.astype(np.float32), (blur_radius, blur_radius), 0)
    return softened


def apply_distortions(fg, mask, bg):
    h, w = bg.shape[:2]

    # 1. Random scale — wider range than before for more size variation
    scale = random.uniform(0.25, 0.75)
    new_w = int(w * scale)
    new_h = int(fg.shape[0] * new_w / fg.shape[1])
    if new_w < 10 or new_h < 10:
        new_w, new_h = max(new_w, 10), max(new_h, 10)
    fg   = cv2.resize(fg,   (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))

    # 2. Random horizontal flip (50% chance)
    if random.random() > 0.5:
        fg   = cv2.flip(fg, 1)
        mask = cv2.flip(mask, 1)

    # 3. Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    center = (new_w // 2, new_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    fg   = cv2.warpAffine(fg,   rot_mat, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, rot_mat, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 4. Re-threshold mask after rotation interpolation
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 5. Find individual disconnected objects inside the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Pick ONE random piece as the grasp target
        target_contour = random.choice(contours)
        M = cv2.moments(target_contour)
        if M["m00"] != 0:
            local_cx = int(M["m10"] / M["m00"])
            local_cy = int(M["m01"] / M["m00"])
        else:
            local_cx = new_w // 2
            local_cy = new_h // 2
    else:
        local_cx = new_w // 2
        local_cy = new_h // 2

    # 6. Random position (keep object fully in frame)
    x = random.randint(0, max(0, w - new_w))
    y = random.randint(0, max(0, h - new_h))

    # 7. Photometric augmentation on foreground
    #    Brightness + contrast (existing)
    alpha_contrast = random.uniform(0.6, 1.4)
    beta_brightness = random.randint(-40, 40)
    fg = np.clip(alpha_contrast * fg.astype(np.float32) + beta_brightness, 0, 255).astype(np.uint8)

    #    Hue/saturation jitter via HSV space
    hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180   # hue shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)  # sat scale
    fg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. Soften mask edges to prevent hard-seam shortcut learning
    soft_mask = soften_mask_edges(mask, blur_radius=7)

    # 9. Composite onto background using softened mask
    composite = bg.copy()
    roi = composite[y:y+new_h, x:x+new_w]
    fg_region = fg[:new_h, :new_w]
    mask_region = soft_mask[:new_h, :new_w, None] / 255.0
    blended = (fg_region * mask_region + roi * (1 - mask_region)).astype(np.uint8)
    composite[y:y+new_h, x:x+new_w] = blended

    # 10. Add light Gaussian noise to the full composite
    noise = np.random.normal(0, random.uniform(1, 5), composite.shape).astype(np.float32)
    composite = np.clip(composite.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 11. Return composite + exact coordinate of the CHOSEN item in output image space
    cx = x + local_cx
    cy = y + local_cy
    return composite, (cx, cy)


def map_to_action_tokens(cx, cy, img_w, img_h):
    """Quantize 2D target into 256-bin action token list.
    Returns list of 7 ints: [x_bin, y_bin, z, roll, pitch, yaw, gripper]
    where z/roll/pitch/yaw/gripper are fixed task-constant values."""
    norm_x = cx / img_w
    norm_y = cy / img_h

    # Static values for the non-spatial dimensions
    z, roll, pitch, yaw, gripper = 0.5, 0.5, 0.5, 0.5, 1.0
    continuous_actions = [norm_x, norm_y, z, roll, pitch, yaw, gripper]

    return [int(np.clip(round(a * 255), 0, 255)) for a in continuous_actions]


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(exist_ok=True)

    backgrounds = list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.png"))
    random.shuffle(backgrounds)
    print(f"Loaded {len(backgrounds)} backgrounds")

    # ── Split backgrounds by identity to prevent leakage ──
    n_bg = len(backgrounds)
    n_bg_train = int(n_bg * TRAIN_RATIO)
    n_bg_val   = int(n_bg * VAL_RATIO)
    bg_train = backgrounds[:n_bg_train]
    bg_val   = backgrounds[n_bg_train:n_bg_train + n_bg_val]
    bg_test  = backgrounds[n_bg_train + n_bg_val:]

    bg_pools = {"train": bg_train, "val": bg_val, "test": bg_test}
    print(f"Background split: train {len(bg_train)} / val {len(bg_val)} / test {len(bg_test)}")

    # Ensure every pool has at least 1 background
    for split_name, pool in bg_pools.items():
        if len(pool) == 0:
            print(f"[WARN] {split_name} background pool is empty, borrowing from train pool")
            bg_pools[split_name] = bg_train

    annotations = {"train": [], "val": [], "test": []}
    counters    = {"train": 0, "val": 0, "test": 0}

    for cat_name in CATEGORIES:
        cat_path = BASE / cat_name
        rgb_dir  = cat_path / "rgb-imgs"
        mask_dir = cat_path / "segmentation-masks"

        rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        random.shuffle(rgb_files)

        n       = len(rgb_files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = (
            [("train", f) for f in rgb_files[:n_train]] +
            [("val",   f) for f in rgb_files[n_train:n_train+n_val]] +
            [("test",  f) for f in rgb_files[n_train+n_val:]]
        )

        instruction = CATEGORY_INSTRUCTIONS.get(cat_name, "Pick up the transparent plastic object.")
        print(f"\n{cat_name}: {n} images -> train {n_train} / val {n_val} / test {n - n_train - n_val}")

        for split, rgb_path in tqdm(splits, desc=cat_name):
            # Handle ClearGrasp naming convention
            if "-rgb" in rgb_path.stem:
                mask_stem = rgb_path.stem.replace("-rgb", "-segmentation-mask")
            else:
                mask_stem = rgb_path.stem

            mask_path = mask_dir / f"{mask_stem}.png"
            if not mask_path.exists():
                continue

            fg = cv2.imread(str(rgb_path))
            mask = load_mask(mask_path)
            if fg is None or mask is None:
                continue

            for i in range(COMPOSITES_PER_IMAGE):
                # Select background from the CORRECT split-specific pool
                bg_path = random.choice(bg_pools[split])
                raw_bg = cv2.imread(str(bg_path))
                if raw_bg is None:
                    continue
                bg = cv2.resize(raw_bg, IMG_SIZE)

                composite, (cx, cy) = apply_distortions(fg, mask, bg)
                action_tokens = map_to_action_tokens(cx, cy, IMG_SIZE[0], IMG_SIZE[1])

                out_name = f"{cat_name}_{rgb_path.stem}_aug{i}.png"
                out_path = OUT_DIR / split / out_name
                cv2.imwrite(str(out_path), composite)

                annotations[split].append({
                    "image":          str(out_path.relative_to(OUT_DIR)),
                    "category":       cat_name,
                    "instruction":    instruction,
                    "source_file":    rgb_path.name,
                    "background":     bg_path.name,
                    "click_2d":       [cx, cy],
                    "action_tokens":  action_tokens,
                })
                counters[split] += 1

    with open(ANNOTATIONS, "w") as f:
        json.dump(annotations, f, indent=2)

    print("\n── Done ──────────────────────────────────────")
    for split, count in counters.items():
        print(f"  {split:5s}: {count:,} composites")
    print(f"  Annotations saved to {ANNOTATIONS}")


if __name__ == "__main__":
    main()