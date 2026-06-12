import cv2
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset\cleargrasp-dataset-train")
BG_DIR      = Path(r"C:\Users\m_vit\Documents\MscProject\data\raw_backgrounds")
OUT_DIR     = Path(r"C:\Users\m_vit\Documents\MscProject\data\processed_dataset")
ANNOTATIONS = OUT_DIR / "annotations.json"

CATEGORIES  = [
    "cup-with-waves-train",
    "flower-bath-bomb-train",
    "heart-bath-bomb-train",
    "square-plastic-bottle-train",
    "stemless-plastic-champagne-glass-train",
]

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE        = (640, 480)   # resize everything to this
COMPOSITES_PER_IMAGE = 3       # how many augmented versions per source image
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# remaining 0.15 → test (held out)

random.seed(42)
np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────────────
def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return binary

def apply_distortions(fg, mask, bg):
    h, w = bg.shape[:2]

    # 1. Random scale (object takes of background width)
    scale = random.uniform(0.5, 0.75)    
    new_w = int(w * scale)
    new_h = int(fg.shape[0] * new_w / fg.shape[1])
    fg    = cv2.resize(fg,   (new_w, new_h))
    mask  = cv2.resize(mask, (new_w, new_h))

    # 2. Find individual disconnected plastic objects inside the pre-made mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Pick ONE random plastic piece from the cluster as our target
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

    # 3. Random position (keep object fully in frame)
    x = random.randint(0, max(0, w - new_w))
    y = random.randint(0, max(0, h - new_h))

    # 4. Brightness/contrast jitter on foreground
    alpha = random.uniform(0.6, 1.4)   # contrast
    beta  = random.randint(-40, 40)    # brightness
    fg    = np.clip(alpha * fg.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 5. Composite entire cluster onto background
    composite = bg.copy()
    roi       = composite[y:y+new_h, x:x+new_w]
    fg_mask   = mask[:new_h, :new_w, None] / 255.0
    blended   = (fg[:new_h, :new_w] * fg_mask + roi * (1 - fg_mask)).astype(np.uint8)
    composite[y:y+new_h, x:x+new_w] = blended

    # 6. Return composite + exact coordinate of the CHOSEN item
    cx = x + local_cx
    cy = y + local_cy
    return composite, (cx, cy)




def map_to_action_tokens(cx, cy, img_w, img_h):
    norm_x = cx / img_w
    norm_y = cy / img_h 
    
    # Static task values for placement base stance
    z, roll, pitch, yaw, gripper = 0.5, 0.5, 0.5, 0.5, 1.0
    continuous_actions = [norm_x, norm_y, z, roll, pitch, yaw, gripper]
    
    # Uniform 256-bin quantization
    return [int(np.clip(round(a * 255), 0, 255)) for a in continuous_actions]

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(exist_ok=True)

    backgrounds = list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.png"))
    print(f"Loaded {len(backgrounds)} backgrounds")

    annotations = {"train": [], "val": [], "test": []}
    counters    = {"train": 0, "val": 0, "test": 0}

    for cat_name in CATEGORIES:
        cat_path = BASE / cat_name
        rgb_dir  = cat_path / "rgb-imgs"
        mask_dir = cat_path / "segmentation-masks"

        rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
        random.shuffle(rgb_files)

        #rgb_files = rgb_files[:10]

        n      = len(rgb_files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = (
            [("train", f) for f in rgb_files[:n_train]] +
            [("val",   f) for f in rgb_files[n_train:n_train+n_val]] +
            [("test",  f) for f in rgb_files[n_train+n_val:]]
        )

        print(f"\n{cat_name}: {n} images → "
              f"train {n_train} / val {n_val} / test {n - n_train - n_val}")

        # Inside main(), update the generation block:
        for split, rgb_path in tqdm(splits, desc=cat_name):
            # Handles ClearGrasp naming convention (swapping '-rgb' suffix for '-segmentation-mask')
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
                # Background selected inside loop for maximum visual diversity
                bg_path = random.choice(backgrounds)
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
                    "image":        str(out_path.relative_to(OUT_DIR)),
                    "category":     cat_name,
                    "source_file":  rgb_path.name,
                    "click_2d":     [cx, cy],
                    "action_tokens": action_tokens,
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