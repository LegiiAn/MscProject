"""
Pre-training label audit — run BEFORE tune_vla.py.
Checks that 100% of ground-truth targets lie inside their target mask,
and prints the dataset composition (real vs composite, size distribution).

    python verify_labels.py
"""
import json
import numpy as np
import cv2
from pathlib import Path
PROJECT_ROOT = Path(r"C:\Users\m_vit\Documents\MscProject")

DATA_DIR = PROJECT_ROOT / "data" / "processed_dataset"


def main():
    with open(DATA_DIR / "annotations.json") as f:
        ann = json.load(f)

    total_bad = 0
    for split in ["train", "val", "test"]:
        records = ann.get(split, [])
        n = len(records)
        bad, missing = 0, 0
        types = {"composite": 0, "real": 0}
        areas = []

        for r in records:
            types[r.get("sample_type", "composite")] = types.get(r.get("sample_type", "composite"), 0) + 1
            areas.append(r.get("target_area_px", 0))
            tgt_path = DATA_DIR / r["mask_target"]
            m = cv2.imread(str(tgt_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                missing += 1
                continue
            cx, cy = r["click_2d"]
            cx = int(np.clip(cx, 0, m.shape[1] - 1))
            cy = int(np.clip(cy, 0, m.shape[0] - 1))
            if m[cy, cx] == 0:
                bad += 1

        validity = 100.0 * (n - bad - missing) / max(n, 1)
        areas = np.array(areas)
        print(f"\n[{split}] {n:,} samples "
              f"({types.get('composite',0):,} composite / {types.get('real',0):,} real)")
        print(f"  Target-on-mask validity: {validity:.2f}%  "
              f"(bad: {bad}, missing masks: {missing})")
        if len(areas):
            print(f"  Target area px  — min {areas.min():.0f} / "
                  f"median {np.median(areas):.0f} / max {areas.max():.0f}")
        total_bad += bad + missing

    print("\n" + "=" * 50)
    if total_bad == 0:
        print("PASS — 100% label validity. Safe to train.")
    else:
        print(f"FAIL — {total_bad} invalid/missing labels. DO NOT TRAIN.")
    print("=" * 50)


if __name__ == "__main__":
    main()