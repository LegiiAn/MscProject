"""
Coordinate Distribution Analysis
================================
Plots the spatial distribution of grasp targets (click_2d) across the
generated dataset. Reveals whether ReSort-IT placement introduces a
positional bias that the VLA could exploit as a shortcut prior.

If the X or Y distribution is heavily skewed, the model can achieve
low error by learning "objects usually appear here" rather than actually
localizing from pixels. A roughly uniform distribution is what you want.

Reads: data/processed_dataset/annotations.json
No GPU needed. Runs locally in seconds.

Usage:
    python src/evaluation/coordinate_distribution.py
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Try common locations
    candidates = [
        Path("data/processed_dataset/annotations.json"),
        Path.home() / "MscProject" / "data" / "processed_dataset" / "annotations.json",
    ]
    ann_path = next((p for p in candidates if p.exists()), None)
    if ann_path is None:
        print("[ERROR] annotations.json not found in expected locations.")
        return

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    IMG_W, IMG_H = 640, 480

    # Collect coordinates per split
    splits = ["train", "val", "test"]
    coords = {s: {"x": [], "y": []} for s in splits}

    for split in splits:
        for record in annotations.get(split, []):
            click = record.get("click_2d")
            if click and len(click) == 2:
                coords[split]["x"].append(click[0])
                coords[split]["y"].append(click[1])

    # ── Statistics ───────────────────────────────────────────────────────
    print("=" * 55)
    print("COORDINATE DISTRIBUTION ANALYSIS")
    print("=" * 55)
    for split in splits:
        xs = np.array(coords[split]["x"])
        ys = np.array(coords[split]["y"])
        if len(xs) == 0:
            continue
        print(f"\n{split.upper()} ({len(xs)} samples):")
        print(f"  X: mean={np.mean(xs):.0f} (center={IMG_W/2:.0f}), "
              f"std={np.std(xs):.0f}, range=[{np.min(xs)}, {np.max(xs)}]")
        print(f"  Y: mean={np.mean(ys):.0f} (center={IMG_H/2:.0f}), "
              f"std={np.std(ys):.0f}, range=[{np.min(ys)}, {np.max(ys)}]")

        # Uniformity check: coefficient of variation of a histogram
        x_hist, _ = np.histogram(xs, bins=10, range=(0, IMG_W))
        y_hist, _ = np.histogram(ys, bins=10, range=(0, IMG_H))
        x_uniformity = np.std(x_hist) / (np.mean(x_hist) + 1e-9)
        y_uniformity = np.std(y_hist) / (np.mean(y_hist) + 1e-9)
        print(f"  X non-uniformity: {x_uniformity:.2f} (0=perfectly uniform)")
        print(f"  Y non-uniformity: {y_uniformity:.2f}")

    print("=" * 55)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # Use train split for main analysis
    xs = np.array(coords["train"]["x"])
    ys = np.array(coords["train"]["y"])

    # Panel 1: X histogram
    axes[0, 0].hist(xs, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(IMG_W / 2, color="red", linestyle="--", label="Image center")
    axes[0, 0].set_xlabel("X coordinate (pixels)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Horizontal Target Distribution", fontsize=11, fontweight="bold")
    axes[0, 0].legend()

    # Panel 2: Y histogram
    axes[0, 1].hist(ys, bins=40, color="#ff7f0e", alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(IMG_H / 2, color="red", linestyle="--", label="Image center")
    axes[0, 1].set_xlabel("Y coordinate (pixels)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Vertical Target Distribution", fontsize=11, fontweight="bold")
    axes[0, 1].legend()

    # Panel 3: 2D heatmap of target positions
    hist2d, xedges, yedges = np.histogram2d(xs, ys, bins=32, range=[[0, IMG_W], [0, IMG_H]])
    im = axes[1, 0].imshow(hist2d.T, origin="upper", extent=[0, IMG_W, IMG_H, 0],
                           cmap="hot", aspect="auto")
    axes[1, 0].set_xlabel("X (pixels)")
    axes[1, 0].set_ylabel("Y (pixels)")
    axes[1, 0].set_title("2D Target Density Heatmap", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[1, 0], label="Count")

    # Panel 4: scatter
    axes[1, 1].scatter(xs, ys, alpha=0.2, s=8, color="#2ca02c")
    axes[1, 1].set_xlim(0, IMG_W)
    axes[1, 1].set_ylim(IMG_H, 0)
    axes[1, 1].set_xlabel("X (pixels)")
    axes[1, 1].set_ylabel("Y (pixels)")
    axes[1, 1].set_title("Target Scatter", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("coordinate_distribution.pdf", format="pdf", bbox_inches="tight")
    print("[SAVED] coordinate_distribution.pdf")


if __name__ == "__main__":
    main()
