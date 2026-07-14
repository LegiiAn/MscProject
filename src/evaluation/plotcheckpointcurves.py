"""
Generate the checkpoint-trajectory figure from vla_eval_results.json.
Replaces the (never-generated) vla_evaluation_detailed.pdf.

Run locally or on cluster wherever vla_eval_results.json lives:
    python plot_checkpoint_curves.py
Output: vla_checkpoint_curves.pdf
"""
import json
import re
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def step_of(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else -1


def main():
    path = Path("vla_eval_results.json")
    if not path.exists():
        print("[ERROR] vla_eval_results.json not found in current directory.")
        return

    with open(path) as f:
        results = json.load(f)

    # Keep only step checkpoints with clean parsing (drop lora_best duplicate
    # and any entry with >5 parse failures, whose X means are contaminated).
    rows = []
    for name, m in results.items():
        s = step_of(name)
        if s < 0 or "best" in name:
            continue
        if m.get("parse_failures", 0) > 5:
            continue
        rows.append((s, m))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("[ERROR] No usable checkpoint entries.")
        return

    steps = [r[0] for r in rows]
    l1 = [r[1]["spatial_l1_mean"] for r in rows]
    l1_med = [r[1].get("spatial_l1_median", np.nan) for r in rows]
    xe = [r[1]["x_l1_mean"] for r in rows]
    ye = [r[1]["y_l1_mean"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    axes[0].plot(steps, l1, marker="o", linewidth=2, color="#d62728", label="Mean")
    axes[0].plot(steps, l1_med, marker="s", linewidth=1.5, color="#9467bd",
                 alpha=0.8, label="Median")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Spatial L1 error (bins)")
    axes[0].set_title("Spatial error across checkpoints", fontsize=11, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, xe, marker="^", linewidth=2, color="#1f77b4", label="X error")
    axes[1].plot(steps, ye, marker="v", linewidth=2, color="#ff7f0e", label="Y error")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Per-axis L1 error (bins)")
    axes[1].set_title("Per-axis coordinate error", fontsize=11, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vla_checkpoint_curves.pdf", format="pdf", bbox_inches="tight")
    print(f"[SAVED] vla_checkpoint_curves.pdf ({len(steps)} checkpoints plotted)")


if __name__ == "__main__":
    main()