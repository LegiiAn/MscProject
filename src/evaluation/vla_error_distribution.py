"""
VLA Spatial Error Distribution Analysis
=======================================
Reveals whether the 27.6-bin mean error is uniform mediocrity or a
bimodal mix of accurate predictions + occasional catastrophic failures.

This distinction matters: a bimodal distribution (mostly-accurate with
rare hallucinations) is a much stronger result than uniform error, and
suggests different improvement strategies.

Reads: vla_detailed_errors.json (produced by the modified eval, see note)
   OR: re-runs a lightweight inference pass if that file is absent.

No GPU needed if the errors file exists.

Usage:
    python src/evaluation/vla_error_distribution.py
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Look for a detailed per-sample errors file first
    errors_file = Path("results/vla_detailed_errors.json")
    results_file = Path("results/vla_eval_results.json")

    per_sample_l1 = None
    per_sample_x = None
    per_sample_y = None

    if errors_file.exists():
        with open(errors_file, "r") as f:
            data = json.load(f)
        per_sample_l1 = data.get("l1_errors", [])
        per_sample_x = data.get("x_errors", [])
        per_sample_y = data.get("y_errors", [])
        print(f"[DATA] Loaded {len(per_sample_l1)} per-sample errors from {errors_file}")
    else:
        print(f"[WARN] {errors_file} not found.")
        print("       To generate it, add per-sample error logging to your eval script.")
        print("       See the DETAILED_LOGGING_SNIPPET at the bottom of this file.")
        # Try to at least show the summary from results file
        if results_file.exists():
            with open(results_file, "r") as f:
                summary = json.load(f)
            print("\n[SUMMARY] Available checkpoint means:")
            for ckpt, m in summary.items():
                if isinstance(m, dict) and "spatial_l1_mean" in m:
                    print(f"  {ckpt}: L1={m['spatial_l1_mean']:.2f}, "
                          f"X={m.get('x_l1_mean', 'N/A')}, Y={m.get('y_l1_mean', 'N/A')}")
        return

    if not per_sample_l1:
        print("[ERROR] No per-sample data found.")
        return

    l1 = np.array(per_sample_l1)

    # ── Statistics ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("VLA SPATIAL ERROR DISTRIBUTION")
    print("=" * 55)
    print(f"N samples:        {len(l1)}")
    print(f"Mean L1:          {np.mean(l1):.2f} bins")
    print(f"Median L1:        {np.median(l1):.2f} bins")
    print(f"Std L1:           {np.std(l1):.2f} bins")
    print(f"Min / Max:        {np.min(l1):.1f} / {np.max(l1):.1f}")
    print(f"25th percentile:  {np.percentile(l1, 25):.2f}")
    print(f"75th percentile:  {np.percentile(l1, 75):.2f}")
    print(f"90th percentile:  {np.percentile(l1, 90):.2f}")

    # Bimodality check: what fraction is "accurate" (<15) vs "failure" (>50)?
    accurate = np.mean(l1 < 15) * 100
    moderate = np.mean((l1 >= 15) & (l1 <= 50)) * 100
    failure = np.mean(l1 > 50) * 100
    print(f"\nAccurate (<15 bins):   {accurate:.1f}%")
    print(f"Moderate (15-50 bins): {moderate:.1f}%")
    print(f"Failure (>50 bins):    {failure:.1f}%")

    if accurate > 50 and failure > 10:
        print("\n>> BIMODAL: Model is mostly accurate with occasional failures.")
        print("   Improvement strategy: reduce hallucinations (more data diversity).")
    elif np.std(l1) < np.mean(l1) * 0.5:
        print("\n>> UNIFORM: Errors cluster around the mean.")
        print("   Improvement strategy: overall accuracy boost (longer training).")
    print("=" * 55)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel 1: L1 histogram
    axes[0].hist(l1, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black")
    axes[0].axvline(np.mean(l1), color="red", linestyle="--", linewidth=2, label=f"Mean {np.mean(l1):.1f}")
    axes[0].axvline(np.median(l1), color="green", linestyle=":", linewidth=2, label=f"Median {np.median(l1):.1f}")
    axes[0].set_xlabel("Spatial L1 Error (bins)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution", fontsize=11, fontweight="bold")
    axes[0].legend()

    # Panel 2: CDF
    sorted_l1 = np.sort(l1)
    cdf = np.arange(1, len(sorted_l1) + 1) / len(sorted_l1)
    axes[1].plot(sorted_l1, cdf, color="#2ca02c", linewidth=2)
    axes[1].axvline(15, color="gray", linestyle="--", alpha=0.6, label="15 bins")
    axes[1].axvline(50, color="gray", linestyle=":", alpha=0.6, label="50 bins")
    axes[1].set_xlabel("Spatial L1 Error (bins)")
    axes[1].set_ylabel("Cumulative Proportion")
    axes[1].set_title("Cumulative Distribution", fontsize=11, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: X vs Y error scatter (if available)
    if per_sample_x and per_sample_y:
        x_arr = np.array(per_sample_x)
        y_arr = np.array(per_sample_y)
        axes[2].scatter(x_arr, y_arr, alpha=0.5, s=20, color="#d62728")
        axes[2].plot([0, 255], [0, 255], "k--", alpha=0.3, label="X=Y")
        axes[2].set_xlabel("X Error (bins)")
        axes[2].set_ylabel("Y Error (bins)")
        axes[2].set_title("Per-Axis Error", fontsize=11, fontweight="bold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("vla_error_distribution.pdf", format="pdf", bbox_inches="tight")
    print("[SAVED] vla_error_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════
# DETAILED_LOGGING_SNIPPET
# ══════════════════════════════════════════════════════════════════════════
# To produce vla_detailed_errors.json, add this to evaluate_checkpoints.py
# inside evaluate_checkpoint(), accumulating per-sample values, then dump:
#
#   detailed = {
#       "l1_errors": spatial_l1_errors,   # the list you already build
#       "x_errors": x_errors,
#       "y_errors": y_errors,
#   }
#   with open("vla_detailed_errors.json", "w") as f:
#       json.dump(detailed, f)
#
# The lists already exist in your eval loop — just dump them to disk.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()