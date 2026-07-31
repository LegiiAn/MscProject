"""
Statistical Significance Testing for Depth Ablation
===================================================
Turns the ablation table from descriptive ("0.123 vs 0.416") into
inferential ("the difference is significant, p < 0.001").

Runs:
  - Paired Wilcoxon signed-rank tests between methods (non-parametric,
    appropriate since RMSE errors are not normally distributed)
  - Bootstrap 95% confidence intervals on each method's mean RMSE
  - Effect size (median difference)

Requires per-image RMSE arrays. Modify ablation_depth.py to also dump
the raw per-image lists (see snippet at bottom), OR this script will
regenerate them if the ablation is importable.

Reads: ablation_per_image.json  (per-image RMSE arrays)
No GPU needed.

Usage:
    python src/evaluation/ablation_statistics.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def bootstrap_ci(data, n_boot=10000, ci=95):
    """Bootstrap confidence interval for the mean."""
    data = np.array(data)
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper


def main():
    per_image_file = Path("ablation_per_image.json")

    if not per_image_file.exists():
        print(f"[ERROR] {per_image_file} not found.")
        print("        Add this to the end of ablation_depth.py main(), before the figure:")
        print("""
    per_image = {m: summary[m]["rmses"] for m in methods if m in summary}
    with open("ablation_per_image.json", "w") as f:
        json.dump(per_image, f)
    print("[SAVED] ablation_per_image.json")
        """)
        return

    with open(per_image_file, "r") as f:
        per_image = json.load(f)

    methods = list(per_image.keys())
    print("=" * 65)
    print("STATISTICAL ANALYSIS OF DEPTH RECONSTRUCTION ABLATION")
    print("=" * 65)

    # ── Bootstrap CIs ────────────────────────────────────────────────────
    print("\nBootstrap 95% Confidence Intervals on Mean RMSE:")
    print("-" * 65)
    ci_results = {}
    for m in methods:
        data = per_image[m]
        if not data:
            continue
        mean, lo, hi = bootstrap_ci(data)
        ci_results[m] = (mean, lo, hi)
        print(f"  {m:<28} {mean:.4f} m  [{lo:.4f}, {hi:.4f}]")

    # ── Paired Wilcoxon tests ────────────────────────────────────────────
    print("\nPaired Wilcoxon Signed-Rank Tests:")
    print("-" * 65)

    # Key comparisons
    comparisons = [
        ("inpaint_baseline", "poisson_gt_normals"),
        ("inpaint_baseline", "poisson_pred_normals"),
        ("poisson_gt_normals", "poisson_pred_normals"),
        ("raw_corrupted", "inpaint_baseline"),
    ]

    test_results = {}
    for a, b in comparisons:
        if a not in per_image or b not in per_image:
            continue
        arr_a = np.array(per_image[a])
        arr_b = np.array(per_image[b])
        # Match lengths (paired on same images)
        n = min(len(arr_a), len(arr_b))
        arr_a, arr_b = arr_a[:n], arr_b[:n]

        try:
            stat, pval = stats.wilcoxon(arr_a, arr_b)
        except ValueError as e:
            print(f"  {a} vs {b}: test failed ({e})")
            continue

        median_diff = np.median(arr_a - arr_b)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {a:<24} vs {b:<24}")
        print(f"      W={stat:.1f}, p={pval:.2e} [{sig}], median diff={median_diff:+.4f}m")
        test_results[f"{a}_vs_{b}"] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "median_diff": float(median_diff),
            "significant": bool(pval < 0.05),
        }

    print("\n  Legend: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")
    print("=" * 65)

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "confidence_intervals": {m: {"mean": float(ci_results[m][0]), "ci_low": float(ci_results[m][1]),
                                     "ci_high": float(ci_results[m][2])} for m in ci_results},
        "wilcoxon_tests": test_results,
    }
    with open("ablation_statistics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[SAVED] ablation_statistics.json")

    # ── Figure: CI plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    labels = []
    means = []
    errs_low = []
    errs_high = []
    colors = {"raw_corrupted": "#d62728", "inpaint_baseline": "#ff7f0e",
              "poisson_gt_normals": "#2ca02c", "poisson_pred_normals": "#1f77b4"}
    bar_colors = []

    for m in methods:
        if m in ci_results:
            mean, lo, hi = ci_results[m]
            labels.append(m.replace("_", "\n"))
            means.append(mean)
            errs_low.append(mean - lo)
            errs_high.append(hi - mean)
            bar_colors.append(colors.get(m, "gray"))

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=[errs_low, errs_high], capsize=6,
           color=bar_colors, alpha=0.7, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean RMSE (metres)")
    ax.set_title("Depth Reconstruction with 95% Bootstrap CIs", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for i, mean in enumerate(means):
        ax.text(i, mean + errs_high[i] + 0.005, f"{mean:.4f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("ablation_statistics.pdf", format="pdf", bbox_inches="tight")
    print("[SAVED] ablation_statistics.pdf")


if __name__ == "__main__":
    main()