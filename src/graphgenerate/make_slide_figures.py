#!/usr/bin/env python3
"""
Slide figure generator - "New experiments" slide
=================================================
One 3-panel figure (plus each panel saved separately) for the slide covering:
  A. Initial (base) VLA tested on our data  -> REAL Step-1 results
  B. Run 3 diverse-instruction retrain      -> REAL Run-2 baseline + status
  C. Gazebo Tier-1 reachability             -> PLANNED design (no results yet)

HONESTY RULES (enforced, do not weaken):
  - Panel A reads real numbers from base_model_discrimination.json.
  - Panel B plots the REAL Run-2 parse-fail gradient from instruction_sensitivity_v2.json.
    The Run-3 outcome is NOT fabricated: with no Run-3 JSON it is drawn as 'pending';
    pass --v2-run3 <file> once re-eval exists and the real after-bars replace the badge.
  - Panel C is a schematic of the PLANNED pipeline; it contains NO measured values.
  - Missing required JSON => hard abort (no silent hardcoded fallback).
  - Every plotted number is printed to stdout so it can be eyeballed against the JSON.

Run (now, pre-Run-3):
    python make_slide_figures.py --base base_model_discrimination.json \
                                 --v2 instruction_sensitivity_v2.json
Run (after Run-3 re-eval):
    python make_slide_figures.py --base base_model_discrimination.json \
                                 --v2 instruction_sensitivity_v2.json \
                                 --v2-run3 instruction_sensitivity_v2_run3.json
"""
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_BASE   = "#1b9e77"   # base / positive / Run-3-good  (teal)
C_WARN   = "#d95f02"   # collapse / Run-2              (orange)
C_PEND   = "#9e9e9e"   # pending / planned            (grey)
C_HYP    = "#7570b3"   # hypothesis line              (muted purple)
GRAD     = plt.cm.YlOrRd


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, c - m) * 100, min(1.0, c + m) * 100


def load_json(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[ABORT] missing {what}: {p}\n"
                         f"        (no hardcoded fallback by design - point the flag at the real file)")
    return json.load(open(p))


# --------------------------------------------------------------------------
def panel_a(ax, base):
    """Base VLA is instruction-conditioned on our data (Step 1, real)."""
    pc_frac = base["positive_control"]["sensitive_fraction"]
    pc_n    = base["positive_control"]["n_frames"]
    cb_frac = base["composite_base"]["sensitive_fraction"]
    cb_n    = base["composite_base"]["n_frames"]
    l2      = base["composite_base"]["mean_correct_vs_wrongcan_l2"]

    pc, cb = pc_frac * 100, cb_frac * 100
    pc_lo, pc_hi = wilson(round(pc_frac * pc_n), pc_n)
    cb_lo, cb_hi = wilson(round(cb_frac * cb_n), cb_n)
    print(f"[A] positive control : {pc:.1f}% sensitive (n={pc_n})  CI[{pc_lo:.0f},{pc_hi:.0f}]")
    print(f"[A] our composites   : {cb:.1f}% sensitive (n={cb_n})  CI[{cb_lo:.0f},{cb_hi:.0f}]")
    print(f"[A] correct-vs-wrong-object action shift: L2 = {l2:.3f}")

    x = [0, 1]
    vals = [pc, cb]
    elo = [max(0.0, pc - pc_lo), max(0.0, cb - cb_lo)]
    ehi = [max(0.0, pc_hi - pc), max(0.0, cb_hi - cb)]
    ax.bar(x, vals, width=0.6, color=C_BASE, yerr=[elo, ehi], capsize=6, ecolor="#333")
    for xi, v in zip(x, vals):
        ax.text(xi, min(v + 3, 101), f"{v:.0f}%", ha="center", va="bottom", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"in-distribution\nrobot frames\n(n={pc_n})",
                        f"our composites\n(n={cb_n})"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("% frames: action changes with instruction")
    ax.set_title("A \u00b7 Base VLA on our data", loc="left", weight="bold")
    ax.text(0.5, 0.28,
            f"base VLA responds to the instruction\ncorrect vs wrong object: L2 = {l2:.2f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#e8f5f0", ec=C_BASE))
    ax.spines[["top", "right"]].set_visible(False)


def panel_b(ax, v2, v2_run3=None):
    """Run-2 single-template collapse (real). Run-3 after-bars only if real JSON given."""
    summ = v2["summary"]
    conds = list(summ.keys())
    pf2 = [summ[c]["parse_fail_pct"] for c in conds]
    labels = [c.replace("_", "\n") for c in conds]
    x = np.arange(len(conds))
    print("[B] Run-2 parse-fail gradient:")
    for c, v in zip(conds, pf2):
        print(f"     {c:16s} {v:5.1f}%")

    if v2_run3 is not None:
        s3 = v2_run3["summary"]
        pf3 = [s3.get(c, {}).get("parse_fail_pct", np.nan) for c in conds]
        print("[B] Run-3 parse-fail gradient (real, plotted as after-bars):")
        for c, v in zip(conds, pf3):
            print(f"     {c:16s} {v:5.1f}%")
        w = 0.4
        ax.bar(x - w / 2, pf2, w, color=C_WARN, label="Run 2 (single template)")
        ax.bar(x + w / 2, pf3, w, color=C_BASE, label="Run 3 (diverse templates)")
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    else:
        colors = GRAD(np.linspace(0.28, 0.9, len(conds)))
        ax.bar(x, pf2, width=0.72, color=colors)
        base_pf = pf2[0]
        ax.axhline(base_pf, ls="--", lw=1.4, color=C_HYP)
        ax.text(len(conds) - 0.4, base_pf + 2,
                f"Run 3 target \u2248 {base_pf:.0f}%\n(hypothesis \u00b7 pending)",
                color=C_HYP, ha="right", va="bottom", fontsize=8, style="italic")
        ax.text(0.98, 0.03, "RUN 3: TRAINING\nafter-bars pending re-eval",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, weight="bold",
                bbox=dict(boxstyle="round", fc="#fff3cd", ec=C_WARN))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("parse-fail %  (fine-tuned)")
    ax.set_title("B \u00b7 Run 3: recover instruction-following", loc="left", weight="bold")
    ax.text(0.02, 0.97, "Run 2: parse-fail rises off the\ntraining phrase (single-template collapse)",
            transform=ax.transAxes, va="top", fontsize=8, color="#333")
    ax.spines[["top", "right"]].set_visible(False)


def panel_c(ax):
    """Gazebo Tier-1 planned pipeline. Schematic only - no measured values."""
    ax.axis("off")
    ax.set_title("C \u00b7 Gazebo Tier-1: reachability (planned)", loc="left", weight="bold")
    steps = ["Predicted 3D grasp points\n(replayed, not re-perceived)",
             "MoveIt IK  \u00b7  Panda / UR5",
             "Reachable & collision-free?"]
    ys = [0.80, 0.52, 0.24]
    for s, y in zip(steps, ys):
        ax.add_patch(FancyBboxPatch((0.08, y - 0.075), 0.84, 0.15,
                     boxstyle="round,pad=0.02", fc="#efefef", ec=C_PEND, lw=1.4,
                     transform=ax.transAxes))
        ax.text(0.5, y, s, ha="center", va="center", transform=ax.transAxes, fontsize=9)
    for y0, y1 in [(0.725, 0.595), (0.445, 0.315)]:
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), transform=ax.transAxes,
                     arrowstyle="-|>", mutation_scale=16, color=C_PEND, lw=1.4))
    ax.text(0.5, 0.06,
            "compare predicted vs ground-truth vs centre-prior\n"
            "sim depth is perfect on glass \u2192 replay only, no perception in loop",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=7.5,
            style="italic", color="#555")
    ax.text(0.97, 0.97, "PLANNED", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, weight="bold",
            bbox=dict(boxstyle="round", fc="#e2e3e5", ec=C_PEND))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/MscProject/results/base_model_discrimination.json")  
    ap.add_argument("--v2", default="/MscProject/results/instruction_sensitivity_v2.json")
    ap.add_argument("--v2-run3", default=None,
                    help="Run-3 re-eval JSON; when present, Panel B plots real after-bars")
    ap.add_argument("--out", default="slide_figs")
    args = ap.parse_args()

    base = load_json(args.base, "/MscProject/results/base_model_discrimination.json")
    v2 = load_json(args.v2, "/MscProject/results/instruction_sensitivity_v2.json")
    v2_run3 = load_json(args.v2_run3, "/MscProject/results/Run-3_v2_JSON") if args.v2_run3 else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("A = REAL Step-1 data | B = REAL Run-2 baseline"
          + (" + REAL Run-3" if v2_run3 else " + Run-3 PENDING (no fabricated bars)"))
    print("C = PLANNED schematic (no measured values)")
    print("=" * 66)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panel_a(axes[0], base)
    panel_b(axes[1], v2, v2_run3)
    panel_c(axes[2])
    fig.suptitle("New experiments:  base VLA on our data  \u2192  Run 3  \u2192  Gazebo",
                 weight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    combined = out / "slide_new_experiments.png"
    fig.savefig(combined, dpi=200, bbox_inches="tight")
    print(f"[SAVED] {combined}")

    for name, fn in [("panelA_base_vla", lambda a: panel_a(a, base)),
                     ("panelB_run3", lambda a: panel_b(a, v2, v2_run3)),
                     ("panelC_gazebo", lambda a: panel_c(a))]:
        f, a = plt.subplots(figsize=(5.6, 5.0))
        fn(a)
        f.tight_layout()
        p = out / f"{name}.png"
        f.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[SAVED] {p}")


if __name__ == "__main__":
    main()