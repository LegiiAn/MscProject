#!/usr/bin/env python3
"""
make_presentation_figures.py
============================
Generates the QUANTITATIVE presentation figures that do not already exist as
dissertation PDFs.

Run it from your project root (or pass --root):

    python make_presentation_figures.py --root ~/MscProject --out slides_figs

Every figure tries to read your REAL result files first. If a file is missing,
it falls back to the numbers recorded in your dissertation and prints a loud
"!! FALLBACK" warning. Anything marked FALLBACK must be checked against your
dissertation before it goes on a slide.

Figures produced (slide numbers refer to presentation_blueprint_v2.md):
    fig_loss_curves            -> Slide 15   Run 1 vs Run 2 training dynamics
    fig_centre_prior           -> Slide 15   model vs centre-of-image prior
    fig_grasp_run1_vs_run2     -> Slide 16   grasp success, both runs, with CIs
    fig_native_vs_composite    -> Slide 17   THE headline finding
    fig_size_effect            -> backup     on-target vs object size
    fig_perception_ood         -> Slide 18   generalisation by shape similarity
    fig_depth_ablation_bars    -> Slide 19   inpainting vs Poisson (slide-ready)
    fig_latency                -> Slide 20   where the 3.9 s actually goes
    fig_compute_budget         -> Slide 13   LoRA trainable-parameter fraction
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from graphgenerate.slide_style import apply_style, save, annotate_bars, warn_fallback, C


# ===========================================================================
# FALLBACK VALUES — from your dissertation. Edit here if any number changes.
# ===========================================================================
FB = {
    # Grasp success (%) with 95% bootstrap CIs
    "run1": {"on_target": (2.6, 1.1, 6.0),
             "on_object": (4.7, 2.5, 8.8),
             "proximity": (14.7, 10.4, 20.5),
             "l1_bins": 27.6, "n": 190},
    "run2": {"on_target": (7.2, 4.6, 11.0),
             "on_object": (12.7, 9.2, 17.4),
             "proximity": (None, None, None),   # fill if you have it
             "l1_bins": 20.59, "n": 251},

    # Run 2 breakdown by sample type — the sharpest result in the project
    "by_type": {"Native ClearGrasp": 11.3, "Composite": 3.1},

    # Run 2 breakdown by object size (px area buckets)
    "by_size": {"< 142 px": 0.0, "142–562 px": 4.5, "> 562 px": 14.1},

    # Perception: mean angular error (deg) and segmentation IoU (%)
    "perception": {
        "Bottle\n(in-distribution)": (19.55, 91.10),
        "Champagne glass\n(unseen, bottle-like)": (25.53, 77.87),
        "Wavy cup\n(unseen, corrugated)": (34.06, 77.28),
    },

    # Depth reconstruction RMSE (metres), 100-image ablation
    "depth": {
        "Corrupted\n(no repair)":      0.930,
        "Inpainting\n(Navier-Stokes)": 0.1227,
        "Poisson\n(GT normals)":       0.1364,
        "Poisson\n(predicted normals)": 0.4160,
    },

    # Latency (ms per frame)
    "latency": {"Perception (GPU)": 15.0,
                "Depth solver (CPU)": 2513.0,
                "VLA inference": 1337.0},

    # Centre-of-image prior vs model, mean L1 in bins
    "centre_prior": {"Run 1 model": 27.6, "Run 1 centre prior": 33.0,
                     "Run 2 model": 20.59, "Run 2 centre prior": 34.2},


    # LoRA budget
    "lora": {"trainable": 16.78e6, "total": 7.56e9},
}

TOTAL_LAT = sum(FB["latency"].values())


# ===========================================================================
# Loaders — read real files if present
# ===========================================================================
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_loss_curves(root):
    """
    Read vla_checkpoints*/training_log.json for both runs.
    Expected: a list of dicts with 'step' and 'loss' (and optionally 'val_loss'),
    or a dict with those keys as lists. Both shapes handled.
    """
    def parse(log):
        if log is None:
            return None
        if isinstance(log, dict):
            for k in ("history", "log", "entries", "records"):
                if k in log and isinstance(log[k], list):
                    log = log[k]
                    break
        if isinstance(log, dict):
            steps = log.get("step") or log.get("steps")
            loss = log.get("loss") or log.get("train_loss")
            val = log.get("val_loss") or log.get("eval_loss")
            if steps and loss:
                return np.array(steps), np.array(loss), (np.array(val) if val else None)
            return None
        if isinstance(log, list) and log and isinstance(log[0], dict):
            steps = [e.get("step", i) for i, e in enumerate(log)]
            loss = [e.get("loss", e.get("train_loss")) for e in log]
            val = [e.get("val_loss", e.get("eval_loss")) for e in log]
            if any(v is None for v in loss):
                return None
            val_arr = np.array([np.nan if v is None else v for v in val], dtype=float)
            if np.all(np.isnan(val_arr)):
                val_arr = None
            return np.array(steps, dtype=float), np.array(loss, dtype=float), val_arr
        return None

    r1 = parse(load_json(root / "vla_checkpoints_run1" / "training_log.json"))
    r2 = parse(load_json(root / "vla_checkpoints" / "training_log.json"))
    return r1, r2


# ===========================================================================
# SLIDE 15 — training dynamics
# ===========================================================================
def fig_loss_curves(root, out):
    r1, r2 = get_loss_curves(root)
    fig, ax = plt.subplots(figsize=(13, 7))

    if r1 is None:
        warn_fallback("Run 1 loss curve", root / "vla_checkpoints_run1/training_log.json")
        s1 = np.linspace(0, 13000, 260)
        l1 = 1.9 * np.exp(-s1 / 1100)
        l1[s1 > 4000] = 0.0                      # the collapse to exactly zero
    else:
        s1, l1, _ = r1

    if r2 is None:
        warn_fallback("Run 2 loss curve", root / "vla_checkpoints/training_log.json")
        s2 = np.linspace(0, 21500, 320)
        l2 = 0.154 + 1.5 * np.exp(-s2 / 2600) + 0.012 * np.random.RandomState(0).randn(s2.size)
        v2 = 0.160 + 1.5 * np.exp(-s2 / 2600)
    else:
        s2, l2, v2 = r2

    ax.plot(s1, l1, color=C["run1"], label="Run 1 — dummy actions (broken)")
    ax.plot(s2, l2, color=C["run2"], label="Run 2 — real per-image targets")
    if v2 is not None:
        ax.plot(s2, v2, color=C["run2"], ls="--", lw=2.2, alpha=0.75,
                label="Run 2 — validation")

    ax.axhline(0.0, color=C["run1"], ls=":", lw=2, alpha=0.55)
    ax.annotate("loss = 0.0000\nmemorised one constant",
                xy=(s1[-1] * 0.72, 0.0), xytext=(s1[-1] * 0.42, 0.62),
                color=C["run1"], fontweight="bold", fontsize=18,
                arrowprops=dict(arrowstyle="->", color=C["run1"], lw=2.4))
    ax.annotate("plateaus ≈ 0.15\nlearning something real",
                xy=(s2[-1] * 0.85, 0.155), xytext=(s2[-1] * 0.50, 0.95),
                color=C["run2"], fontweight="bold", fontsize=18,
                arrowprops=dict(arrowstyle="->", color=C["run2"], lw=2.4))

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (action tokens)")
    ax.set_ylim(-0.08, 2.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save(fig, out, "fig_loss_curves")


# ===========================================================================
# SLIDE 15 companion — the uncomfortable comparison
# ===========================================================================
def fig_centre_prior(root, out):
    # Try to load your new JSON file
    cp_data = load_json(root / "results" / "center_prior.json")
    cp = dict(FB["centre_prior"])
    
    if cp_data is None:
        warn_fallback("centre-prior comparison", root / "results" / "center_prior.json")
    else:
        # Swap out the 'None' for your real mean L1 bins value!
        cp["Run 2 centre prior"] = cp_data["centre_prior_mean_l1_bins"]

    labels, vals, cols = [], [], []
    for k in ("Run 1 model", "Run 1 centre prior", "Run 2 model", "Run 2 centre prior"):
        v = cp[k]
        labels.append(k.replace(" ", "\n", 1))
        vals.append(np.nan if v is None else v)
        cols.append(C["run1"] if "Run 1" in k else C["run2"])
        if "prior" in k:
            cols[-1] = C["baseline"]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    plotted = [0 if np.isnan(v) else v for v in vals]
    bars = ax.bar(x, plotted, color=cols, width=0.62, edgecolor="none")

    for i, (b, v) in enumerate(zip(bars, vals)):
        if np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, 1.5, "NOT YET\nRECOMPUTED",
                    ha="center", va="bottom", color=C["run1"],
                    fontweight="bold", fontsize=17)
            b.set_facecolor("none")
            b.set_edgecolor(C["run1"])
            b.set_linewidth(2.5)
            b.set_linestyle("--")
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", va="bottom", fontweight="bold", fontsize=21)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean spatial L1 error (bins)")
    ax.set_ylim(0, 40)
    ax.set_title("Lower is better — the model must beat the grey bar", pad=16)
    fig.tight_layout()
    return save(fig, out, "fig_centre_prior")


# ===========================================================================
# SLIDE 16 — grasp success, both runs, with confidence intervals
# ===========================================================================
# ===========================================================================
# SLIDE 16 — grasp success, both runs, with confidence intervals
# ===========================================================================
def fig_grasp_run1_vs_run2(root, out):
    res2 = load_json(root / "results" / "grasp_success_results_run2.json")
    if res2 is None:
        warn_fallback("Run 2 grasp success", root / "results" / "grasp_success_results_run2.json")
    else:
        # Calculate real percentages from the JSON counts
        n2 = res2["overall"]["n"]
        ot = res2["overall"]["on_target"] / n2 * 100
        oa = res2["overall"]["on_any"] / n2 * 100
        pr = res2["overall"]["proximity"] / n2 * 100
        
        ot_ci = res2["overall"].get("on_target_ci", [ot, ot])
        oa_ci = res2["overall"].get("on_any_ci", [oa, oa])
        
        # Overwrite the fallback dictionary with your real data
        FB["run2"]["on_target"] = (ot, ot_ci[0], ot_ci[1])
        FB["run2"]["on_object"] = (oa, oa_ci[0], oa_ci[1])
        FB["run2"]["proximity"] = (pr, pr, pr) 

    crit = ["on_target", "on_object", "proximity"]
    nice = ["ON-TARGET\n(inside mask)", "ON-OBJECT\n(any object)", "PROXIMITY\n(within radius)"]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(crit))
    w = 0.36

    for off, run, col, lab in [(-w / 2, "run1", C["run1"], "Run 1"),
                               (+w / 2, "run2", C["run2"], "Run 2")]:
        vals, los, his = [], [], []
        for c in crit:
            m, lo, hi = FB[run][c]
            vals.append(np.nan if m is None else m)
            los.append(0 if m is None or lo is None else m - lo)
            his.append(0 if m is None or hi is None else hi - m)
        plotted = [0 if np.isnan(v) else v for v in vals]
        bars = ax.bar(x + off, plotted, w, color=col, label=lab, edgecolor="none")
        ax.errorbar(x + off, plotted, yerr=[los, his], fmt="none",
                    ecolor=C["ink"], elinewidth=2, capsize=7, capthick=2)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.6, "n/a", ha="center",
                        color=C["muted"], fontsize=16)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}%",
                        ha="center", va="bottom", fontweight="bold", fontsize=19)

    ax.set_xticks(x)
    ax.set_xticklabels(nice)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 26)
    ax.legend(loc="upper left")
    ax.set_title("Error bars: 95% bootstrap CI", fontsize=17, color=C["muted"], pad=14)
    fig.tight_layout()
    return save(fig, out, "fig_grasp_run1_vs_run2")



# ===========================================================================
# SLIDE 17 — THE headline finding
# ===========================================================================
# ===========================================================================
# SLIDE 17 — THE headline finding
# ===========================================================================
def fig_native_vs_composite(root, out):
    res2 = load_json(root / "results" / "grasp_success_results_run2.json")
    d = FB["by_type"]
    if res2 is None:
        warn_fallback("native vs composite breakdown", root / "results" / "grasp_success_results_run2.json")
    else:
        n_real = res2["by_type"]["real"]["n"]
        n_comp = res2["by_type"]["composite"]["n"]
        d = {
            "Native ClearGrasp": res2["by_type"]["real"]["on_target"] / n_real * 100,
            "Composite": res2["by_type"]["composite"]["on_target"] / n_comp * 100
        }

    fig, ax = plt.subplots(figsize=(11, 7.5))
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    bars = ax.bar(keys, vals, color=[C["native"], C["composite"]],
                  width=0.52, edgecolor="none")
    annotate_bars(ax, bars, fmt="{:.1f}%", dy=0.35, fontsize=30)

    ratio = vals[0] / vals[1]
    ax.annotate("", xy=(0, vals[0] + 1.4), xytext=(1, vals[1] + 1.4),
                arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=3))
    ax.text(0.5, max(vals) + 2.3, f"{ratio:.1f}×", ha="center",
            color=C["accent"], fontweight="bold", fontsize=34)

    ax.set_ylabel("ON-TARGET rate (%)")
    ax.set_ylim(0, max(vals) + 5)
    ax.set_title("Same model, same weights — only the training-image physics differs",
                 fontsize=17, color=C["muted"], pad=14)
    fig.tight_layout()
    return save(fig, out, "fig_native_vs_composite")


# ===========================================================================
# Backup — size effect
# ===========================================================================
# ===========================================================================
# Backup — size effect
# ===========================================================================
def fig_size_effect(root, out):
    res2 = load_json(root / "results" / "grasp_success_results_run2.json")
    d = FB["by_size"]
    if res2 is None:
        warn_fallback("size-bucket breakdown", root / "results" / "grasp_success_results_run2.json")
    else:
        s = res2["by_size"]
        d = {
            "< 142 px": s["SMALL (area<142px)"]["on_target"] / s["SMALL (area<142px)"]["n"] * 100,
            "142–562 px": s["MEDIUM"]["on_target"] / s["MEDIUM"]["n"] * 100,
            "> 562 px": s["LARGE (area>=562px)"]["on_target"] / s["LARGE (area>=562px)"]["n"] * 100
        }

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(list(d.keys()), list(d.values()),
                  color=[C["run1"], C["composite"], C["native"]],
                  width=0.55, edgecolor="none")
    annotate_bars(ax, bars, fmt="{:.1f}%", dy=0.3, fontsize=24)
    ax.set_xlabel("Object area in image")
    ax.set_ylabel("ON-TARGET rate (%)")
    ax.set_ylim(0, 18)
    ax.set_title("Small objects are invisible to a 224×224 tokeniser",
                 fontsize=17, color=C["muted"], pad=14)
    fig.tight_layout()
    return save(fig, out, "fig_size_effect")


# ===========================================================================
# SLIDE 18 — perception generalisation
# ===========================================================================
def fig_perception_ood(root, out):
    p = FB["perception"]
    labels = list(p.keys())
    ang = [p[k][0] for k in labels]
    iou = [p[k][1] for k in labels]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6))
    x = np.arange(len(labels))
    cols = [C["run2"], C["composite"], C["run1"]]

    b1 = a1.bar(x, ang, color=cols, width=0.58, edgecolor="none")
    annotate_bars(a1, b1, fmt="{:.1f}°", dy=0.6, fontsize=20)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=15)
    a1.set_ylabel("Mean angular error (°)")
    a1.set_ylim(0, 42)
    a1.set_title("Geometry degrades", pad=12)
    a1.annotate("+6.0°", xy=(1, ang[1] + 3.2), ha="center",
                color=C["muted"], fontsize=17, fontweight="bold")
    a1.annotate("+14.5°", xy=(2, ang[2] + 3.2), ha="center",
                color=C["run1"], fontsize=17, fontweight="bold")

    b2 = a2.bar(x, iou, color=cols, width=0.58, edgecolor="none")
    annotate_bars(a2, b2, fmt="{:.1f}%", dy=1.0, fontsize=20)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=15)
    a2.set_ylabel("Segmentation IoU (%)")
    a2.set_ylim(0, 105)
    a2.set_title("Segmentation holds", pad=12)

    fig.tight_layout()
    return save(fig, out, "fig_perception_ood")


# ===========================================================================
# SLIDE 19 — depth ablation, slide-ready single panel
# ===========================================================================
# ===========================================================================
# SLIDE 19 — depth ablation, slide-ready single panel
# ===========================================================================
def fig_depth_ablation_bars(root, out):
    ab = load_json(root / "results" / "ablation_depth_results.json")
    d = FB["depth"]
    if ab is None:
        warn_fallback("depth ablation RMSE", root / "results" / "ablation_depth_results.json")
    else:
        d = {
            "Corrupted\n(no repair)": ab["raw_corrupted"]["rmse_mean"],
            "Inpainting\n(Navier-Stokes)": ab["inpaint_baseline"]["rmse_mean"],
            "Poisson\n(GT normals)": ab["poisson_gt_normals"]["rmse_mean"],
            "Poisson\n(predicted normals)": ab["poisson_pred_normals"]["rmse_mean"]
        }

    labels = list(d.keys())
    vals = [d[k] for k in labels]
    cols = [C["baseline"], C["native"], C["run2"], C["run1"]]

    fig, ax = plt.subplots(figsize=(13.5, 7))
    bars = ax.bar(labels, vals, color=cols, width=0.58, edgecolor="none")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.3f}",
                ha="center", va="bottom", fontweight="bold", fontsize=20)
    ax.set_yscale("log")
    ax.set_ylabel("Depth RMSE (m, log scale)")
    ax.set_ylim(0.05, 2.0)
    ax.tick_params(axis="x", labelsize=15)

    ax.annotate("inpainting wins\n(p = 0.02)",
                xy=(1, vals[1]), xytext=(1.55, 0.045),
                color=C["native"], fontweight="bold", fontsize=19,
                arrowprops=dict(arrowstyle="->", color=C["native"], lw=2.4))
    ax.set_title("Even given ground-truth normals, the solver loses on shallow objects",
                 fontsize=16, color=C["muted"], pad=14)
    fig.tight_layout()
    return save(fig, out, "fig_depth_ablation_bars")


# ===========================================================================
# SLIDE 20 — latency
# ===========================================================================
def fig_latency(root, out):
    bench = load_json(root / "results" / "latency_benchmark.json")
    lat = dict(FB["latency"])
    if bench is None:
        warn_fallback("latency breakdown", root / "results" / "latency_benchmark.json")
    else:
        for key, aliases in [("Perception (GPU)", ["perception", "wp2", "perception_ms"]),
                             ("Depth solver (CPU)", ["depth", "poisson", "solver", "depth_ms"]),
                             ("VLA inference", ["vla", "policy", "openvla", "vla_ms"])]:
            for a in aliases:
                if a in bench:
                    v = bench[a]
                    lat[key] = v.get("mean", v) if isinstance(v, dict) else v
                    break

    total = sum(lat.values())
    fig, ax = plt.subplots(figsize=(14, 4.6))
    left = 0
    order = ["Perception (GPU)", "Depth solver (CPU)", "VLA inference"]
    cols = {"Perception (GPU)": C["native"],
            "Depth solver (CPU)": C["run1"],
            "VLA inference": C["run2"]}
    for k in order:
        v = lat[k]
        ax.barh([0], [v], left=left, color=cols[k], height=0.55, edgecolor="none")
        pct = 100 * v / total
        if pct > 6:
            ax.text(left + v / 2, 0, f"{v:.0f} ms\n{pct:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=21)
        left += v

    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.6, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("Latency per frame (ms)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(facecolor=cols[k], label=k) for k in order],
              loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=3, fontsize=17)
    ax.text(total * 0.5, -0.48, f"total {total:.0f} ms  →  {1000/total:.2f} FPS",
            ha="center", fontsize=22, fontweight="bold", color=C["ink"])
    fig.tight_layout()
    return save(fig, out, "fig_latency")


# ===========================================================================
# SLIDE 13 — compute budget visual
# ===========================================================================
def fig_compute_budget(root, out):
    tr, tot = FB["lora"]["trainable"], FB["lora"]["total"]
    frac = tr / tot

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh([0], [100], color=C["grid"], height=0.42, edgecolor="none")
    ax.barh([0], [frac * 100], color=C["accent"], height=0.42, edgecolor="none")

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.3, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(50, 0.42, f"frozen base — {tot/1e9:.2f} B parameters (4-bit NF4)",
            ha="center", fontsize=20, color=C["muted"])
    ax.text(2, -0.62, f"LoRA trainable: {tr/1e6:.2f} M  =  {frac*100:.2f}%",
            ha="left", fontsize=25, fontweight="bold", color=C["accent"])
    ax.text(2, -1.05, "rank 32 · α 32 · dropout 0.05 · single A100",
            ha="left", fontsize=18, color=C["muted"])
    fig.tight_layout()
    return save(fig, out, "fig_compute_budget")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root (e.g. ~/MscProject)")
    ap.add_argument("--out", default="slides_figs", help="output directory")
    ap.add_argument("--only", default=None, help="run one figure by name")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    apply_style()

    figs = {
        "loss_curves":        fig_loss_curves,
        "centre_prior":       fig_centre_prior,
        "grasp_run1_vs_run2": fig_grasp_run1_vs_run2,
        "native_vs_composite": fig_native_vs_composite,
        "size_effect":        fig_size_effect,
        "perception_ood":     fig_perception_ood,
        "depth_ablation":     fig_depth_ablation_bars,
        "latency":            fig_latency,
        "compute_budget":     fig_compute_budget,
    }
    if args.only:
        figs = {args.only: figs[args.only]}

    print(f"project root : {root}")
    print(f"output dir   : {out}\n")
    for name, fn in figs.items():
        print(f"[{name}]")
        fn(root, out)
        print()

    print("Done. Check every '!! FALLBACK' warning above before presenting.")


if __name__ == "__main__":
    main()