#!/usr/bin/env python3
"""
make_new_slide_figures.py — figures for the v4 deck.
"""

import os, json, argparse, random
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"run1": "#C44536", "run2": "#2A6F97", "native": "#2A9D5C",
     "composite": "#E08A1E", "baseline": "#8A8A8A", "accent": "#6A4C93",
     "ink": "#1A1A1A", "muted": "#6B6B6B"}
plt.rcParams.update({"font.size": 14, "figure.facecolor": "none",
                     "savefig.transparent": True, "axes.facecolor": "none"})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight", transparent=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig); print(f"  wrote {out/name}.png/.pdf")


# ---------------------------------------------------------------- run2 vs run3
FB_R2 = {"baseline": 18, "paraphrase_1": 28, "paraphrase_2": 50, "paraphrase_3": 26,
         "absent_can": 52, "absent_cube": 73, "null_generic": 76}
FB_R3 = {"baseline": 1.1, "paraphrase_1": 1.5, "paraphrase_2": 3.6, "paraphrase_3": 1.0,
         "absent_can": 2.0, "absent_cube": 0.9, "null_generic": 0.4}

def read_parsefail(path):
    try:
        d = json.load(open(path))
        s = d.get("summary", d)
        return {k: v.get("parse_fail_pct") for k, v in s.items()
                if isinstance(v, dict) and "parse_fail_pct" in v}
    except Exception:
        return None


def fig_run2vs3(root, out):
    # --- 1. Load Data (Corrected Paths) ---
    r2 = read_parsefail(root / "results" / "instruction_sensitivity_v2.json")         
    r3 = read_parsefail(root / "results" / "instruction_sensitivity_run3.json")    
    
    if r2 is None:
        print("  !! FALLBACK Run2 values"); r2 = FB_R2
    if r3 is None:
        print("  !! FALLBACK Run3 values"); r3 = FB_R3
        
    conds = [c for c in FB_R2 if c in r2 and c in r3]
            
    # --- 2. Clean X-Axis Labels ---
    def clean_label(c):
        if c == "baseline": return "Baseline\n(Train Phrase)"
        if "paraphrase" in c: return c.replace("paraphrase_", "Para ")
        if "absent" in c: return c.replace("absent_", "Absent\n")
        if "null" in c: return "Null\nGeneric"
        return c.replace("_", "\n")
        
    labels = [clean_label(c) for c in conds]
    
    # --- 3. Figure Setup ---
    fig, ax = plt.subplots(figsize=(14, 6.5))
    
    color_r2 = "#E06666"  
    color_r3 = "#2CA02C"  
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE", linestyle="-", linewidth=1.5)
    
    x = np.arange(len(conds))
    w = 0.35
    
    # --- 4. Plot Bars ---
    b2 = ax.bar(x - w/2, [r2[c] for c in conds], w, color=color_r2, 
                edgecolor="white", linewidth=1, label="Run 2 (Single Train Phrase)")
    b3 = ax.bar(x + w/2, [r3[c] for c in conds], w, color=color_r3, 
                edgecolor="white", linewidth=1, label="Run 3 (Diverse Phrasings)")
    
    # --- 5. Add Value Annotations ---
    for b in list(b2) + list(b3):
        h = b.get_height()
        val_str = f"{h:.0f}%" if h >= 3 else f"{h:.1f}%"
        ax.text(b.get_x() + b.get_width()/2, h + 1.5,
                val_str, ha="center", va="bottom", 
                fontweight="bold", fontsize=11, color="#333333")
                
    # --- 6. Grouping Dividers ---
    para_start = next((i for i, c in enumerate(conds) if "paraphrase" in c), -1)
    edge_start = next((i for i, c in enumerate(conds) if "absent" in c or "null" in c), -1)
    
    if para_start > 0:
        ax.axvline(x=para_start - 0.5, color='black', alpha=0.15, linestyle='--', linewidth=1.5)
    if edge_start > 0:
        ax.axvline(x=edge_start - 0.5, color='black', alpha=0.15, linestyle='--', linewidth=1.5)
        
    # --- 7. Axes & Spines Cleanup ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="medium")
    ax.set_ylabel("Parse Failure Rate (%)\nLower = More Robust", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylim(0, max(max(r2.values()), max(r3.values())) + 15) 
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=11) 
    
    ax.legend(fontsize=12, loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD")
    
    plt.tight_layout()
    save(fig, out, "fig_instruction_run2_vs_run3")

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Default to your exact local Windows directory
    ap.add_argument("--root", default=r"C:\Users\m_vit\Documents\MscProject")
    ap.add_argument("--out", default="slides_figs")
    ap.add_argument("--only", default=None, choices=["run2vs3"])
    args = ap.parse_args()
    
    root = Path(args.root).expanduser()
    out = root / args.out
    
    jobs = {"run2vs3": lambda: fig_run2vs3(root, out)}
    
    for name, fn in jobs.items():
        if args.only and name != args.only:
            continue
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  !! failed: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()