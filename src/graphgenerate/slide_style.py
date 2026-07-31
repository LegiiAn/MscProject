"""
slide_style.py — shared look-and-feel for all presentation figures.

Presentation figures are NOT dissertation figures. They need:
  - much larger fonts (readable from the back of a room)
  - fewer elements per figure (one idea per slide)
  - transparent background (so they sit on any slide template)
  - high DPI PNG for PowerPoint + PDF for LaTeX

Import this at the top of every figure script:
    from slide_style import apply_style, save, C
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour palette. Deliberately colour-blind-safe (Okabe-Ito derived) and
# consistent across every figure so "orange = composite" everywhere, etc.
# ---------------------------------------------------------------------------
C = {
    "run1":       "#C44536",   # red    — Run 1 / the broken run
    "run2":       "#2A6F97",   # blue   — Run 2 / the fixed run
    "native":     "#2A9D5C",   # green  — native ClearGrasp frames
    "composite":  "#E08A1E",   # orange — composites
    "baseline":   "#8A8A8A",   # grey   — baselines / priors
    "accent":     "#6A4C93",   # purple — highlights
    "ink":        "#1A1A1A",   # near-black text
    "muted":      "#6B6B6B",   # secondary text
    "grid":       "#D8D8D8",
}

# Bar order used in the depth ablation, kept in one place
ABLATION_COLORS = {
    "Corrupted":            C["baseline"],
    "Inpainting":           C["native"],
    "Poisson (GT normals)": C["run2"],
    "Poisson (predicted)":  C["run1"],
}


def apply_style(base_fontsize=20):
    """Apply presentation-grade typography and axis defaults."""
    plt.rcParams.update({
        "figure.facecolor":  "none",
        "axes.facecolor":    "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,

        "font.family":     "DejaVu Sans",
        "font.size":       base_fontsize,
        "axes.titlesize":  base_fontsize + 4,
        "axes.labelsize":  base_fontsize + 1,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,

        "axes.edgecolor":  C["muted"],
        "axes.labelcolor": C["ink"],
        "text.color":      C["ink"],
        "xtick.color":     C["ink"],
        "ytick.color":     C["ink"],

        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.4,
        "axes.grid":         True,
        "grid.color":        C["grid"],
        "grid.linewidth":    0.9,
        "grid.alpha":        0.7,

        "lines.linewidth":  3.0,
        "legend.frameon":   False,

        "figure.autolayout": False,
    })


def save(fig, outdir, name, also_pdf=True, dpi=220):
    """Save a figure as PNG (for slides) and PDF (for LaTeX)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", transparent=True)
    written = [png]
    if also_pdf:
        pdf = outdir / f"{name}.pdf"
        fig.savefig(pdf, bbox_inches="tight", transparent=True)
        written.append(pdf)
    plt.close(fig)
    for w in written:
        print(f"  wrote {w}")
    return written


def annotate_bars(ax, bars, fmt="{:.1f}%", dy=0.6, fontsize=None, weight="bold"):
    """Put a value label on top of each bar — essential for slide readability."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontweight=weight,
                fontsize=fontsize or plt.rcParams["font.size"] + 1)


def warn_fallback(what, path):
    """Loudly flag when a figure was built from hardcoded numbers, not your data."""
    print(f"  !! FALLBACK: could not read {path}")
    print(f"     -> {what} was drawn from hardcoded values in this script.")
    print(f"     -> Verify these against your dissertation before presenting.")