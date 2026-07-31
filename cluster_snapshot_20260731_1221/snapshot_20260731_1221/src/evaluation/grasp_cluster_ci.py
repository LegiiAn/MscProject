"""
Cluster-Aware Bootstrap CIs for the grasp-success results
=========================================================
WHY: the 251 grasp samples are NOT independent. They come from ~75 source
scenes, each contributing several correlated variants (2 composites + 2 near-
identical real frames of the SAME bottle). Standard Wilson CIs assume
independence, so they are too NARROW — they overstate certainty.

FIX: bootstrap by resampling whole SCENES (source_file), not individual samples.
This propagates the within-scene correlation into the interval width. Point
estimates are unchanged; only the CIs widen to their honest width.

Reports cluster-aware 95% CIs for:
  - overall ON-TARGET / ON-OBJECT
  - real vs composite ON-TARGET  AND the (real - composite) DIFFERENCE
    (the difference CI is the real test of your headline finding: if it
     excludes 0, native>composite is significant under correlation)
  - size terciles

Reads : grasp_per_sample.json   (list of per-sample dicts incl. source_file)
No GPU. No torch. Pure numpy.

Usage:
    python src/evaluation/grasp_cluster_ci.py
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def cluster_bootstrap(rows, key, selector=None, n_boot=10000, seed=42):
    """
    Bootstrap a proportion by resampling whole scenes (source_file).

    rows     : list of per-sample dicts
    key      : boolean field to average, e.g. "on_target"
    selector : optional filter fn(row)->bool to restrict the subset
    Returns  : (point_pct, lo_pct, hi_pct, n_samples, n_scenes)
    """
    if selector is not None:
        rows = [r for r in rows if selector(r)]
    if not rows:
        return (0.0, 0.0, 0.0, 0, 0)

    # group sample-outcomes by scene
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r.get("source_file", "unknown")].append(1.0 if r[key] else 0.0)
    scenes = list(by_scene.values())
    n_scenes = len(scenes)
    n_samples = sum(len(s) for s in scenes)

    point = 100.0 * sum(sum(s) for s in scenes) / n_samples

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    idx = np.arange(n_scenes)
    for b in range(n_boot):
        pick = rng.choice(idx, size=n_scenes, replace=True)   # resample SCENES
        hits = sum(sum(scenes[i]) for i in pick)
        tot = sum(len(scenes[i]) for i in pick)
        boot[b] = hits / tot if tot else 0.0
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
    return (point, lo, hi, n_samples, n_scenes)


def cluster_bootstrap_diff(rows, key, sel_a, sel_b, n_boot=10000, seed=42):
    """
    Bootstrap the DIFFERENCE in a proportion between two subsets, resampling
    scenes jointly so paired structure is preserved. Returns (diff_pct, lo, hi).
    A CI that excludes 0 => significant difference under clustering.
    """
    by_scene_a = defaultdict(list)
    by_scene_b = defaultdict(list)
    for r in rows:
        s = r.get("source_file", "unknown")
        if sel_a(r):
            by_scene_a[s].append(1.0 if r[key] else 0.0)
        if sel_b(r):
            by_scene_b[s].append(1.0 if r[key] else 0.0)

    scenes = list(set(by_scene_a) | set(by_scene_b))
    n_scenes = len(scenes)

    def rate(pick, table):
        hits = sum(sum(table[scenes[i]]) for i in pick if scenes[i] in table)
        tot = sum(len(table[scenes[i]]) for i in pick if scenes[i] in table)
        return hits / tot if tot else np.nan

    pa = rate(range(n_scenes), by_scene_a)
    pb = rate(range(n_scenes), by_scene_b)
    point = 100.0 * (pa - pb)

    rng = np.random.default_rng(seed)
    idx = np.arange(n_scenes)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=n_scenes, replace=True)
        ra, rb = rate(pick, by_scene_a), rate(pick, by_scene_b)
        if not (np.isnan(ra) or np.isnan(rb)):
            boot.append(100.0 * (ra - rb))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


def line(label, res):
    point, lo, hi, n, ns = res
    return f"  {label:<34}{point:5.1f}%  [{lo:4.1f}, {hi:4.1f}]   (n={n}, scenes={ns})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="grasp_per_sample.json")
    ap.add_argument("--out", default="grasp_cluster_ci.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    p = Path(args.data)
    if not p.exists():
        print(f"[ERROR] {p} not found. Add the per-sample dump to "
              f"grasp_success_proxy.py (see the two-line patch) and rerun it.")
        return
    rows = json.load(open(p))
    n_scenes = len({r.get("source_file", "unknown") for r in rows})
    print(f"[DATA] {len(rows)} samples across {n_scenes} scenes "
          f"({len(rows)/max(n_scenes,1):.1f} samples/scene)\n")

    print("=" * 72)
    print("CLUSTER-AWARE BOOTSTRAP 95% CIs (resampled by source scene)")
    print(f"n_boot={args.n_boot}")
    print("=" * 72)

    out = {}

    print("\nOVERALL")
    for key in ["on_target", "on_any"]:
        res = cluster_bootstrap(rows, key, n_boot=args.n_boot)
        print(line(key.upper(), res))
        out[f"overall_{key}"] = res

    print("\nBY SAMPLE TYPE — ON-TARGET")
    for t in ["real", "composite"]:
        res = cluster_bootstrap(rows, "on_target",
                                selector=lambda r, t=t: r.get("sample_type") == t,
                                n_boot=args.n_boot)
        print(line(t, res))
        out[f"{t}_on_target"] = res

    # the headline test: real - composite difference, clustered
    diff, dlo, dhi = cluster_bootstrap_diff(
        rows, "on_target",
        sel_a=lambda r: r.get("sample_type") == "real",
        sel_b=lambda r: r.get("sample_type") == "composite",
        n_boot=args.n_boot)
    verdict = "SIGNIFICANT (excludes 0)" if (dlo > 0 or dhi < 0) else "NOT significant (includes 0)"
    print(f"\n  {'real - composite DIFFERENCE':<34}{diff:5.1f}%  [{dlo:4.1f}, {dhi:4.1f}]   -> {verdict}")
    out["real_minus_composite_diff"] = {"diff": diff, "lo": dlo, "hi": dhi,
                                        "significant": bool(dlo > 0 or dhi < 0)}

    # size terciles — recompute thresholds from the data
    areas = np.array([r.get("area", 0.0) for r in rows])
    t1, t2 = np.percentile(areas, [33.3, 66.6])
    print(f"\nBY OBJECT SIZE — ON-TARGET  (terciles at {t1:.0f}px, {t2:.0f}px)")
    for name, sel in [
        (f"small (<{t1:.0f})",     lambda r: r.get("area", 0) < t1),
        (f"medium",                lambda r: t1 <= r.get("area", 0) < t2),
        (f"large (>={t2:.0f})",    lambda r: r.get("area", 0) >= t2),
    ]:
        res = cluster_bootstrap(rows, "on_target", selector=sel, n_boot=args.n_boot)
        print(line(name, res))
        out[f"size_{name}"] = res

    print("\n" + "=" * 72)
    print("HOW TO READ:")
    print("  These CIs are WIDER than the Wilson CIs in grasp_success_results_run2.json.")
    print("  That is correct — Wilson assumed 251 independent samples; you have ~75 scenes.")
    print("  Point estimates are identical; only certainty is honestly reduced.")
    print("  The real-composite DIFFERENCE CI is your headline test: if it excludes 0,")
    print("  native>composite holds even after accounting for scene correlation.")

    json.dump({k: (v if isinstance(v, dict) else
                   {"point": v[0], "lo": v[1], "hi": v[2], "n": v[3], "scenes": v[4]})
               for k, v in out.items()}, open(args.out, "w"), indent=2)
    print(f"\n[SAVED] {args.out}")


if __name__ == "__main__":
    main()