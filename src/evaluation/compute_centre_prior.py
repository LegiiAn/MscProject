#!/usr/bin/env python3
"""
compute_centre_prior.py
=======================
Fills the blank "Run 2 centre prior" bar on the training/precision slide.

WHAT IT COMPUTES
A "centre prior" is the trivial baseline that ALWAYS predicts the image centre,
i.e. action bin (128, 128) on the 256-bin grid. Its mean spatial L1 error tells
you the score your model must BEAT to be doing anything more than guessing centre.

For every val sample we already stored the ground-truth action tokens
(action_tokens[0], action_tokens[1] = the x,y bins). The centre prior predicts
128,128 for all of them. Mean L1 error (in bins), averaged over x and y like your
eval does, is:

    L1_i = ( |128 - gt_x_i| + |128 - gt_y_i| ) / 2
    prior = mean_i L1_i

This matches how grasp/eval reports "mean spatial L1 in bins", so the number is
directly comparable to your model's 20.59 bins.

No GPU, no model. Reads only annotations.json.

Usage:
    python compute_centre_prior.py
    python compute_centre_prior.py --split val --centre 128
"""

import json
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--centre", type=int, default=128,
                    help="centre bin on the 0-255 grid (128 = image centre)")
    ap.add_argument("--out", default="centre_prior.json")
    args = ap.parse_args()

    ann = json.load(open(Path(args.data) / "annotations.json"))[args.split]

    gx = np.array([r["action_tokens"][0] for r in ann], dtype=float)
    gy = np.array([r["action_tokens"][1] for r in ann], dtype=float)

    c = args.centre
    # per-sample mean-of-axes L1, matching the eval's "mean bin error" convention
    l1_meanaxes = (np.abs(c - gx) + np.abs(c - gy)) / 2.0
    # also report per-axis and the joint (x+y summed) in case the eval used that
    l1_x = np.abs(c - gx)
    l1_y = np.abs(c - gy)

    n = len(ann)
    print("=" * 60)
    print(f"CENTRE-OF-IMAGE PRIOR  —  split={args.split}, centre bin={c}, n={n}")
    print("=" * 60)
    print(f"  mean L1 (avg of x,y axes) : {l1_meanaxes.mean():6.2f} bins   <- compare to model 20.59")
    print(f"  mean L1 (x axis)          : {l1_x.mean():6.2f} bins")
    print(f"  mean L1 (y axis)          : {l1_y.mean():6.2f} bins")
    print(f"  median L1 (avg of x,y)    : {np.median(l1_meanaxes):6.2f} bins")
    print("=" * 60)
    print("Drop the 'mean L1 (avg of x,y axes)' value into the centre-prior figure")
    print("(FB['centre_prior']['Run 2 centre prior'] = <value>), or it is saved below.")

    out = {
        "split": args.split, "centre_bin": c, "n": n,
        "centre_prior_mean_l1_bins": float(l1_meanaxes.mean()),
        "centre_prior_median_l1_bins": float(np.median(l1_meanaxes)),
        "centre_prior_l1_x": float(l1_x.mean()),
        "centre_prior_l1_y": float(l1_y.mean()),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()