"""
Grasp-Success Measurement v3 (Run 2)
====================================
Mask-exact scoring at any resolution, with two diagnostic breakdowns:

  - per SAMPLE TYPE : real ClearGrasp frames vs composites
                      (does the model localise better on authentic
                       transparent-object cues?)
  - per OBJECT SIZE : small / medium / large target-area terciles
                      (Fix 4: is failure concentrated on small objects?)

Criteria (per sample):
  ON-TARGET  : prediction inside the selected target-object mask
  ON-OBJECT  : prediction inside any object mask
  PROXIMITY  : prediction within 6.25% of frame width of the GT point
               (= the Run-1 40px@640 criterion, scale-adjusted)

Run on cluster:
    python src/evaluation/grasp_success_proxy.py
"""

import os
import sys
import re
import json
import numpy as np
import cv2
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.vla_dataset import OpenVLADataset


def extract_action_values(text):
    out_pos = text.rfind("Out:")
    after = text[out_pos:] if out_pos != -1 else text
    matches = re.findall(r"(\d+)>", after)
    vals = [int(v) for v in matches if 0 <= int(v) <= 255]
    return vals if len(vals) >= 2 else None


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def fmt_rate(label, k, n):
    lo, hi = wilson_ci(k, n)
    return (f"{label:<28}{k:>4}/{n:<4} = {100*k/max(n,1):5.1f}%  "
            f"CI [{100*lo:.1f}%, {100*hi:.1f}%]")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = Path("data/processed_dataset")

    # Point at the best Run-2 checkpoint (edit after eval identifies it)
    CHECKPOINT = Path("vla_checkpoints/lora_best")
    if not CHECKPOINT.exists():
        cands = sorted(Path("vla_checkpoints").glob("lora_*"))
        if not cands:
            print("[ERROR] no checkpoints found")
            return
        CHECKPOINT = cands[-1]

    MAX_SAMPLES = 300

    print("[INIT] Loading processor and dataset...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    tokenizer = processor.tokenizer

    val_dataset = OpenVLADataset(data_dir=DATA_DIR, processor=processor, split="val")
    with open(DATA_DIR / "annotations.json") as f:
        val_records = json.load(f)["val"]

    if "mask_target" not in val_records[0]:
        print("[ERROR] annotations lack 'mask_target' — regenerate with v4 generator.")
        return

    print(f"[INIT] Loading model from {CHECKPOINT.name}...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT)
    model.eval()

    rows = []          # per-sample dicts
    parse_failures = 0
    gallery = {"TARGET HIT": [], "OTHER OBJECT": [], "MISS": []}

    n_total = min(MAX_SAMPLES, len(val_dataset))
    print(f"[EVAL] Processing {n_total} samples...")

    with torch.no_grad():
        for idx in range(n_total):
            record = val_records[idx]
            batch = val_dataset[idx]

            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)

            tgt_mask = cv2.imread(str(DATA_DIR / record["mask_target"]), cv2.IMREAD_GRAYSCALE)
            full_mask = cv2.imread(str(DATA_DIR / record["mask_full"]), cv2.IMREAD_GRAYSCALE)
            composite = cv2.imread(str(DATA_DIR / record["image"]))
            if tgt_mask is None or full_mask is None or composite is None:
                continue

            H, W = tgt_mask.shape
            prox_radius = 0.0625 * W          # = 40px at 640 -> 14px at 224

            decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            out_pos = decoded.rfind("Out:")
            if out_pos == -1:
                continue
            prompt_ids = tokenizer.encode(
                decoded[:out_pos + len("Out: ")], return_tensors="pt").to(device)

            outputs = model.generate(
                input_ids=prompt_ids, pixel_values=pixel_values,
                max_new_tokens=50, do_sample=False,
            )
            pred = extract_action_values(
                tokenizer.decode(outputs[0], skip_special_tokens=False))
            if pred is None:
                parse_failures += 1
                continue

            px = int((pred[0] / 255.0) * W)
            py = int((pred[1] / 255.0) * H)
            pxc, pyc = int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))
            gx, gy = record["click_2d"]

            dist = float(np.hypot(px - gx, py - gy))
            rows.append({
                "on_target": bool(tgt_mask[pyc, pxc] > 0),
                "on_any":    bool(full_mask[pyc, pxc] > 0),
                "proximity": bool(dist < prox_radius),
                "gt_valid":  bool(tgt_mask[int(np.clip(gy, 0, H-1)),
                                           int(np.clip(gx, 0, W-1))] > 0),
                "dist": dist,
                "sample_type": record.get("sample_type", "composite"),
                "area": float(record.get("target_area_px", 0.0)),
                "source_file": record.get("source_file", "unknown"), 
            })

            bucket = ("TARGET HIT" if rows[-1]["on_target"]
                      else "OTHER OBJECT" if rows[-1]["on_any"] else "MISS")
            if len(gallery[bucket]) < 2:
                gallery[bucket].append({
                    "img": cv2.cvtColor(composite, cv2.COLOR_BGR2RGB),
                    "tgt": tgt_mask, "full": full_mask,
                    "pred": (px, py), "gt": (gx, gy),
                })

    n = len(rows)
    if n == 0:
        print("[ERROR] No samples evaluated.")
        return

    def block(subrows, title):
        m = len(subrows)
        kt = sum(r["on_target"] for r in subrows)
        ka = sum(r["on_any"] for r in subrows)
        kp = sum(r["proximity"] for r in subrows)
        print(f"\n--- {title} (n={m}) ---")
        print(fmt_rate("ON-TARGET:", kt, m))
        print(fmt_rate("ON-OBJECT:", ka, m))
        print(fmt_rate("PROXIMITY:", kp, m))
        if m:
            print(f"{'Mean pred-GT dist:':<28}"
                  f"{np.mean([r['dist'] for r in subrows]):.1f} px")
        return {"n": m, "on_target": kt, "on_any": ka, "proximity": kp,
                "on_target_ci": [100*x for x in wilson_ci(kt, m)],
                "on_any_ci":    [100*x for x in wilson_ci(ka, m)]}

    print("\n" + "=" * 66)
    print(f"GRASP-SUCCESS MEASUREMENT v3 — checkpoint {CHECKPOINT.name}")
    print("=" * 66)
    overall = block(rows, "OVERALL")
    gt_valid = sum(r["gt_valid"] for r in rows)
    print(f"\n{'GT label validity:':<28}{gt_valid}/{n} "
          f"= {100*gt_valid/n:.1f}%  (must be 100%)")
    print(f"{'Parse failures:':<28}{parse_failures}")

    # Breakdown: sample type
    results = {"overall": overall,
               "gt_validity_pct": 100 * gt_valid / n,
               "parse_failures": parse_failures,
               "checkpoint": CHECKPOINT.name,
               "by_type": {}, "by_size": {}}
    for t in ["real", "composite"]:
        sub = [r for r in rows if r["sample_type"] == t]
        if sub:
            results["by_type"][t] = block(sub, f"SAMPLE TYPE: {t.upper()}")

    # Breakdown: object size terciles
    areas = np.array([r["area"] for r in rows])
    t1, t2 = np.percentile(areas, [33.3, 66.6])
    buckets = {
        f"SMALL (area<{t1:.0f}px)":  [r for r in rows if r["area"] < t1],
        f"MEDIUM":                   [r for r in rows if t1 <= r["area"] < t2],
        f"LARGE (area>={t2:.0f}px)": [r for r in rows if r["area"] >= t2],
    }
    for name, sub in buckets.items():
        if sub:
            results["by_size"][name] = block(sub, f"OBJECT SIZE: {name}")

    print("=" * 66)
    with open("grasp_success_results_run2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[SAVED] grasp_success_results_run2.json")
    with open("grasp_per_sample.json", "w") as f:
        json.dump(rows, f)
    print("[SAVED] grasp_per_sample.json")

    # Gallery
    samples = [(k, s) for k in ["TARGET HIT", "OTHER OBJECT", "MISS"] for s in gallery[k]]
    if samples:
        m = len(samples)
        fig, axes = plt.subplots(1, m, figsize=(4 * m, 4), dpi=200)
        if m == 1:
            axes = [axes]
        colors = {"TARGET HIT": "green", "OTHER OBJECT": "orange", "MISS": "red"}
        for ax, (label, s) in zip(axes, samples):
            ax.imshow(s["img"])
            for msk, c in [(s["full"], "cyan"), (s["tgt"], "yellow")]:
                cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    cc = cnt.squeeze()
                    if cc.ndim == 2 and len(cc) > 2:
                        ax.plot(cc[:, 0], cc[:, 1], color=c, linewidth=1.2)
            ax.plot(*s["pred"], "x", color="red", markersize=13, markeredgewidth=3)
            ax.plot(*s["gt"], "+", color="lime", markersize=13, markeredgewidth=3)
            ax.set_title(label, fontsize=12, fontweight="bold", color=colors[label])
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("grasp_success_gallery_run2.pdf", format="pdf", bbox_inches="tight")
        print("[SAVED] grasp_success_gallery_run2.pdf")


if __name__ == "__main__":
    main()