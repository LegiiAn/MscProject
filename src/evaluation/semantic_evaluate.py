"""
Semantic Reasoning Test — Step 2: Evaluate Instruction Following
================================================================
For each two-object test scene, runs the VLA with TWO prompts:
  - "Pick up the {object_a}"
  - "Pick up the {object_b}"

Measures whether the predicted grasp point is CLOSER to the named
object than to the other one. The instruction-following rate is the
fraction of times this holds. A random/instruction-blind model would
score 50%. Significantly above 50% = the model understands the
instruction. Significantly below = systematic confusion.

Also tests generic prompts:
  - "Pick up the recyclable item"
  - "Pick up the transparent object on the left"

Usage:
    python src/evaluation/semantic_evaluate.py
"""

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


def predict_point(model, processor, image_bgr, instruction, device):
    """Run VLA inference with a given instruction, return (px, py) or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prompt = f"In: What action should the robot take to pick up the {instruction.lower()}\nOut: "

    inputs = processor(
        text=prompt,
        images=image_rgb,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

    # Find the end of prompt for clean generation
    decoded = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    out_pos = decoded.rfind("Out:")
    if out_pos == -1:
        return None
    prompt_text = decoded[:out_pos + len("Out: ")]
    prompt_ids = processor.tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=prompt_ids,
            pixel_values=pixel_values,
            max_new_tokens=50,
            do_sample=False,
        )

    gen = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    vals = extract_action_values(gen)
    if vals is None:
        return None

    H, W = image_bgr.shape[:2]
    px = int((vals[0] / 255.0) * W)
    py = int((vals[1] / 255.0) * H)
    return px, py


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = Path.home() / "MscProject" / "data" / "semantic_test"
    CHECKPOINT = Path("vla_checkpoints/lora_best")
    if not CHECKPOINT.exists():
        cands = sorted(Path("vla_checkpoints").glob("lora_*"))
        CHECKPOINT = cands[-1] if cands else None
    if CHECKPOINT is None:
        print("[ERROR] No checkpoint found")
        return

    with open(DATA_DIR / "test_pairs.json") as f:
        records = json.load(f)
    print(f"[DATA] {len(records)} test pairs loaded")

    print(f"[INIT] Loading model from {CHECKPOINT}...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    model.eval()

    # ── Prompts to test ──────────────────────────────────────────────────
    # For each scene we test:
    #   1. "pick up the {cat_a}" — should point toward object A
    #   2. "pick up the {cat_b}" — should point toward object B
    #   3. "pick up the recyclable item" — semantic reasoning
    #   4. "pick up the transparent object on the {side_a}" — spatial+language

    results = []
    n_follow_specific = 0   # prompt names A, prediction closer to A
    n_total_specific = 0
    n_follow_spatial = 0    # prompt says "left"/"right", prediction on correct side
    n_total_spatial = 0
    parse_failures = 0
    gallery = []

    print(f"[EVAL] Running inference (4 prompts x {len(records)} scenes)...\n")

    for idx, rec in enumerate(records):
        img = cv2.imread(str(DATA_DIR / rec["image"]))
        if img is None:
            continue

        ta = np.array(rec["target_a"])
        tb = np.array(rec["target_b"])

        row = {
            "idx": idx,
            "cat_a": rec["cat_a"], "cat_b": rec["cat_b"],
            "side_a": rec["side_a"], "side_b": rec["side_b"],
            "target_a": rec["target_a"], "target_b": rec["target_b"],
            "predictions": {},
        }

        # --- Prompt 1: "pick up the {A}" ---
        pred_a = predict_point(model, processor, img, rec["prompt_a"], device)
        if pred_a is not None:
            row["predictions"]["prompt_a"] = list(pred_a)
            dist_to_a = np.linalg.norm(np.array(pred_a) - ta)
            dist_to_b = np.linalg.norm(np.array(pred_a) - tb)
            follows = dist_to_a < dist_to_b
            row["prompt_a_follows"] = bool(follows)
            n_follow_specific += int(follows)
            n_total_specific += 1
        else:
            parse_failures += 1

        # --- Prompt 2: "pick up the {B}" ---
        pred_b = predict_point(model, processor, img, rec["prompt_b"], device)
        if pred_b is not None:
            row["predictions"]["prompt_b"] = list(pred_b)
            dist_to_a = np.linalg.norm(np.array(pred_b) - ta)
            dist_to_b = np.linalg.norm(np.array(pred_b) - tb)
            follows = dist_to_b < dist_to_a
            row["prompt_b_follows"] = bool(follows)
            n_follow_specific += int(follows)
            n_total_specific += 1
        else:
            parse_failures += 1

        # --- Prompt 3: "pick up the recyclable item" ---
        pred_r = predict_point(model, processor, img, "recyclable item", device)
        if pred_r is not None:
            row["predictions"]["recyclable"] = list(pred_r)
        else:
            parse_failures += 1

        # --- Prompt 4: "pick up the transparent object on the {side_a}" ---
        pred_s = predict_point(model, processor, img,
                               f"transparent object on the {rec['side_a']}", device)
        if pred_s is not None:
            row["predictions"]["spatial"] = list(pred_s)
            # Check if prediction is on the correct side
            mid = img.shape[1] // 2
            if rec["side_a"] == "left":
                follows_spatial = pred_s[0] < mid
            else:
                follows_spatial = pred_s[0] > mid
            row["spatial_follows"] = bool(follows_spatial)
            n_follow_spatial += int(follows_spatial)
            n_total_spatial += 1
        else:
            parse_failures += 1

        results.append(row)

        # Gallery: save first 6 scenes
        if len(gallery) < 6 and pred_a is not None and pred_b is not None:
            gallery.append({
                "img": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                "ta": ta, "tb": tb,
                "pred_a": pred_a, "pred_b": pred_b,
                "cat_a": rec["cat_a"], "cat_b": rec["cat_b"],
            })

        if (idx + 1) % 20 == 0:
            rate = 100 * n_follow_specific / max(n_total_specific, 1)
            print(f"  [{idx+1}/{len(records)}] instruction-following: "
                  f"{n_follow_specific}/{n_total_specific} = {rate:.1f}%")

    # ── Report ───────────────────────────────────────────────────────────
    spec_rate = n_follow_specific / max(n_total_specific, 1)
    spec_lo, spec_hi = wilson_ci(n_follow_specific, n_total_specific)
    spat_rate = n_follow_spatial / max(n_total_spatial, 1)
    spat_lo, spat_hi = wilson_ci(n_follow_spatial, n_total_spatial)

    print("\n" + "=" * 66)
    print("SEMANTIC REASONING EVALUATION")
    print("=" * 66)
    print(f"Category-specific instruction following:")
    print(f"  {n_follow_specific}/{n_total_specific} = {100*spec_rate:.1f}%  "
          f"Wilson 95% CI [{100*spec_lo:.1f}%, {100*spec_hi:.1f}%]")
    print(f"  (50% = random/instruction-blind baseline)")
    print(f"\nSpatial instruction following (left/right):")
    print(f"  {n_follow_spatial}/{n_total_spatial} = {100*spat_rate:.1f}%  "
          f"Wilson 95% CI [{100*spat_lo:.1f}%, {100*spat_hi:.1f}%]")
    print(f"\nParse failures: {parse_failures}")
    print("=" * 66)

    output = {
        "category_specific": {
            "correct": n_follow_specific, "total": n_total_specific,
            "rate_pct": 100 * spec_rate,
            "wilson_ci_95": [100 * spec_lo, 100 * spec_hi],
        },
        "spatial": {
            "correct": n_follow_spatial, "total": n_total_spatial,
            "rate_pct": 100 * spat_rate,
            "wilson_ci_95": [100 * spat_lo, 100 * spat_hi],
        },
        "parse_failures": parse_failures,
        "n_scenes": len(records),
        "per_scene": results,
    }
    with open("semantic_reasoning_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[SAVED] semantic_reasoning_results.json")

    # ── Gallery ──────────────────────────────────────────────────────────
    if gallery:
        n = len(gallery)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), dpi=200)
        if n == 1:
            axes = axes.reshape(2, 1)

        for col, g in enumerate(gallery):
            # Top row: "pick up A" prompt
            axes[0, col].imshow(g["img"])
            axes[0, col].plot(*g["pred_a"], "x", color="red", markersize=12, markeredgewidth=3)
            axes[0, col].plot(*g["ta"], "+", color="lime", markersize=10, markeredgewidth=2)
            axes[0, col].plot(*g["tb"], "+", color="cyan", markersize=10, markeredgewidth=2)
            axes[0, col].set_title(f'"Pick up the {g["cat_a"]}"', fontsize=9, fontweight="bold")
            axes[0, col].axis("off")

            # Bottom row: "pick up B" prompt
            axes[1, col].imshow(g["img"])
            axes[1, col].plot(*g["pred_b"], "x", color="red", markersize=12, markeredgewidth=3)
            axes[1, col].plot(*g["ta"], "+", color="lime", markersize=10, markeredgewidth=2)
            axes[1, col].plot(*g["tb"], "+", color="cyan", markersize=10, markeredgewidth=2)
            axes[1, col].set_title(f'"Pick up the {g["cat_b"]}"', fontsize=9, fontweight="bold")
            axes[1, col].axis("off")

        fig.suptitle("Semantic Instruction Following\n"
                     "Red ✕ = prediction | Green + = object A target | Cyan + = object B target",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig("semantic_reasoning_gallery.pdf", format="pdf", bbox_inches="tight")
        print("[SAVED] semantic_reasoning_gallery.pdf")


if __name__ == "__main__":
    main()
