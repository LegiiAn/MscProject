"""
ZeroWaste-f Real-World Evaluation  (Run 2)
==========================================
Answers Maria's request: does ANYTHING transfer to real materials-recovery
conveyor images? This is a DOMAIN-GAP measurement, not a semantics test
(semantics was already measured in-distribution by instruction_sensitivity_v2).

Data (confirmed from the dataset's own reco_org/*.py my_classes list):
  RGB   : data/zerowaste/data/splits_final_deblurred/val/data/<stem>.PNG  1920x1080
  masks : data/zerowaste/data/splits_final_deblurred/val/sem_seg/<stem>.PNG uint8
  classes: 0=bg 1=rigid_plastic 2=cardboard 3=metal 4=soft_plastic

METRIC — identical to grasp_success_proxy.py so numbers stay comparable:
  ON-TARGET : predicted pixel inside the TARGET-class mask
  ON-OBJECT : predicted pixel inside ANY non-background mask
  parse-fail = MISS (no coordinate emitted -> no grasp)

HONESTY GUARDS:
  - ON-TARGET only defined on frames CONTAINING the target class. Each condition
    is filtered to target-present frames and reports that n (rigid ~33%, cardboard ~98%).
  - Three baselines recomputed ON THIS DATA: centre-prior, uniform-random,
    random-on-any-object.
  - Inference at 224x224 (resolution the model runs at); predicted (x,y) scaled
    back to 1920x1080 and scored against NATIVE masks.
  - Prompt built identically to vla_dataset.py: "...to {instr.lower()}\nOut: ".

PREDICTION (write before reading): on-target at or below centre-prior on the
plastic conditions; parse-fail high (>70%).

Run:
    python src/evaluation/zerowaste_eval.py --n 40 --dump    # smoke test first
    python src/evaluation/zerowaste_eval.py --n 572          # full
"""

import sys
import re
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = {0: "bg", 1: "rigid_plastic", 2: "cardboard", 3: "metal", 4: "soft_plastic"}

# condition -> (instruction, target_class_id)
CONDITIONS = {
    "train_phrase":  ("pick up the transparent plastic bottle", 1),
    "native_phrase": ("pick up the rigid plastic",              1),
    "generic":       ("pick up the plastic",                    1),
    "cardboard":     ("pick up the cardboard",                  2),
}

VAL = "data/zerowaste/data/splits_final_deblurred/val"


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


def build_prompt(instruction):
    return f"In: What action should the robot take to {instruction.lower()}\nOut: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=572)
    ap.add_argument("--checkpoint", default="vla_checkpoints/lora_best")
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--out", default="zerowaste_eval_results.json")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = Path(args.val)
    rgb_dir, seg_dir = val / "data", val / "sem_seg"

    stems = sorted(p.stem for p in rgb_dir.glob("*.PNG")) or \
            sorted(p.stem for p in rgb_dir.glob("*.png"))
    stems = stems[:args.n]
    print(f"[DATA] {len(stems)} val frames")

    CHECKPOINT = Path(args.checkpoint)
    print(f"[INIT] processor + model {CHECKPOINT.name} (4-bit)...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    tokenizer = processor.tokenizer
    base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    model.eval()

    unique_instructions = sorted({v[0] for v in CONDITIONS.values()})
    out = {c: {} for c in CONDITIONS}
    base_hits = {c: {"centre": 0, "unif": 0, "onobj": 0, "n": 0} for c in CONDITIONS}
    failures = {c: [] for c in CONDITIONS}

    # self-check on first frame
    print(f"[SELF-CHECK] example prompt: {build_prompt(unique_instructions[0])!r}")

    print(f"[EVAL] {len(stems)} frames x {len(unique_instructions)} instructions...")
    with torch.no_grad():
        for fi, stem in enumerate(stems):
            bgr = cv2.imread(str(rgb_dir / f"{stem}.PNG"), cv2.IMREAD_COLOR)
            seg = cv2.imread(str(seg_dir / f"{stem}.PNG"), cv2.IMREAD_UNCHANGED)
            if bgr is None or seg is None:
                continue
            Hf, Wf = seg.shape[:2]
            any_obj = (seg > 0)

            rgb224 = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224))
            pil = Image.fromarray(rgb224)

            # one inference per unique instruction on this frame
            frame_pred, frame_raw = {}, {}
            for instruction in unique_instructions:
                inputs = processor(text=build_prompt(instruction), images=pil,
                                   return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
                dec = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                op = dec.rfind("Out:")
                prompt_ids = tokenizer.encode(dec[:op + len("Out: ")],
                                              return_tensors="pt").to(device)
                gen = model.generate(input_ids=prompt_ids, pixel_values=pixel_values,
                                     max_new_tokens=50, do_sample=False)
                raw = tokenizer.decode(gen[0], skip_special_tokens=False)
                vals = extract_action_values(raw)
                if vals is None:
                    frame_pred[instruction] = None
                    frame_raw[instruction] = raw[raw.rfind("Out:"):][:160]
                else:
                    px = int(np.clip((vals[0] / 255.0) * Wf, 0, Wf - 1))
                    py = int(np.clip((vals[1] / 255.0) * Hf, 0, Hf - 1))
                    frame_pred[instruction] = (px, py)

            cxc, cyc = Wf // 2, Hf // 2
            obj_ys, obj_xs = np.where(any_obj)
            for cond, (instruction, tcls) in CONDITIONS.items():
                tgt = (seg == tcls)
                if not tgt.any():
                    continue  # target absent -> ON-TARGET undefined; skip honestly
                pred = frame_pred[instruction]
                if pred is None:
                    out[cond][stem] = {"parsed": False, "on_target": False, "on_any": False}
                    if args.dump and len(failures[cond]) < 8:
                        failures[cond].append({"stem": stem, "raw": frame_raw.get(instruction, "")})
                else:
                    px, py = pred
                    out[cond][stem] = {"parsed": True,
                                       "on_target": bool(tgt[py, px]),
                                       "on_any": bool(any_obj[py, px])}
                bh = base_hits[cond]; bh["n"] += 1
                bh["centre"] += int(tgt[cyc, cxc])
                bh["unif"]   += int(tgt[int(rng.integers(0, Hf)), int(rng.integers(0, Wf))])
                if len(obj_xs):
                    j = int(rng.integers(0, len(obj_xs)))
                    bh["onobj"] += int(tgt[obj_ys[j], obj_xs[j]])

            if (fi + 1) % 25 == 0:
                print(f"  {fi+1}/{len(stems)}")

    def rate(k, n):
        lo, hi = wilson_ci(k, n)
        return f"{k:>3}/{n:<4}{100*k/max(n,1):5.1f}% [{100*lo:.0f},{100*hi:.0f}]"

    print("\n" + "=" * 92)
    print(f"ZEROWASTE-f REAL-WORLD EVALUATION — {CHECKPOINT.name}")
    print("=" * 92)
    print(f"{'condition':<14}{'target':<14}{'PARSE-FAIL':>16}{'ON-TARGET':>20}{'ON-OBJECT':>20}")
    print("-" * 92)
    summary = {}
    for cond, (instruction, tcls) in CONDITIONS.items():
        o = out[cond]; ss = list(o.keys()); n = len(ss)
        parsed = sum(o[s]["parsed"] for s in ss)
        ot = sum(o[s]["on_target"] for s in ss)
        oa = sum(o[s]["on_any"] for s in ss)
        fail = n - parsed
        print(f"{cond:<14}{CLASS_NAMES[tcls]:<14}{rate(fail,n):>16}{rate(ot,n):>20}{rate(oa,n):>20}")
        summary[cond] = {
            "instruction": instruction, "target_class": CLASS_NAMES[tcls],
            "n_target_present": n, "parsed": parsed,
            "parse_fail": fail, "parse_fail_pct": 100*fail/max(n,1),
            "on_target": ot, "on_target_pct": 100*ot/max(n,1),
            "on_target_ci": [100*x for x in wilson_ci(ot, n)],
            "on_object": oa, "on_object_pct": 100*oa/max(n,1),
        }

    print("\n" + "-" * 92)
    print("BASELINES (recomputed on ZeroWaste, over each condition's target-present frames)")
    print(f"{'condition':<14}{'centre-prior':>18}{'uniform-rand':>18}{'rand-on-object':>20}")
    print("-" * 92)
    for cond in CONDITIONS:
        bh = base_hits[cond]; n = bh["n"]
        print(f"{cond:<14}{rate(bh['centre'],n):>18}{rate(bh['unif'],n):>18}{rate(bh['onobj'],n):>20}")
        summary[cond]["baselines"] = {
            "n": n, "centre_prior_pct": 100*bh["centre"]/max(n,1),
            "uniform_pct": 100*bh["unif"]/max(n,1),
            "on_object_pct": 100*bh["onobj"]/max(n,1)}

    print("=" * 92)
    print("READ IT LIKE THIS:")
    print("  ON-TARGET at/below centre-prior -> the policy does NOT transfer to real images (expected).")
    print("  Compare train_phrase vs native_phrase vs generic: does phrasing matter on real data?")
    print("  cardboard is the one honest discrimination datapoint (a 2nd class actually present).")

    with open(args.out, "w") as f:
        json.dump({"checkpoint": CHECKPOINT.name, "n_frames": len(stems),
                   "summary": summary}, f, indent=2)
    print(f"[SAVED] {args.out}")

    if args.dump:
        dp = args.out.replace(".json", "_failures.txt")
        with open(dp, "w") as f:
            for cond in CONDITIONS:
                f.write(f"\n=== {cond}: {CONDITIONS[cond][0]} ===\n")
                for ex in failures[cond]:
                    f.write(f"[{ex['stem']}] {ex['raw']!r}\n")
        print(f"[SAVED] {dp}  <- what the model emits on failure")


if __name__ == "__main__":
    main()