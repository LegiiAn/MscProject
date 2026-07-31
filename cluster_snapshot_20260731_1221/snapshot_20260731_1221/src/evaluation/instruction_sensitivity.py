"""
Instruction-Sensitivity Evaluation v2  (Run 2)
==============================================
FIX over v1: v1 conditioned ON-TARGET on successful parse, which (a) biased the
rates across differently-sized self-selected subsets and (b) collapsed the paired
McNemar sample to only the images that parsed under ALL conditions (n~19, no power).

v2 treats a parse failure as what it operationally is: a MISS. If the model emits
no valid coordinate, the grasp cannot happen. So:

  PARSE-FAILURE RATE   is reported as a first-class result   (the brittleness finding)
  ON-TARGET (primary)  = hits / ALL attempts, parse-fail = miss  (Wilson CI)
  ON-TARGET | parsed   = hits / parsed, reported but CAVEATED as conditional
  McNEMAR              runs on the FULL paired set (parse-fail=miss) -> real power

Also: --dump writes the raw model output for the first failures per condition so
you can SEE what it emits when it breaks (fluent text? repeats? EOS?) instead of
guessing in the write-up.

Run on the cluster:
    python src/evaluation/instruction_sensitivity.py --n 200 --split val --dump
"""

import os
import sys
import re
import json
import argparse
from math import comb
from pathlib import Path

import numpy as np
import cv2
import torch

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.training.vla_dataset import OpenVLADataset


CONDITIONS = {
    "baseline":      "Pick up the transparent plastic bottle.",
    "paraphrase_1":  "Grasp the clear plastic bottle.",
    "paraphrase_2":  "Pick up the see-through bottle.",
    "paraphrase_3":  "Select the transparent container.",
    "absent_can":    "Pick up the metal can.",
    "absent_cube":   "Pick up the red cube.",
    "null_generic":  "Pick up the object.",
}
DATASET_INSTRUCTION_FALLBACK = "Pick up the transparent plastic bottle."


# ----------------------------- pure helpers --------------------------------
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


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def swap_instruction(prompt_text, old_instr, new_instr):
    candidates = [old_instr, old_instr.lower(),
                  old_instr.rstrip("."), old_instr.lower().rstrip(".")]
    low = prompt_text.lower()
    for cand in candidates:
        if not cand:
            continue
        idx = low.find(cand.lower())
        if idx == -1:
            continue
        matched = prompt_text[idx:idx + len(cand)]
        repl = new_instr
        if matched == matched.lower():
            repl = new_instr.lower()
        if cand.endswith(".") and not repl.endswith("."):
            repl += "."
        if not cand.endswith(".") and repl.endswith("."):
            repl = repl.rstrip(".")
        return prompt_text[:idx] + repl + prompt_text[idx + len(cand):], matched
    return None, None


def build_prompt_text(decoded_full):
    out_pos = decoded_full.rfind("Out:")
    return None if out_pos == -1 else decoded_full[:out_pos + len("Out: ")]


def paired_mcnemar(base_map, cond_map, idxs):
    """b = base hit & cond miss ; c = base miss & cond hit, over shared idxs."""
    b = sum(1 for i in idxs if base_map[i] and not cond_map[i])
    c = sum(1 for i in idxs if not base_map[i] and cond_map[i])
    return b, c, mcnemar_exact(b, c)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data", default="data/processed_dataset")
    ap.add_argument("--checkpoint", default="vla_checkpoints/lora_best")
    ap.add_argument("--out", default="instruction_sensitivity_v2.json")
    ap.add_argument("--dump", action="store_true",
                    help="save raw model output for the first failures per condition")
    ap.add_argument("--dump-k", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(args.data)

    CHECKPOINT = Path(args.checkpoint)
    if not CHECKPOINT.exists():
        cands = sorted(Path("vla_checkpoints").glob("lora_*"))
        if not cands:
            print("[ERROR] no checkpoints"); return
        CHECKPOINT = cands[-1]

    print("[INIT] processor + dataset...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    tokenizer = processor.tokenizer
    dataset = OpenVLADataset(data_dir=DATA_DIR, processor=processor, split=args.split)
    with open(DATA_DIR / "annotations.json") as f:
        records = json.load(f)[args.split]

    print(f"[INIT] model {CHECKPOINT.name} (4-bit)...")
    base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    model.eval()

    n_total = min(args.n, len(dataset))

    # self-check
    b0 = dataset[0]
    prompt0 = build_prompt_text(tokenizer.decode(b0["input_ids"], skip_special_tokens=False))
    instr0 = records[0].get("instruction", DATASET_INSTRUCTION_FALLBACK)
    if prompt0 is None or swap_instruction(prompt0, instr0, "X")[0] is None:
        print("[ABORT] prompt self-check failed; prompt was:"); print(repr(prompt0)); return
    print(f"\n[SELF-CHECK] template : {prompt0!r}")
    print(f"[SELF-CHECK] example  : {swap_instruction(prompt0, instr0, 'Grasp the clear bottle.')[0]!r}\n")

    # outcome[cond][idx] = dict(parsed, on_target, on_any)   (on_* False if not parsed)
    outcome = {c: {} for c in CONDITIONS}
    failures = {c: [] for c in CONDITIONS}
    attempted = []  # idxs that passed mask-loading (shared across all conditions)

    print(f"[EVAL] {n_total} samples x {len(CONDITIONS)} conditions...")
    with torch.no_grad():
        for idx in range(n_total):
            record = records[idx]
            batch = dataset[idx]
            tgt = cv2.imread(str(DATA_DIR / record["mask_target"]), cv2.IMREAD_GRAYSCALE)
            full = cv2.imread(str(DATA_DIR / record["mask_full"]), cv2.IMREAD_GRAYSCALE)
            if tgt is None or full is None:
                continue
            H, W = tgt.shape
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            prompt_text = build_prompt_text(
                tokenizer.decode(batch["input_ids"], skip_special_tokens=False))
            if prompt_text is None:
                continue
            instr = record.get("instruction", DATASET_INSTRUCTION_FALLBACK)
            attempted.append(idx)

            for cond, new_instr in CONDITIONS.items():
                swapped, _ = swap_instruction(prompt_text, instr, new_instr)
                prompt_ids = tokenizer.encode(swapped, return_tensors="pt").to(device)
                out = model.generate(input_ids=prompt_ids, pixel_values=pixel_values,
                                     max_new_tokens=50, do_sample=False)
                raw = tokenizer.decode(out[0], skip_special_tokens=False)
                pred = extract_action_values(raw)
                if pred is None:
                    outcome[cond][idx] = {"parsed": False, "on_target": False, "on_any": False}
                    if args.dump and len(failures[cond]) < args.dump_k:
                        tail = raw[raw.rfind("Out:"):][:200] if "Out:" in raw else raw[:200]
                        failures[cond].append({"idx": idx, "raw_tail": tail})
                    continue
                px = int(np.clip((pred[0] / 255.0) * W, 0, W - 1))
                py = int(np.clip((pred[1] / 255.0) * H, 0, H - 1))
                outcome[cond][idx] = {"parsed": True,
                                      "on_target": bool(tgt[py, px] > 0),
                                      "on_any": bool(full[py, px] > 0)}
            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{n_total}")

    A = len(attempted)
    if A == 0:
        print("[ERROR] nothing attempted"); return

    # --------------------------- report ------------------------------------
    def counts(cond):
        o = outcome[cond]
        parsed = sum(1 for i in attempted if o[i]["parsed"])
        ot_all = sum(1 for i in attempted if o[i]["on_target"])                 # fail=miss
        ot_par = sum(1 for i in attempted if o[i]["parsed"] and o[i]["on_target"])
        return parsed, ot_all, ot_par

    print("\n" + "=" * 92)
    print(f"INSTRUCTION SENSITIVITY v2 — {CHECKPOINT.name} — {args.split} — attempted n={A}")
    print("=" * 92)
    print(f"{'condition':<15}{'instruction':<34}{'PARSE-FAIL':>14}{'ON-TARGET(all)':>18}{'|parsed':>10}")
    print("-" * 92)
    summary = {}
    for cond in CONDITIONS:
        parsed, ot_all, ot_par = counts(cond)
        fail = A - parsed
        flo, fhi = wilson_ci(fail, A)
        olo, ohi = wilson_ci(ot_all, A)
        print(f"{cond:<15}{CONDITIONS[cond][:32]:<34}"
              f"{fail:>3}/{A} {100*fail/A:4.0f}%   "
              f"{ot_all:>2}/{A} {100*ot_all/A:4.1f}% [{100*olo:.0f},{100*ohi:.0f}] "
              f"{(100*ot_par/parsed if parsed else 0):5.1f}%")
        summary[cond] = {
            "instruction": CONDITIONS[cond],
            "attempted": A, "parsed": parsed,
            "parse_fail": fail, "parse_fail_pct": 100*fail/A,
            "parse_fail_ci": [100*flo, 100*fhi],
            "on_target_all": ot_all, "on_target_all_pct": 100*ot_all/A,
            "on_target_all_ci": [100*olo, 100*ohi],
            "on_target_given_parsed_pct": (100*ot_par/parsed if parsed else 0),
        }

    # McNemar on ON-TARGET (all attempts, fail=miss) — full power
    print("\n" + "-" * 92)
    print(f"PAIRED McNEMAR vs baseline — ON-TARGET, parse-fail=miss (paired n={A})")
    print("-" * 92)
    base_ot = {i: outcome["baseline"][i]["on_target"] for i in attempted}
    base_fail = {i: (not outcome["baseline"][i]["parsed"]) for i in attempted}
    mc = {}
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        cond_ot = {i: outcome[cond][i]["on_target"] for i in attempted}
        b, c, p = paired_mcnemar(base_ot, cond_ot, attempted)
        arrow = "WORSE" if b > c else ("better" if c > b else "equal")
        sig = "*" if p < 0.05 else " "
        print(f"  {cond:<15} b={b:<3} c={c:<3} p={p:.3f} {sig}  ({arrow} than baseline)")
        mc[cond] = {"metric": "on_target_all", "b": b, "c": c, "p": p}

    # McNemar on PARSE-FAILURE — does instruction change the break rate?
    print("\n" + "-" * 92)
    print(f"PAIRED McNEMAR vs baseline — PARSE-FAILURE (paired n={A})")
    print("-" * 92)
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        cond_fail = {i: (not outcome[cond][i]["parsed"]) for i in attempted}
        b, c, p = paired_mcnemar(base_fail, cond_fail, attempted)
        arrow = "MORE failures" if c > b else ("fewer" if b > c else "equal")
        sig = "*" if p < 0.05 else " "
        print(f"  {cond:<15} b={b:<3} c={c:<3} p={p:.3f} {sig}  ({arrow} than baseline)")
        mc.setdefault(cond, {})["parsefail_b"] = b
        mc[cond]["parsefail_c"] = c
        mc[cond]["parsefail_p"] = p

    print("=" * 92)
    print("HOW TO READ:")
    print("  PARSE-FAIL rising with distance from training phrase = single-template")
    print("    LoRA broke action GENERATION (catastrophic forgetting) — likely your headline.")
    print("  ON-TARGET(all) is the honest operational rate. |parsed is conditional (biased); report both, lead with (all).")
    print("  '*' = p<0.05.")

    with open(args.out, "w") as f:
        json.dump({"checkpoint": CHECKPOINT.name, "split": args.split,
                   "attempted": A, "summary": summary, "mcnemar_vs_baseline": mc}, f, indent=2)
    print(f"[SAVED] {args.out}")

    if args.dump:
        dump_path = args.out.replace(".json", "_failures.txt")
        with open(dump_path, "w") as f:
            for cond in CONDITIONS:
                f.write(f"\n{'='*70}\n{cond}: {CONDITIONS[cond]}\n{'='*70}\n")
                for ex in failures[cond]:
                    f.write(f"[idx {ex['idx']}] ...{ex['raw_tail']!r}\n")
        print(f"[SAVED] {dump_path}  <- read this to see WHAT the model emits on failure")


if __name__ == "__main__":
    main()