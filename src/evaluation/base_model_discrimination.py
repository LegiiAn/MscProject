#!/usr/bin/env python3
"""
Step 1 - Base-Model Instruction-Discrimination Harness  (control for Run 3)
===========================================================================
PURPOSE
  Attribute the fine-tuned model's instruction-collapse to the FINE-TUNING, not
  to the synthetic domain. The fine-tuned model is instruction-insensitive on
  its OWN training distribution (composites). This harness asks whether the BASE
  OpenVLA-7B is instruction-SENSITIVE, so the collapse can be blamed on training
  rather than on the data being OOD-for-everything.

WHY THIS IS NOT AN ON-TARGET TEST
  Base OpenVLA-7B does NOT emit a 2D image point. It emits a native 7-DoF
  end-effector action (x,y,z,roll,pitch,yaw,gripper) via model.predict_action().
  There is no pixel to score against your masks. The well-defined quantity is
  INSTRUCTION SENSITIVITY, measured WITHIN each model, image held fixed:
        does the output CHANGE when the instruction changes?
  Metric is scale-free: count distinct outputs across an instruction set.
  base sensitive + fine-tuned collapsed  =>  fine-tuning caused the collapse.

THE POSITIVE CONTROL IS LOAD-BEARING - DO NOT SKIP IT
  If the base model is ALSO flat on your composites, that could mean "fine-tuning
  broke it" (your thesis) OR "composites are OOD for everything" (uninformative).
  The positive control disambiguates: it first confirms the base model DOES change
  its action with the instruction on IN-DISTRIBUTION robot frames (BridgeData V2 /
  OXE). If it fails there, the whole Step-1 result is uninterpretable and this
  script says so instead of printing a verdict.

PROMPT CONFOUND - EACH MODEL GETS ITS HOME-TURF PROMPT (both halves)
  BASE       -> native OpenVLA template, WITH "?", not lowercased:
                "In: What action should the robot take to {instr}?\nOut:"
  FINE-TUNED -> your training template, NO "?", lowercased, period kept:
                "In: What action should the robot take to {instr.lower()}\nOut: ..."
  Feeding the base model your no-"?" template would make a flat base result a
  PROMPT-FORMAT artifact masquerading as "collapse". Never do it - on either half.

RUN
  python src/evaluation/base_model_discrimination.py \
      --n 60 --split val --poscontrol-dir data/poscontrol --dump
  # optional real instructions (stronger positive control):
  #   --poscontrol-instructions data/poscontrol/instructions.json
"""
import sys, re, json, argparse, itertools
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.training.vla_dataset import OpenVLADataset

MODEL_ID = "openvla/openvla-7b"

# --- instruction sets -------------------------------------------------------
# Composite contrast: correct object vs wrong objects vs null. "correct" is the
# FT training phrase, so its parse-fail/on-target act as an alignment tripwire.
COMPOSITE_INSTRUCTIONS = {
    "correct":     "pick up the transparent plastic bottle",
    "wrong_can":   "pick up the metal can",
    "wrong_cube":  "pick up the red cube",
    "null_object": "pick up the object",
}
FT_TRAINING_PHRASE = "pick up the transparent plastic bottle"

# Positive-control fallback: content-agnostic, clearly-different commands. If the
# base model is instruction-conditioned at all, these yield different actions
# regardless of scene content. Real BridgeData instructions (via
# --poscontrol-instructions) are stronger; prefer them when available.
POSCONTROL_DEFAULT = [
    "pick up the object",
    "move the gripper to the left",
    "move the gripper to the right",
    "lift the arm up",
    "push the object forward",
]


# --- prompt builders --------------------------------------------------------
def base_prompt(instruction):
    # native OpenVLA template: WITH "?", not lowercased
    return f"In: What action should the robot take to {instruction}?\nOut:"


def build_prompt_text(decoded_full):
    out_pos = decoded_full.rfind("Out:")
    return None if out_pos == -1 else decoded_full[:out_pos + len("Out: ")]


def swap_instruction(prompt_text, old_instr, new_instr):
    """Swap the instruction phrase INSIDE the dataset's own decoded prompt,
    preserving its lowercasing + trailing-period convention (identical to v2)."""
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
        return prompt_text[:idx] + repl + prompt_text[idx + len(cand):]
    return None


# --- FT action parse (byte-identical to instruction_sensitivity_v2) ----------
def extract_action_values(text):
    out_pos = text.rfind("Out:")
    after = text[out_pos:] if out_pos != -1 else text
    matches = re.findall(r"(\d+)>", after)
    vals = [int(v) for v in matches if 0 <= int(v) <= 255]
    return vals if len(vals) >= 2 else None


# --- base action decode -----------------------------------------------------
def base_action(model, processor, pil_image, instruction, device, unnorm_key):
    inputs = processor(base_prompt(instruction), pil_image).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        act = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(act, dtype=np.float64).reshape(-1)


def n_distinct_base(actions, dp=3):
    return len({tuple(np.round(a, dp)) for a in actions})


def base_pairwise_l2(actions):
    if len(actions) < 2:
        return 0.0
    return float(np.mean([np.linalg.norm(a - b)
                          for a, b in itertools.combinations(actions, 2)]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="composite frames (first-N of split, matches v2's default iteration)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data", default="data/processed_dataset")
    ap.add_argument("--checkpoint", default="vla_checkpoints/lora_best")
    ap.add_argument("--poscontrol-dir", default="data/poscontrol")
    ap.add_argument("--poscontrol-instructions", default=None,
                    help="optional JSON {filename: instruction} of REAL instructions")
    ap.add_argument("--unnorm-key", default="bridge_orig")
    ap.add_argument("--out", default="base_model_discrimination.json")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path(args.data)

    print("[INIT] processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer

    print("[INIT] BASE model (4-bit; identical to your lora_best load, minus the adapter)...")
    base = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        load_in_4bit=True, device_map="auto", trust_remote_code=True)
    base.eval()
    if not hasattr(base, "predict_action"):
        print("[ABORT] base model has no predict_action"); return

    CKPT = Path(args.checkpoint)
    if not CKPT.exists():
        print(f"[ABORT] checkpoint not found: {CKPT}"); return

    results = {"model_id": MODEL_ID, "unnorm_key": args.unnorm_key,
               "split": args.split, "n_composite": args.n}

    # ======================================================================
    # PHASE 1 - POSITIVE CONTROL  (base only, in-distribution frames)
    #   All PURE-BASE inference happens BEFORE the PeftModel wrap in Phase 3.
    #   PeftModel.from_pretrained mutates `base` in place; wrapping earlier
    #   would contaminate every base output. >>> DO NOT REORDER. <<<
    # ======================================================================
    pc_dir = Path(args.poscontrol_dir)
    pc_imgs = sorted([p for p in pc_dir.glob("*")
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) if pc_dir.exists() else []
    pc_real = None
    if args.poscontrol_instructions and Path(args.poscontrol_instructions).exists():
        pc_real = json.load(open(args.poscontrol_instructions))

    print(f"\n[POSITIVE CONTROL] {len(pc_imgs)} frame(s) in {pc_dir}"
          f" | instructions: {'REAL BridgeData' if pc_real else 'default contrastive'}")
    pc_records = []
    for p in pc_imgs:
        img = Image.open(str(p)).convert("RGB")
        if pc_real and p.name in pc_real:
            instrs = [pc_real[p.name]] + POSCONTROL_DEFAULT
        else:
            instrs = list(POSCONTROL_DEFAULT)
        seen = set(); instrs = [x for x in instrs if not (x in seen or seen.add(x))]
        acts = [base_action(base, processor, img, ins, device, args.unnorm_key) for ins in instrs]
        nd = n_distinct_base(acts)
        pc_records.append({"file": p.name, "instructions": instrs, "n_distinct": nd,
                           "pairwise_l2": base_pairwise_l2(acts), "sensitive": nd > 1,
                           "actions": [a.round(4).tolist() for a in acts]})
        print(f"  {p.name:<30} n_distinct={nd}  L2={base_pairwise_l2(acts):.4f}  "
              f"{'SENSITIVE' if nd > 1 else 'FLAT'}")

    pc_frac = float(np.mean([r["sensitive"] for r in pc_records])) if pc_records else None
    results["positive_control"] = {
        "n_frames": len(pc_records), "sensitive_fraction": pc_frac,
        "mean_n_distinct": float(np.mean([r["n_distinct"] for r in pc_records])) if pc_records else None,
        "instruction_source": "real_bridgedata" if pc_real else "default_contrastive",
        "per_frame": pc_records,
    }
    pc_ok = pc_frac is not None and pc_frac >= 0.80
    if not pc_imgs:
        print("\n[POSITIVE CONTROL] EMPTY - composite results below are NOT interpretable")
        print("  as 'fine-tuning collapse' until this folder holds in-distribution robot frames.")
    elif not pc_ok:
        print(f"\n[POSITIVE CONTROL] WEAK (sensitive fraction {pc_frac:.2f} < 0.80) - base barely")
        print("  responds even in-distribution; fix measurement/model BEFORE reading composites.")
    else:
        print(f"\n[POSITIVE CONTROL] PASS (sensitive fraction {pc_frac:.2f}).")

    # ======================================================================
    # PHASE 2 - BASE MODEL ON COMPOSITES  (still pure base - before the wrap)
    # ======================================================================
    dataset = OpenVLADataset(data_dir=DATA_DIR, processor=processor, split=args.split)
    with open(DATA_DIR / "annotations.json") as f:
        records = json.load(f)[args.split]
    n_total = min(args.n, len(dataset))

    print(f"\n[BASE on COMPOSITES] {n_total} frames x {len(COMPOSITE_INSTRUCTIONS)} instructions...")
    base_comp = []
    for idx in range(n_total):
        rec = records[idx]
        img = Image.open(str(DATA_DIR / rec["image"])).convert("RGB")
        per = {}
        acts = []
        for name, ins in COMPOSITE_INSTRUCTIONS.items():
            a = base_action(base, processor, img, ins, device, args.unnorm_key)
            per[name] = a.round(4).tolist(); acts.append(a)
        nd = n_distinct_base(acts)
        corr, wcan = np.asarray(per["correct"]), np.asarray(per["wrong_can"])
        base_comp.append({"idx": idx, "n_distinct": nd, "pairwise_l2": base_pairwise_l2(acts),
                          "correct_vs_wrongcan_l2": float(np.linalg.norm(corr - wcan)),
                          "sensitive": nd > 1, "actions": per})
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{n_total}")
    results["composite_base"] = {
        "n_frames": len(base_comp),
        "sensitive_fraction": float(np.mean([r["sensitive"] for r in base_comp])) if base_comp else None,
        "mean_n_distinct": float(np.mean([r["n_distinct"] for r in base_comp])) if base_comp else None,
        "mean_correct_vs_wrongcan_l2": float(np.mean([r["correct_vs_wrongcan_l2"] for r in base_comp])) if base_comp else None,
        "per_frame": base_comp,
    }

    # ======================================================================
    # PHASE 3 - FINE-TUNED MODEL ON COMPOSITES
    #   The wrap happens HERE. After this line `base` is no longer pure base.
    # ======================================================================
    print(f"\n[WRAP] applying adapter {CKPT} -> fine-tuned model")
    ft = PeftModel.from_pretrained(base, str(CKPT))
    ft.eval()

    print(f"[FT on COMPOSITES] {n_total} frames x {len(COMPOSITE_INSTRUCTIONS)} instructions...")
    ft_comp, ft_dump = [], []
    tp_attempt = tp_parsefail = tp_ontarget = 0
    with torch.no_grad():
        for idx in range(n_total):
            rec = records[idx]
            batch = dataset[idx]
            prompt_text = build_prompt_text(
                tokenizer.decode(batch["input_ids"], skip_special_tokens=False))
            if prompt_text is None:
                continue
            instr = rec.get("instruction", FT_TRAINING_PHRASE)
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            tgt = cv2.imread(str(DATA_DIR / rec["mask_target"]), cv2.IMREAD_GRAYSCALE)
            H, W = (tgt.shape if tgt is not None else (None, None))

            per = {}
            for name, ins in COMPOSITE_INSTRUCTIONS.items():
                swapped = swap_instruction(prompt_text, instr, ins)
                if swapped is None:
                    per[name] = "SWAP_FAIL"; continue
                pid = tokenizer.encode(swapped, return_tensors="pt").to(device)
                out = ft.generate(input_ids=pid, pixel_values=pixel_values,
                                  max_new_tokens=50, do_sample=False)
                raw = tokenizer.decode(out[0], skip_special_tokens=False)
                pred = extract_action_values(raw)
                if pred is None:
                    per[name] = "PARSE_FAIL"
                    if args.dump and len(ft_dump) < 24:
                        ft_dump.append({"idx": idx, "instr": name,
                                        "raw_tail": raw[raw.rfind('Out:'):][:160]})
                else:
                    per[name] = [int(pred[0]), int(pred[1])]
                if name == "correct" and tgt is not None:  # alignment tripwire
                    tp_attempt += 1
                    if pred is None:
                        tp_parsefail += 1
                    else:
                        px = int(np.clip((pred[0] / 255.0) * W, 0, W - 1))
                        py = int(np.clip((pred[1] / 255.0) * H, 0, H - 1))
                        if tgt[py, px] > 0:
                            tp_ontarget += 1

            keys = {tuple(v) if isinstance(v, list) else v for v in per.values()}
            valid_pts = {tuple(v) for v in per.values() if isinstance(v, list)}
            ft_comp.append({"idx": idx, "outputs": per,
                            "n_distinct_incl_fail": len(keys),
                            "n_distinct_valid_points": len(valid_pts),
                            "sensitive": len(keys) > 1})
            if (idx + 1) % 20 == 0:
                print(f"  {idx + 1}/{n_total}")

    results["composite_ft"] = {
        "n_frames": len(ft_comp),
        "sensitive_fraction": float(np.mean([r["sensitive"] for r in ft_comp])) if ft_comp else None,
        "mean_n_distinct_valid_points": float(np.mean([r["n_distinct_valid_points"] for r in ft_comp])) if ft_comp else None,
        "per_frame": ft_comp,
        "tripwire_training_phrase": {
            "attempted": tp_attempt, "parse_fail": tp_parsefail, "on_target": tp_ontarget,
            "parse_fail_pct": (100 * tp_parsefail / tp_attempt) if tp_attempt else None,
            "on_target_pct": (100 * tp_ontarget / tp_attempt) if tp_attempt else None,
        },
    }

    # ======================================================================
    # VERDICT (conservative; defers detailed FT collapse to v2)
    # ======================================================================
    base_sens = results["composite_base"]["sensitive_fraction"]
    ft_sens = results["composite_ft"]["sensitive_fraction"]
    ft_vpts = results["composite_ft"]["mean_n_distinct_valid_points"]
    tw = results["composite_ft"]["tripwire_training_phrase"]
    v = []
    if not pc_imgs:
        v.append("INCONCLUSIVE - positive control EMPTY. Populate --poscontrol-dir with in-distribution "
                 "robot frames; without it, no flatness can be attributed to fine-tuning.")
    elif not pc_ok:
        v.append(f"INCONCLUSIVE - positive control WEAK (base sensitive on only {pc_frac:.0%} of "
                 f"in-distribution frames). Investigate measurement/model before interpreting composites.")
    else:
        v.append(f"Positive control PASS: base action changes with the instruction on {pc_frac:.0%} of "
                 f"in-distribution frames (measurement works; base IS instruction-conditioned).")
        v.append(f"Same composites - base sensitive {base_sens:.0%} vs fine-tuned {ft_sens:.0%} "
                 f"(FT distinct valid points/frame = {ft_vpts:.2f}).")
        v.append("Read with instruction_sensitivity_v2 (parse-fail gradient, semantic=chance): a "
                 "conditioned base + a flat fine-tuned model on the FT's OWN training distribution "
                 "attributes the collapse to fine-tuning, not to the domain being OOD.")
    if tw["parse_fail_pct"] is not None:
        v.append(f"[tripwire] FT training-phrase parse-fail {tw['parse_fail_pct']:.1f}% "
                 f"(expect ~16-18% from v2 baseline), on-target {tw['on_target_pct']:.1f}%. "
                 f"Far off => alignment/prompt broke; STOP and re-check before trusting anything above.")
    results["verdict"] = v

    print("\n" + "=" * 84)
    for line in v:
        print("  " + line)
    print("=" * 84)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[SAVED] {args.out}")
    if args.dump and ft_dump:
        dp = args.out.replace(".json", "_ft_failures.txt")
        with open(dp, "w") as f:
            for ex in ft_dump:
                f.write(f"[idx {ex['idx']} | {ex['instr']}] {ex['raw_tail']!r}\n")
        print(f"[SAVED] {dp}  <- read this to see WHAT the FT model emits when it breaks")


if __name__ == "__main__":
    main()