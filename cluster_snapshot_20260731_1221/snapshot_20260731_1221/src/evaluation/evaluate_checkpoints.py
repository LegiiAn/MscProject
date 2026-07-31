import os
import sys
import re
import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.vla_dataset import OpenVLADataset


def extract_action_values(text):
    """
    Extract action token values from generated text using flexible parsing.
    Handles multiple formats the model might produce:
      - <action_127>      (full format from training)
      - <127>             (prefix dropped)
      - 127>              (angle bracket dropped too)
      - _127>             (partial prefix)
    Returns list of ints, or None if fewer than 7 found.
    """
    # Strategy: find "Out:" then extract all numbers that appear in token-like contexts
    out_pos = text.rfind("Out:")
    if out_pos == -1:
        out_pos = 0

    after_out = text[out_pos:]

    # Try full format first
    full_matches = re.findall(r"<action_(\d+)>", after_out)
    if len(full_matches) >= 7:
        return [int(v) for v in full_matches[:7]]

    # Flexible: grab any number that appears between angle-bracket-like contexts
    # Matches: <action_N>, <N>, _N>, or standalone numbers after Out:
    flexible_matches = re.findall(r"(\d+)>", after_out)
    if len(flexible_matches) >= 7:
        return [int(v) for v in flexible_matches[:7]]

    # Last resort: just grab all numbers after "Out:"
    all_numbers = re.findall(r"\b(\d{1,3})\b", after_out)
    if len(all_numbers) >= 7:
        # Filter to plausible action range 0-255
        valid = [int(n) for n in all_numbers if 0 <= int(n) <= 255]
        if len(valid) >= 7:
            return valid[:7]

    return None


def extract_step_number(checkpoint_name):
    numbers = re.findall(r"\d+", checkpoint_name)
    return int(numbers[-1]) if numbers else 999999


def evaluate_checkpoint(checkpoint_path, val_dataset, processor, device, max_samples=50):
    """Loads a LoRA checkpoint and computes validation metrics."""
    print(f"\n[EVAL] Loading: {checkpoint_path.name}")

    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    tokenizer = processor.tokenizer

    total_samples = 0
    correct_action_tokens = 0
    total_action_tokens = 0
    spatial_l1_errors = []
    x_errors = []
    y_errors = []
    parse_failures = 0
    unique_predictions = set()

    with torch.no_grad():
        for idx in range(min(max_samples, len(val_dataset))):
            batch = val_dataset[idx]

            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)

            # Decode full input to get ground truth actions
            # Decode full input to get ground truth actions
            decoded_full = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            gt_actions = extract_action_values(decoded_full)
            if gt_actions is None:
                parse_failures += 1
                continue

            # ── THE FIX: SLICE THE TENSOR DIRECTLY TO PRESERVE <image> TOKENS ──
            input_ids_list = input_ids[0].tolist()
            split_idx = len(input_ids_list) - 8  # Fallback (7 actions + 1 EOS)
            
            # Find exactly where the "Out:" prompt ends in token-space
            for i in range(1, len(input_ids_list) + 1):
                text_so_far = tokenizer.decode(input_ids_list[:i])
                if text_so_far.rstrip().endswith("Out:"):
                    split_idx = i
                    break
                    
            # Slice the tensor directly. This keeps the <image> features intact!
            prompt_ids = input_ids[:, :split_idx]

            # Generate
            outputs = model.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                max_new_tokens=50,  # 15 is plenty when it uses proper <action_N> tokens
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )


            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            pred_actions = extract_action_values(generated_text)
            # ──────────────────────────────────────────────────────────────────

            if pred_actions is None:
                parse_failures += 1
                if idx < 5:
                    print(f"  [DEBUG] Parse fail sample {idx}: ...{generated_text[-150:]}")
                continue

            unique_predictions.add(tuple(pred_actions[:2]))

            total_samples += 1
            gt = np.array(gt_actions)
            pred = np.array(pred_actions)

            # Token accuracy
            matches = (pred == gt).sum()
            correct_action_tokens += matches
            total_action_tokens += 7

            # Spatial errors
            bin_errors = np.abs(pred.astype(float) - gt.astype(float))
            spatial_l1_errors.append(np.mean(bin_errors))
            x_errors.append(bin_errors[0])
            y_errors.append(bin_errors[1])

            if idx < 5:
                print(f"  [SAMPLE {idx}] GT={gt_actions[:2]} Pred={pred_actions[:2]} "
                      f"L1={np.mean(bin_errors):.1f}")
                
    # Dump per-sample errors for distribution analysis (best checkpoint only)
    if "best" in checkpoint_path.name or "13000" in checkpoint_path.name:
        with open("vla_detailed_errors.json", "w") as f:
            json.dump({
                "l1_errors": spatial_l1_errors,
                "x_errors": x_errors,
                "y_errors": y_errors,
            }, f)
        print("[SAVED] vla_detailed_errors.json")

    print(f"\n[DIAGNOSTIC] Unique coordinate pairs predicted: {len(unique_predictions)} / {total_samples}")    
    
    del model, base_model
    torch.cuda.empty_cache()

    if total_samples == 0:
        print(f"[WARN] No valid samples. Parse failures: {parse_failures}")
        return None

    return {
        "action_token_accuracy": (correct_action_tokens / total_action_tokens) * 100,
        "spatial_l1_mean": float(np.mean(spatial_l1_errors)),
        "spatial_l1_median": float(np.median(spatial_l1_errors)),
        "x_l1_mean": float(np.mean(x_errors)),
        "y_l1_mean": float(np.mean(y_errors)),
        "n_samples": total_samples,
        "parse_failures": parse_failures,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path("vla_checkpoints")
    data_dir = Path("data/processed_dataset")

    print("[INIT] Setting up evaluation pipeline...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    val_dataset = OpenVLADataset(data_dir=data_dir, processor=processor, split="val")
    
    

    # Prioritize evaluating the best checkpoint
    checkpoints = []
    best_dir = checkpoint_root / "lora_best"
    if best_dir.exists():
        checkpoints.append(best_dir)

    # Add other checkpoints
    for d in sorted(checkpoint_root.iterdir(), key=lambda x: extract_step_number(x.name)):
        if d.is_dir() and d != best_dir:
            checkpoints.append(d)

    if not checkpoints:
        print("[ERROR] No checkpoints found.")
        return

    results = {}
    for ckpt in checkpoints:
        label = ckpt.name
        metrics = evaluate_checkpoint(ckpt, val_dataset, processor, device)
        if metrics is None:
            continue
        results[label] = metrics
        print(
            f"\n[RESULTS] {label}:\n"
            f"  Action Token Acc: {metrics['action_token_accuracy']:.2f}%\n"
            f"  Spatial L1 Mean:  {metrics['spatial_l1_mean']:.2f} bins\n"
            f"  Spatial L1 Median:{metrics['spatial_l1_median']:.2f} bins\n"
            f"  X error mean:     {metrics['x_l1_mean']:.2f} bins\n"
            f"  Y error mean:     {metrics['y_l1_mean']:.2f} bins\n"
            f"  Samples/Failures: {metrics['n_samples']}/{metrics['parse_failures']}"
        )

    # Save results
    with open("vla_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results to vla_eval_results.json")


if __name__ == "__main__":
    main()
