import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# Fix path resolution if executed from different directories
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Force headless rendering backend for cluster compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import your custom dataset loader verified by grep
from src.training.vla_dataset import OpenVLADataset

def extract_step_number(checkpoint_name):
    """Parses step or epoch numbers out of directory names for chronological sorting."""
    numbers = re.findall(r'\d+', checkpoint_name)
    return int(numbers[-1]) if numbers else 999999

def evaluate_checkpoint(checkpoint_path, dataloader, device):
    """Loads a specific LoRA adapter checkpoint and calculates validation metrics."""
    print(f"\n[EVAL] Booting weights from: {checkpoint_path.name}")
    
    # Load base model wrapped in the specific checkpoint's LoRA layers
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    total_samples = 0
    correct_tokens = 0
    spatial_l1_errors = []
    semantic_formatting_successes = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= 25:  # Evaluate across 25 batches for speed
                break
                
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            
            # OpenVLA packs actions at the end of input_ids.
            # Slice off the last 7 tokens to isolate the ground-truth actions.
            target_actions = input_ids[:, -7:].cpu().numpy()
            
            # Feed ONLY the prompt context (everything before the last 7 tokens) to generate
            prompt_input_ids = input_ids[:, :-7]
            
            outputs = model.generate(
                input_ids=prompt_input_ids,
                pixel_values=pixel_values,
                max_new_tokens=7,
                do_sample=False
            )
            
            # Isolate the generated response tokens
            pred_actions = outputs[:, -7:].cpu().numpy()
            
            for b in range(pred_actions.shape[0]):
                total_samples += 1
                p_act = pred_actions[b]
                t_act = target_actions[b]
                
                if len(p_act) == 7:
                    semantic_formatting_successes += 1
                else:
                    continue
                
                # Calculate exact token matches and coordinate bin drift
                token_matches = (p_act == t_act).sum()
                correct_tokens += token_matches
                
                bin_errors = np.abs(p_act - t_act)
                spatial_l1_errors.append(np.mean(bin_errors))
                
    # Explicitly clear VRAM footprint before moving to next checkpoint
    del model, base_model
    torch.cuda.empty_cache()
    
    return {
        "token_accuracy": (correct_tokens / (total_samples * 7)) * 100,
        "spatial_l1_error_bins": float(np.mean(spatial_l1_errors)) if spatial_l1_errors else 255.0,
        "semantic_success_rate": (semantic_formatting_successes / total_samples) * 100
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path("vla_checkpoints")
    data_dir = Path.home() / "MscProject" / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train" / "square-plastic-bottle-train"
    
    print("[INIT] Setting up validation pipeline...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    # Updated to pass data_dir to match your OpenVLADataset signature
    val_dataset = OpenVLADataset(data_dir=data_dir, processor=processor)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    checkpoints = sorted(
        [d for d in checkpoint_root.iterdir() if d.is_dir()],
        key=lambda x: extract_step_number(x.name)
    )
    
    results = {}
    for ckpt in checkpoints:
        step = extract_step_number(ckpt.name)
        metrics = evaluate_checkpoint(ckpt, val_dataloader, device)
        results[step] = metrics
        print(f"[METRICS] Step {step} -> Accuracy: {metrics['token_accuracy']:.2f}% | L1 Error: {metrics['spatial_l1_error_bins']:.2f} bins")

    # --- Plotting Generation ---
    steps = sorted(results.keys())
    accuracies = [results[s]["token_accuracy"] for s in steps]
    l1_errors = [results[s]["spatial_l1_error_bins"] for s in steps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    ax1.plot(steps, accuracies, color="#2ca02c", marker="o", linewidth=2, label="Action Token Accuracy")
    ax1.set_title("Robotic Action Selection Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Steps", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, l1_errors, color="#d62728", marker="s", linewidth=2, label="Coordinate L1 Drift")
    ax2.set_title("Spatial Coordinate Error Tracking", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("Error (Bins)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("vla_comprehensive_evaluation.pdf", format="pdf", bbox_inches="tight")
    print("[SUCCESS] Exported plot to vla_comprehensive_evaluation.pdf")

if __name__ == "__main__":
    main()
