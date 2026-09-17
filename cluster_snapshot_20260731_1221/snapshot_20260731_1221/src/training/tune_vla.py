import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from pathlib import Path
import json
import numpy as np

# Import the corrected dataset class
from vla_dataset import OpenVLADataset


def find_action_token_start(input_ids, tokenizer):
    """
    Locate where the action tokens begin in each sequence.
    Returns a tensor of shape (batch,) with the index of the first action token.

    Strategy: find the token for "Out:" in the sequence — action tokens follow it.
    If not found, fall back to masking only pad tokens.
    """
    # Encode the marker that precedes action tokens
    # We search for the last occurrence of the newline + "Out:" pattern
    batch_starts = []
    for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()
        decoded = tokenizer.decode(ids, skip_special_tokens=False)

        # Find the position of "Out:" in the decoded string
        out_pos = decoded.rfind("Out:")
        if out_pos == -1:
            # Fallback: don't mask anything extra (just pad)
            batch_starts.append(len(ids))
        else:
            # Re-encode everything up to and including "Out: " to find its token boundary
            prefix_text = decoded[:out_pos + len("Out: ")]
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            batch_starts.append(len(prefix_ids))

    return batch_starts


def train_vla():
    # --- Configuration ---
    MODEL_ID = "openvla/openvla-7b"

    # FIXED: Point directly to the processed_dataset directory.
    # vla_dataset.py reads annotations.json from this path directly — no double nesting.
    DATA_DIR = Path("data/processed_dataset")

    CHECKPOINT_DIR = Path("vla_checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = 2
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    CHECKPOINT_STEPS = 500
    VAL_EVERY_STEPS = 500       # Run validation this often
    VAL_BATCHES = 25            # Number of val batches per check
    PATIENCE = 15                # Early stopping: stop if val loss doesn't improve for this many checks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Active training device: {device}")

    # 1. Load Processor and Base Model
    print(f"[INIT] Loading Processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"[INIT] Loading Base Model in 4-bit quantization...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=True,
        device_map="auto",
    )

    # 2. LoRA setup — always start fresh for this corrected run
    print("[CONFIG] Wrapping model with fresh LoRA layers (clean start)...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 3. Load TRAIN and VAL datasets
    print("[DATA] Loading train and val splits...")
    train_dataset = OpenVLADataset(data_dir=DATA_DIR, processor=processor, split="train")
    val_dataset = OpenVLADataset(data_dir=DATA_DIR, processor=processor, split="val")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop with validation monitoring
    print("[START] Beginning training...")

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    training_log = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(
                device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            # CRITICAL: Build labels that mask the PROMPT PREFIX, not just pad tokens.
            # The model should only be trained to predict the action tokens.
            labels = input_ids.clone()

            # Mask pad tokens
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            # Mask prompt prefix tokens (everything before "Out: <action_...>")
            action_starts = find_action_token_start(input_ids, processor.tokenizer)
            for b, start_idx in enumerate(action_starts):
                labels[b, :start_idx] = -100

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            global_step += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] | "
                    f"Batch [{batch_idx}/{len(train_dataloader)}] | "
                    f"Loss: {loss.item():.4f}"
                )

            # --- Periodic validation ---
            if global_step % VAL_EVERY_STEPS == 0:
                val_loss = run_validation(
                    model, val_dataloader, processor, device, VAL_BATCHES
                )
                training_log.append({
                    "step": global_step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                })
                print(
                    f"  [VAL @ step {global_step}] val_loss={val_loss:.4f} | "
                    f"best={best_val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_dir = CHECKPOINT_DIR / "lora_best"
                    model.save_pretrained(best_dir)
                    print(f"  [BEST] New best model saved to {best_dir}")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"  [EARLY STOP] Val loss has not improved for {PATIENCE} checks. Stopping.")
                        save_training_log(training_log, CHECKPOINT_DIR)
                        return

                model.train()  # Back to training mode

            # --- Periodic checkpoint ---
            if global_step > 0 and global_step % CHECKPOINT_STEPS == 0:
                step_dir = CHECKPOINT_DIR / f"lora_epoch_{epoch+1}_step_{global_step}"
                model.save_pretrained(step_dir)
                print(f"  [CHECKPOINT] Saved at {step_dir}")

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")

        # End-of-epoch save
        output_dir = CHECKPOINT_DIR / f"lora_epoch_{epoch+1}_final"
        model.save_pretrained(output_dir)

    save_training_log(training_log, CHECKPOINT_DIR)
    print("[DONE] Training complete.")


def run_validation(model, val_dataloader, processor, device, max_batches):
    """Run a quick validation pass and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(
                device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            labels = input_ids.clone()
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100

            action_starts = find_action_token_start(input_ids, processor.tokenizer)
            for b, start_idx in enumerate(action_starts):
                labels[b, :start_idx] = -100

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            count += 1

    return total_loss / max(count, 1)


def save_training_log(log, checkpoint_dir):
    log_path = checkpoint_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[LOG] Training log saved to {log_path}")


if __name__ == "__main__":
    train_vla()