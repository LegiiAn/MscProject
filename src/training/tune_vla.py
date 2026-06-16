import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from pathlib import Path

# Import the verified dataset class
from vla_dataset import OpenVLADataset

def train_vla():
    # --- Configuration ---
    MODEL_ID = "openvla/openvla-7b"
    DATA_DIR = Path(r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset\cleargrasp-dataset-train\square-plastic-bottle-train")
    CHECKPOINT_DIR = Path("vla_checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 2  # Keep low for VRAM safety
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Active training device: {device}")

    # 1. Load Processor and Model
    print(f"[INIT] Loading Processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print(f"[INIT] Loading Base Model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # 2. Configure and Inject LoRA
    print("[CONFIG] Wrapping model with LoRA adapters...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Setup Dataset & Dataloader
    print("[DATA] Loading OpenVLA Dataset...")
    dataset = OpenVLADataset(data_dir=DATA_DIR, processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("[START] Beginning training loop...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # Move all input tensors to the active device
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            # OpenVLA uses input labels shifted by 1 internally for causal language modeling
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                # We break early on local CPU test runs to avoid freezing your machine
                if device.type == "cpu":
                    print("[INFO] CPU dry-run check complete. Exiting batch loop.")
                    break
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")
        
        # Save fine-tuned weights (saves only the lightweight adapter weights)
        output_dir = CHECKPOINT_DIR / f"lora_epoch_{epoch+1}"
        model.save_pretrained(output_dir)
        print(f"[SAVED] Adapter weights stored at {output_dir}")
        if device.type == "cpu":
            break

if __name__ == "__main__":
    train_vla()
