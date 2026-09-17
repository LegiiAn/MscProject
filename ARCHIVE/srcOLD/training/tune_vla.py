import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from pathlib import Path

# Import the verified dataset class
from vla_dataset import OpenVLADataset

def train_vla():
    # --- Configuration ---
    MODEL_ID = "openvla/openvla-7b"
    DATA_DIR = Path("data/cleargrasp_dataset/cleargrasp-dataset-train/square-plastic-bottle-train")
    CHECKPOINT_DIR = Path("vla_checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Chemin du dernier checkpoint valide trouvé sur ton cluster
    RESUME_DIR = CHECKPOINT_DIR / "lora_epoch_1_step_4500"
    
    BATCH_SIZE = 2  
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    CHECKPOINT_STEPS = 500 

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
        device_map="auto"
    )

    # 2. Load Existing LoRA Weights OR Create New Adapters
    if RESUME_DIR.exists():
        print(f"[CONFIG] Resuming from checkpoint: {RESUME_DIR}")
        model = PeftModel.from_pretrained(base_model, RESUME_DIR, is_trainable=True)
    else:
        print("[CONFIG] No checkpoint found. Wrapping model with fresh LoRA adapters...")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
        
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
    
    start_epoch = 0
    if RESUME_DIR.exists():
        print("[INFO] Adapters loaded successfully. Continuing training stream...")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # Avance rapide : on saute le calcul des 4500 premiers batches déjà appris
            if epoch == 0 and batch_idx < 4500:
                if batch_idx % 500 == 0:
                    print(f"[FAST-FORWARD] Saut du batch {batch_idx}/4500...")
                continue
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
            
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
            
            if batch_idx > 0 and batch_idx % CHECKPOINT_STEPS == 0:
                step_dir = CHECKPOINT_DIR / f"lora_epoch_{epoch+1}_step_{batch_idx}"
                model.save_pretrained(step_dir)
                print(f"[CHECKPOINT] Persistent step backup stored at {step_dir}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")
        
        output_dir = CHECKPOINT_DIR / f"lora_epoch_{epoch+1}_final"
        model.save_pretrained(output_dir)
        print(f"[SAVED] Final Epoch adapter weights stored at {output_dir}")

if __name__ == "__main__":
    train_vla()
