import os
import random
from pathlib import Path
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# Import your verified synthetic dataset
from vla_dataset import OpenVLADataset


class HFBaselineDataset(Dataset):
    """
    Custom local dataset loader that completely bypasses the 'lerobot' and 'datasets' libraries.
    It reads the raw Parquet metadata and extracts video frames on the fly using OpenCV.
    """
    def __init__(self, base_dir, processor):
        self.processor = processor
        self.base_dir = Path(base_dir)
        
        # 1. Find the data Parquet files (LeRobot stores them under a 'data' folder or at the root)
        data_dir = self.base_dir / "data"
        if not data_dir.exists():
            data_dir = self.base_dir
            
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"[ERROR] No .parquet files found in {data_dir}. "
                f"Please verify that your baseline dataset is extracted there."
            )
        
        print(f"[DATA] Loading baseline Parquet metadata from: {data_dir}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Ensure consistent indexing by sorting by episode and frame
        if "episode_index" in self.df.columns and "frame_index" in self.df.columns:
            self.df = self.df.sort_values(by=["episode_index", "frame_index"]).reset_index(drop=True)
            
        print(f"[DATA] Successfully loaded {len(self.df)} baseline frames without LeRobot!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Extract Episode and Frame Index to target the correct video frame
        episode_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        
        # Standard LeRobot folder structure for videos: videos/<camera_name>/episode_<id>.mp4
        video_path = self.base_dir / "videos" / "observation.images.top" / f"episode_{episode_idx:06d}.mp4"
        
        # Robust fallback: if it's nested differently, search recursively for the video file
        if not video_path.exists():
            video_matches = list(self.base_dir.glob(f"**/episode_{episode_idx:06d}.mp4"))
            if video_matches:
                video_path = video_matches[0]
            else:
                raise FileNotFoundError(f"[ERROR] Could not find video file for episode {episode_idx} inside {self.base_dir}")

        # 2. Extract specific frame using OpenCV (cv2)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"[ERROR] Failed to extract frame {frame_idx} from video: {video_path}")
            
        # OpenCV reads frames in BGR, convert it to an RGB PIL Image for the processor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # 3. Default Real-World Instruction
        instruction = "pick up the block and insert it into the peg"

        # 4. Action Extraction and Quantization
        raw_action = row["action"]
        if isinstance(raw_action, np.ndarray) or hasattr(raw_action, "tolist"):
            raw_action = raw_action.tolist()
        
        # Normalize continuous actions roughly into 0-255 discrete bins for OpenVLA tokenization
        action_bins = [int(min(max((a + 1.0) / 2.0 * 255, 0), 255)) for a in raw_action[:7]]

        # 5. OpenVLA Format String Construction
        prompt = f"In: What action should the robot take to {instruction}?\nOut: "
        action_str = "".join([f"<0x{a:02x}>" for a in action_bins])
        full_text = prompt + action_str

        # 6. Tokenize the inputs using the model's processor
        inputs = self.processor(
            text=full_text, 
            images=image, 
            return_tensors="pt",
            padding="max_length",
            max_length=128,  # Keeps tensor sizes uniform for DataLoader batching
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"][0],
            "pixel_values": inputs["pixel_values"][0]
        }


def train_vla():
    # --- Configuration ---
    MODEL_ID = "openvla/openvla-7b"
    SYNTHETIC_DATA_DIR = Path("data/cleargrasp_dataset/cleargrasp-dataset-train/square-plastic-bottle-train")
    
    # PATH TO YOUR EXTRACTED ALOHA BASELINE DATASET (Adjust folder name if different)
    BASELINE_DATA_DIR = Path("data/aloha_sim_insertion_human")
    
    CHECKPOINT_DIR = Path("vla_checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Resume Directory
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
    print("[DATA] Loading OpenVLA Synthetic Dataset...")
    synthetic_dataset = OpenVLADataset(data_dir=SYNTHETIC_DATA_DIR, processor=processor)
    
    # Initialize the new Custom Baseline Dataset (No HuggingFace or LeRobot imports used)
    print(f"[DATA] Initializing Baseline Dataset from {BASELINE_DATA_DIR}...")
    baseline_dataset = HFBaselineDataset(base_dir=BASELINE_DATA_DIR, processor=processor)
    
    # Ratio Calculation and Bootstrapping Logic
    openx_needed = int(len(synthetic_dataset) * 0.25)
    available_openx = len(baseline_dataset)
    print(f"[DATA] Synthetic size: {len(synthetic_dataset)} | Baseline size available: {available_openx}")

    if available_openx >= openx_needed:
        openx_indices = random.sample(range(available_openx), openx_needed)
    else:
        print(f"[WARN] Bootstrapping baseline data to meet the {openx_needed} required samples.")
        openx_indices = random.choices(range(available_openx), k=openx_needed)

    openx_subset = Subset(baseline_dataset, openx_indices)
    
    # Mix the datasets
    mixed_train_dataset = ConcatDataset([synthetic_dataset, openx_subset])
    dataloader = DataLoader(mixed_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
            # Fast-forward past already computed batches
            if epoch == 0 and batch_idx < 4500:
                if batch_idx % 500 == 0:
                    print(f"[FAST-FORWARD] Skipping batch {batch_idx}/4500...")
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