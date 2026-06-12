import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path

# Import your updated custom modules
from utils.dataset import ClearGraspPerceptionDataset
from models.cleargrasp_net import ClearGraspDualNet

def train():
    # --- Configuration ---
    DATA_DIR = Path(r"C:\Users\m_vit\Documents\MscProject\data\cleargrasp_dataset\cleargrasp-dataset-train\square-plastic-bottle-train")
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 4      
    EPOCHS = 10         
    LEARNING_RATE = 1e-4

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Data Loading ---
    print("Loading multi-modal dataset...")
    dataset = ClearGraspPerceptionDataset(base_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Total batches per epoch: {len(dataloader)}")

    # --- Model, Optimizers, & Loss Functions ---
    model = ClearGraspDualNet().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Binary Cross Entropy with Logits for the mask channel
    mask_criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (rgb_batch, normals_target, mask_target) in enumerate(dataloader):
            # Move tensors to active device
            rgb_batch = rgb_batch.to(device)
            normals_target = normals_target.to(device)
            mask_target = mask_target.to(device)

            # Forward pass returning BOTH predictions
            pred_normals, pred_mask_logits = model(rgb_batch)

            # 1. Calculate Cosine Loss for Surface Normals
            cosine_sim = F.cosine_similarity(pred_normals, normals_target, dim=1)
            loss_normals = (1.0 - cosine_sim).mean()

            # 2. Calculate BCE Loss for Segmentation Mask
            loss_mask = mask_criterion(pred_mask_logits, mask_target)

            # 3. Combine Losses (Equal weight distribution baseline)
            total_loss = loss_normals + loss_mask

            # Backward pass & optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Total Loss: {total_loss.item():.4f} (Normals: {loss_normals.item():.4f}, Mask: {loss_mask.item():.4f})")

        # Epoch Summary
        avg_loss = running_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Complete | Average Joint Loss: {avg_loss:.4f} ---")
        
        # Save multi-task checkpoint weights
        checkpoint_path = CHECKPOINT_DIR / f"cleargrasp_dualnet_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()