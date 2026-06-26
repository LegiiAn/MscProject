import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import perception models and structures
from src.perception.utils.dataset import ClearGraspPerceptionDataset
from src.perception.models.cleargrasp_net import ClearGraspDualNet

def compute_angular_error(pred_normals, gt_normals, mask):
    """Calculates the absolute angular error in degrees over valid object pixels."""
    # Ensure unit vector status via L2 normalization
    pred_normals = F.normalize(pred_normals, p=2, dim=1)
    gt_normals = F.normalize(gt_normals, p=2, dim=1)
    
    # Dot product calculation across channels
    dot_product = torch.sum(pred_normals * gt_normals, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Extract absolute angle in degrees
    angular_error_rad = torch.acos(dot_product)
    angular_error_deg = angular_error_rad * (180.0 / np.pi)
    
    # Isolate calculation to object boundaries to eliminate background bias
    valid_pixels = angular_error_deg[mask.squeeze(1) > 0.5]
    return valid_pixels.cpu().numpy() if valid_pixels.numel() > 0 else np.array([0.0])

def compute_iou(pred_mask_logits, gt_mask):
    """Computes Intersection over Union accuracy for binary boundary predictions."""
    pred_mask = (torch.sigmoid(pred_mask_logits) > 0.5).float()
    intersection = (pred_mask * gt_mask).sum().item()
    union = pred_mask.sum().item() + gt_mask.sum().item() - intersection
    return (intersection / union) if union > 0 else 0.0

def evaluate_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path.home() / "MscProject" / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train" / "square-plastic-bottle-train"
    checkpoint_path = Path("checkpoints") / "cleargrasp_dualnet_epoch_10.pth"
    
    print("[INIT] Booting Perception Evaluation Engine...")
    dataset = ClearGraspPerceptionDataset(base_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = ClearGraspDualNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_angular_errors = []
    all_ious = []
    
    # Store a target triplet sample for qualitative thesis plotting
    sample_visuals = None
    
    with torch.no_grad():
        for idx, (rgb_tensor, normals_target, mask_target) in enumerate(dataloader):
            rgb_tensor = rgb_tensor.to(device)
            normals_target = normals_target.to(device)
            mask_target = mask_target.to(device)
            
            pred_normals, pred_mask_logits = model(rgb_tensor)
            
            # Compute analytical batch performance metrics
            errors = compute_angular_error(pred_normals, normals_target, mask_target)
            all_angular_errors.extend(errors)
            all_ious.append(compute_iou(pred_mask_logits, mask_target))
            
            # Capture the 10th sample for your qualitative analysis grid
            if idx == 10:
                sample_visuals = {
                    'rgb': rgb_tensor[0].cpu().numpy().transpose(1, 2, 0),
                    'gt_norm': normals_target[0].cpu().numpy().transpose(1, 2, 0),
                    'pred_norm': F.normalize(pred_normals[0], p=2, dim=0).cpu().numpy().transpose(1, 2, 0),
                    'gt_mask': mask_target[0, 0].cpu().numpy(),
                    'pred_mask': torch.sigmoid(pred_mask_logits[0, 0]).cpu().numpy()
                }

    # --- Quantitative Metric Compilation ---
    all_angular_errors = np.array(all_angular_errors)
    mean_err = np.mean(all_angular_errors)
    median_err = np.median(all_angular_errors)
    mean_iou = np.mean(all_ious)
    
    print(f"\n================ PERCEPTION REPORT ================")
    print(f"Mean Angular Error:   {mean_err:.2f}°")
    print(f"Median Angular Error: {median_err:.2f}°")
    print(f"Object Mask Mean IoU: {mean_iou * 100:.2f}%")
    print(f"Accuracy Thresholds:  <11.25°: {np.mean(all_angular_errors < 11.25)*100:.1f}% | <22.5°: {np.mean(all_angular_errors < 22.5)*100:.1f}%")
    print(f"===================================================\n")

    # --- Print-Ready Qualitative Comparison Figure ---
    if sample_visuals is not None:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), dpi=300)
        
        # Helper to safely rescale vector normal formats [-1, 1] into viewing ranges [0, 1]
        def norm_to_rgb(img):
            return np.clip((img + 1.0) / 2.0, 0.0, 1.0)
            
        axes[0].imshow(np.clip(sample_visuals['rgb'], 0.0, 1.0))
        axes[0].set_title("Input RGB Space", fontsize=10, fontweight='bold')
        
        axes[1].imshow(norm_to_rgb(sample_visuals['gt_norm']))
        axes[1].set_title("Ground-Truth Normals", fontsize=10, fontweight='bold')
        
        axes[2].imshow(norm_to_rgb(sample_visuals['pred_norm']))
        axes[2].set_title("Predicted Normals", fontsize=10, fontweight='bold')
        
        axes[3].imshow(sample_visuals['gt_mask'], cmap='gray')
        axes[3].set_title("Ground-Truth Mask", fontsize=10, fontweight='bold')
        
        axes[4].imshow(sample_visuals['pred_mask'], cmap='magma')
        axes[4].set_title("Predicted Logits", fontsize=10, fontweight='bold')
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        output_plot = "perception_surface_reconstruction_matrix.pdf"
        plt.savefig(output_plot, format='pdf', bbox_inches='tight')
        print(f"[SUCCESS] Exported structural vector visualization asset to: {output_plot}")

if __name__ == "__main__":
    evaluate_and_visualize()
