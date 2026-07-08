import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.perception.utils.dataset import ClearGraspPerceptionDataset
from src.perception.models.cleargrasp_net import ClearGraspDualNet


def compute_angular_error(pred_normals, gt_normals, mask):
    """Angular error in degrees over valid object pixels."""
    pred_normals = F.normalize(pred_normals, p=2, dim=1)
    gt_normals = F.normalize(gt_normals, p=2, dim=1)

    dot = torch.sum(pred_normals * gt_normals, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)

    error_deg = torch.acos(dot) * (180.0 / np.pi)

    valid = error_deg[mask.squeeze(1) > 0.5]
    return valid.cpu().numpy() if valid.numel() > 0 else np.array([0.0])


def compute_iou(pred_logits, gt_mask):
    """IoU for binary segmentation."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * gt_mask).sum().item()
    union = pred.sum().item() + gt_mask.sum().item() - intersection
    return (intersection / union) if union > 0 else 0.0


def evaluate_on_dataset(model, dataloader, device, label=""):
    """Run evaluation and return metrics dict + sample visuals."""
    all_errors = []
    all_ious = []
    sample_visuals = None

    with torch.no_grad():
        for idx, (rgb, normals_gt, mask_gt) in enumerate(dataloader):
            rgb = rgb.to(device)
            normals_gt = normals_gt.to(device)
            mask_gt = mask_gt.to(device)

            pred_normals, pred_mask_logits = model(rgb)

            errors = compute_angular_error(pred_normals, normals_gt, mask_gt)
            all_errors.extend(errors)
            all_ious.append(compute_iou(pred_mask_logits, mask_gt))

            if idx == 10 and sample_visuals is None:
                sample_visuals = {
                    "rgb": rgb[0].cpu().numpy().transpose(1, 2, 0),
                    "gt_norm": normals_gt[0].cpu().numpy().transpose(1, 2, 0),
                    "pred_norm": F.normalize(pred_normals[0], p=2, dim=0)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0),
                    "gt_mask": mask_gt[0, 0].cpu().numpy(),
                    "pred_mask": torch.sigmoid(pred_mask_logits[0, 0]).cpu().numpy(),
                }

    all_errors = np.array(all_errors)

    metrics = {
        "mean_angular_error": float(np.mean(all_errors)),
        "median_angular_error": float(np.median(all_errors)),
        "rmse_angular_error": float(np.sqrt(np.mean(all_errors ** 2))),
        "pct_within_11.25": float(np.mean(all_errors < 11.25) * 100),
        "pct_within_22.5": float(np.mean(all_errors < 22.5) * 100),
        "pct_within_30": float(np.mean(all_errors < 30.0) * 100),
        "mean_iou": float(np.mean(all_ious) * 100),
        "n_images": len(all_ious),
    }

    print(f"\n{'='*20} {label} {'='*20}")
    print(f"  Images evaluated:    {metrics['n_images']}")
    print(f"  Mean Angular Error:  {metrics['mean_angular_error']:.2f}°")
    print(f"  Median Angular Err:  {metrics['median_angular_error']:.2f}°")
    print(f"  RMSE Angular Error:  {metrics['rmse_angular_error']:.2f}°")
    print(f"  < 11.25°:            {metrics['pct_within_11.25']:.1f}%")
    print(f"  < 22.5°:             {metrics['pct_within_22.5']:.1f}%")
    print(f"  < 30°:               {metrics['pct_within_30']:.1f}%")
    print(f"  Mean IoU:            {metrics['mean_iou']:.2f}%")

    return metrics, all_errors, all_ious, sample_visuals


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path.home() / "MscProject" / "data" / "cleargrasp_dataset" / "cleargrasp-dataset-train"
    checkpoint_path = Path("checkpoints") / "cleargrasp_dualnet_epoch_10.pth"

    # Trained on: square-plastic-bottle-train
    train_category = "square-plastic-bottle-train"

    # Held-out categories for generalization testing
    test_categories = [
        "cup-with-waves-train",
        "stemless-plastic-champagne-glass-train",
    ]

    print("[INIT] Loading model...")
    model = ClearGraspDualNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Evaluate on training category (in-distribution) ---
    train_dataset = ClearGraspPerceptionDataset(base_dir=base / train_category)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_metrics, train_errors, _, train_visuals = evaluate_on_dataset(
        model, train_loader, device, label=f"IN-DISTRIBUTION ({train_category})"
    )

    # --- Evaluate on held-out categories (out-of-distribution) ---
    ood_results = {}
    for cat in test_categories:
        cat_path = base / cat
        if not cat_path.exists():
            print(f"[SKIP] {cat} not found at {cat_path}")
            continue
        try:
            ds = ClearGraspPerceptionDataset(base_dir=cat_path)
            dl = DataLoader(ds, batch_size=1, shuffle=False)
            metrics, errors, _, _ = evaluate_on_dataset(
                model, dl, device, label=f"OUT-OF-DISTRIBUTION ({cat})"
            )
            ood_results[cat] = {"metrics": metrics, "errors": errors}
        except Exception as e:
            print(f"[ERROR] Failed on {cat}: {e}")

    # --- Generate thesis figures ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Panel 1: Angular error histogram (in-dist vs out-of-dist)
    axes[0].hist(train_errors, bins=50, alpha=0.6, color="#2ca02c", label=train_category, density=True)
    for cat, data in ood_results.items():
        short_name = cat.replace("-train", "")
        axes[0].hist(data["errors"], bins=50, alpha=0.5, label=short_name, density=True)
    axes[0].set_xlabel("Angular Error (degrees)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Surface Normal Error Distribution", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(0, 90)

    # Panel 2: CDF of angular error
    for label, errs in [(train_category, train_errors)] + [
        (c, d["errors"]) for c, d in ood_results.items()
    ]:
        sorted_e = np.sort(errs)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        axes[1].plot(sorted_e, cdf, label=label.replace("-train", ""))
    axes[1].axvline(x=11.25, color="gray", linestyle="--", alpha=0.5, label="11.25°")
    axes[1].axvline(x=22.5, color="gray", linestyle=":", alpha=0.5, label="22.5°")
    axes[1].set_xlabel("Angular Error (degrees)")
    axes[1].set_ylabel("Cumulative Proportion")
    axes[1].set_title("Threshold-Accuracy Curve", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=7)
    axes[1].set_xlim(0, 60)

    # Panel 3: Qualitative comparison (if available)
    if train_visuals is not None:
        def norm_rgb(img):
            return np.clip((img + 1.0) / 2.0, 0, 1)

        # Create a 2x2 inset
        for i, (title, img, cmap) in enumerate([
            ("Input RGB", np.clip(train_visuals["rgb"], 0, 1), None),
            ("GT Normals", norm_rgb(train_visuals["gt_norm"]), None),
            ("Pred Normals", norm_rgb(train_visuals["pred_norm"]), None),
            ("Pred Mask", train_visuals["pred_mask"], "magma"),
        ]):
            ax_inset = fig.add_axes([0.68 + (i % 2) * 0.155, 0.55 - (i // 2) * 0.45, 0.14, 0.38])
            if cmap:
                ax_inset.imshow(img, cmap=cmap)
            else:
                ax_inset.imshow(img)
            ax_inset.set_title(title, fontsize=6)
            ax_inset.axis("off")
        axes[2].axis("off")
    else:
        axes[2].text(0.5, 0.5, "No sample visuals", ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("perception_evaluation_full.pdf", format="pdf", bbox_inches="tight")
    print("\n[SUCCESS] Saved perception_evaluation_full.pdf")


if __name__ == "__main__":
    main()