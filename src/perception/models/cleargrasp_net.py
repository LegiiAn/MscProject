import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class ClearGraspDualNet(nn.Module):
    """
    Dual-Head DeepLabV3 architecture for joint Surface Normal Estimation
    and Semantic Segmentation Mask prediction.
    """
    def __init__(self):
        super().__init__()
        
        # Load core DeepLabV3 model with ResNet50 backbone
        base_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        
        # Extract the backbone and the shared intermediate classifier blocks
        self.backbone = base_model.backbone
        self.shared_classifier = nn.Sequential(
            base_model.classifier[0],
            base_model.classifier[1],
            base_model.classifier[2],
            base_model.classifier[3]
        )
        
        # Create separate output channels from the shared 256-feature layer
        self.normals_head = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.mask_head = nn.Conv2d(256, 1, kernel_size=(1, 1))

    def forward(self, x):
        input_shape = x.shape[-2:] # Get original height and width (e.g., 256, 256)
        
        # 1. Feature Extraction
        features = self.backbone(x)['out']
        shared_feats = self.shared_classifier(features)
        
        # 2. Compute Surface Normals Head & Upsample
        raw_normals = self.normals_head(shared_feats)
        raw_normals = nn.functional.interpolate(
            raw_normals, size=input_shape, mode='bilinear', align_corners=False
        )
        # L2 normalize along the channel axis (dim=1) to ensure valid 3D unit vectors
        normals = nn.functional.normalize(raw_normals, p=2, dim=1)
        
        # 3. Compute Mask Head & Upsample
        raw_mask = self.mask_head(shared_feats)
        mask_logits = nn.functional.interpolate(
            raw_mask, size=input_shape, mode='bilinear', align_corners=False
        )
        
        return normals, mask_logits

if __name__ == "__main__":
    # Rapid local tensor check
    net = ClearGraspDualNet()
    net.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        pred_normals, pred_mask = net(dummy_input)
        
    print(f"Input image shape:   {dummy_input.shape}")
    print(f"Predicted Normals:   {pred_normals.shape}")
    print(f"Predicted Mask:      {pred_mask.shape}")