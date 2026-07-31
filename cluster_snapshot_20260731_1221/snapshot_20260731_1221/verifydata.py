import cv2
import json
from pathlib import Path

OUT_DIR = Path.home() / "MscProject/data/processed_dataset"
with open(OUT_DIR / "annotations.json", "r") as f:
    annotations = json.load(f)

# Check the first 5 validation samples
for anno in annotations["val"][:5]:
    img_path = OUT_DIR / anno["image"]
    cx, cy = anno["click_2d"]
    
    img = cv2.imread(str(img_path))
    print(f"Image: {img_path.name} | Loaded Shape (H, W): {img.shape[:2]} | Expected Target: ({cx}, {cy})")
    
    # Draw a bright red circle at the target coordinate
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
    
    # Save it to your home directory to look at
    cv2.imwrite(f"verify_{img_path.name}", img)
