import json
import random
import cv2
from pathlib import Path

OUT_DIR = Path(r"C:\Users\m_vit\Documents\MscProject\data\processed_dataset")
with open(OUT_DIR / "annotations.json", "r") as f:
    data = json.load(f)

# Pick a random sample from the train split
sample = random.choice(data["train"])
img_path = OUT_DIR / sample["image"]
cx, cy = sample["click_2d"]
tokens = sample["action_tokens"]

# Load and draw a crosshair on the calculated center
img = cv2.imread(str(img_path))
cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)

print(f"Showing: {sample['image']}")
print(f"Category: {sample['category']}")
print(f"VLA Action String: {tokens}")
print(f"Target Token X (Index 0): {tokens[0]} | Target Token Y (Index 1): {tokens[1]}")

cv2.imshow("Sanity Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()