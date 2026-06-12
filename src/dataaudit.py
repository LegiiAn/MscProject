from pathlib import Path

base = Path("data/cleargrasp_dataset/cleargrasp-dataset-train")
categories = [d for d in base.iterdir() if d.is_dir()]

for cat in categories:
    rgb = list((cat / "rgb-imgs").glob("*.png")) + \
          list((cat / "rgb-imgs").glob("*.jpg"))
    masks = list((cat / "segmentation-masks").glob("*.png"))
    depth = list((cat / "depth-imgs-rectified").glob("*.png")) + \
            list((cat / "depth-imgs-rectified").glob("*.exr"))
    
    print(f"\n{cat.name}")
    print(f"  RGB images:  {len(rgb)}")
    print(f"  Masks:       {len(masks)}")
    print(f"  Depth:       {len(depth)}")
    print(f"  Counts match: {len(rgb) == len(masks) == len(depth)}")

bg = Path("data/raw_backgrounds")
backgrounds = list(bg.glob("*.jpg")) + list(bg.glob("*.png"))
print(f"\nBackgrounds: {len(backgrounds)}")