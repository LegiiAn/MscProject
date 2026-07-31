import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor


class OpenVLADataset(Dataset):
    """
    Dataset loader for ReSort-IT composited images with per-sample action targets.
    Reads annotations.json written by generatedata.py.
    
    JSON structure from generatedata.py:
    {
        "train": [
            {
                "image": "train/category_stem_aug0.png",
                "category": "square-plastic-bottle-train",
                "instruction": "Pick up the transparent plastic bottle.",
                "source_file": "000000-rgb.png",
                "background": "bg_001.jpg",
                "click_2d": [320, 240],
                "action_tokens": [127, 127, 127, 127, 127, 127, 255]
            },
            ...
        ],
        "val": [...],
        "test": [...]
    }
    """

    def __init__(self, data_dir: str, processor: AutoProcessor, split: str = "train"):
        """
        Args:
            data_dir:  Path to the processed_dataset directory containing annotations.json
                       and train/val/test image subdirectories.
            processor: HuggingFace AutoProcessor for the VLA model.
            split:     One of "train", "val", "test".
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split

        annotations_path = self.data_dir / "annotations.json"
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"annotations.json not found at {annotations_path}. "
                f"Run generatedata.py first to create the ReSort-IT dataset."
            )

        with open(annotations_path, "r") as f:
            all_annotations = json.load(f)

        # Select the correct split
        if split not in all_annotations:
            raise KeyError(
                f"Split '{split}' not found in annotations.json. "
                f"Available keys: {list(all_annotations.keys())}"
            )

        self.samples = all_annotations[split]
        print(f"[DATA] Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]

        # 1. Load the composited image
        #    record["image"] is relative to data_dir, e.g. "train/cup-with-waves_000001-rgb_aug0.png"
        img_path = self.data_dir / record["image"]
        image = Image.open(str(img_path)).convert("RGB")

        # 2. Read pre-computed action tokens (already discretized 0-255 by generatedata.py)
        action_tokens = record["action_tokens"]  # list of 7 ints

        # 3. Format as OpenVLA special vocabulary tokens
        token_string = " ".join([f"<action_{val}>" for val in action_tokens])

        # 4. Build the VLA prompt using the category-specific instruction
        instruction = record.get("instruction", "Pick up the transparent plastic object.")
        prompt = f"In: What action should the robot take to {instruction.lower()}\nOut: {token_string}"

        # 5. Tokenize through the HuggingFace processor
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        # Squeeze out the batch dimension added by the processor
        return {k: v.squeeze(0) for k, v in inputs.items()}


if __name__ == "__main__":
    from transformers import AutoProcessor

    MODEL_ID = "openvla/openvla-7b"
    print("[INIT] Loading Processor...")
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Point to your processed_dataset directory
    DATASET_DIR = r"C:\Users\m_vit\Documents\MscProject\data\processed_dataset"

    for split in ["train", "val", "test"]:
        try:
            ds = OpenVLADataset(data_dir=DATASET_DIR, processor=proc, split=split)
            if len(ds) > 0:
                sample = ds[0]
                print(f"[OK] split={split} | {len(ds)} samples | "
                      f"input_ids={sample['input_ids'].shape} | "
                      f"pixel_values={sample['pixel_values'].shape}")

                # Decode the token IDs to verify the prompt + action format
                decoded = proc.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
                # Print first 300 chars to visually confirm action tokens at the end
                print(f"     Decoded (first 300): {decoded[:300]}")
            else:
                print(f"[WARN] split={split} is empty")
        except Exception as e:
            print(f"[FAIL] split={split}: {e}")