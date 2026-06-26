import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor

class OpenVLADataset(Dataset):
    def __init__(self, data_dir: str, processor: AutoProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Load your synthetic image paths
        self.rgb_dir = self.data_dir / "rgb-imgs"
        self.rgb_files = sorted(list(self.rgb_dir.glob("*.jpg")) + list(self.rgb_dir.glob("*.jpeg")))
        
        # --- Action Discretization Setup ---
        self.num_bins = 256
        # In a real scenario, you calculate the 1st and 99th percentiles of your dataset.
        # Here we define structural bounding limits for the 7 continuous dimensions:
        # [x, y, z, roll, pitch, yaw, gripper]
        self.action_mins = np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0])
        self.action_maxs = np.array([ 1.0,  1.0,  1.0,  np.pi,  np.pi,  np.pi, 1.0])

    def tokenize_action(self, continuous_action: np.ndarray) -> str:
        """Converts a 7-DOF continuous array into a string of discrete action tokens."""
        # 1. Clip actions to boundaries to prevent outlier crashes
        clipped_action = np.clip(continuous_action, self.action_mins, self.action_maxs)
        
        # 2. Normalize values between 0.0 and 1.0
        normalized_action = (clipped_action - self.action_mins) / (self.action_maxs - self.action_mins)
        
        # 3. Map to 256 discrete bins (0 to 255)
        discretized_action = np.floor(normalized_action * (self.num_bins - 1)).astype(np.int32)
        
        # 4. Format into OpenVLA's special vocabulary tokens
        token_string = " ".join([f"<action_{val}>" for val in discretized_action])
        return token_string

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        
        # 1. Load Image
        image = Image.open(str(rgb_path)).convert("RGB")
        
        # 2. Fetch Dummy Action (In production, load this from your WP1 trajectory JSONs)
        # Simulating: Move forward slightly, lower arm, keep rotation flat, close gripper
        raw_action = np.array([0.15, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0]) 
        
        # 3. Convert math to text tokens
        action_tokens = self.tokenize_action(raw_action)
        
        # 4. Construct the precise VLA conversational prompt format
        instruction = "Pick up the transparent plastic bottle."
        prompt = f"In: What action should the robot take to {instruction.lower()}\nOut: {action_tokens}"
        
        # 5. Pass through the Hugging Face processor to convert text and images to model tensors
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        # Squeeze out the batch dimension added by the processor
        return {k: v.squeeze(0) for k, v in inputs.items()}

if __name__ == "__main__":
    # Local check using the updated transformers class
    from transformers import AutoProcessor
    MODEL_ID = "openvla/openvla-7b"
    
    print("[INIT] Loading Processor...")
    # trust_remote_code is required for OpenVLA's custom image processing logic
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    TEST_DIR = r"data/cleargrasp_dataset/cleargrasp-dataset-train/square-plastic-bottle-train"
    
    try:
        dataset = OpenVLADataset(data_dir=TEST_DIR, processor=proc)
        sample = dataset[0]
        print(f"[SUCCESS] VLA Dataset loaded. Triplet zero processed.")
        print(f"  Input IDs Shape (Text):  {sample['input_ids'].shape}")
        print(f"  Pixel Values Shape (Img):{sample['pixel_values'].shape}")
    except Exception as e:
        print(f"[FAILURE] VLA Dataset check crashed: {e}")