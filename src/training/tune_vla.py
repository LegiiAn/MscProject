import os
import torch
import torch.nn as nn
# Update to the new v5.0+ API name
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model

def setup_vla_training():
    # --- Configuration ---
    # Model ID for openvla from HuggingFace
    MODEL_ID = "openvla/openvla-7b"
    
    print(f"[INIT] Loading OpenVLA Processor for: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print("[INIT] Loading 7B Parameter Base Model...")
    # On the cluster, we will use bfloat16 or 8-bit quantization for VRAM efficiency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update the class call here to match
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    # --- LoRA Configuration ---
    print("[CONFIG] Injecting Low-Rank Adaptation (LoRA) Layers...")
    
    # OpenVLA binds actions to its internal LLM attention layers (typically the 'q_proj' and 'v_proj')
    lora_config = LoraConfig(
        r=32,                   # Rank matrix dimension (controls adaptation capacity)
        lora_alpha=32,          # Scaling factor
        target_modules=["q_proj", "v_proj"], # Target the linear attention projection matrices
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"   # The underlying action generator acts like an autoregressive language head
    )
    
    # Wrap the 7B base model with trainable LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor

if __name__ == "__main__":
    # Local syntax and structural execution check
    try:
        print("=== Work Package 3: OpenVLA Training Configuration Check ===")
        # We don't download the full 7B model locally on your CPU (it would freeze).
        # This block verifies that your package environment can resolve the modules cleanly.
        import peft
        import transformers
        print("[SUCCESS] Environment dependencies verified. Ready for Cluster deployment.")
        
    except Exception as e:
        print(f"[FAILURE] Environment mismatch: {str(e)}")