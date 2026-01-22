import torch
import os
from torchvision.utils import save_image
from config import DiffNCAConfig, TrainConfig
from models import DiffNCA
from diffusion import sample_ddpm
from utils import denormalize, set_seed

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints_bcss_diffnca/final.pt"
save_path = "generated_large_bcss.png"

# 2. Load the Model Architecture
model_cfg = DiffNCAConfig()
model = DiffNCA(model_cfg).to(device)

# 3. Load Weights
print(f"Loading weights from {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, map_location=device)
# Handle standard vs compiled state dict keys
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("_orig_mod.", "") # Remove compile prefix if present
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)

# Generate at 512x512
# (Batch_Size, Channels, Height, Width)
target_shape = (1, 3, 512, 512)

print(f"Generating image of size {target_shape}...")

# Optional: Use mixed precision for speed/memory
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    samples = sample_ddpm(
        model,
        shape=target_shape,
        device=device,
        T=1000
    )


samples_denorm = denormalize(samples.float())
save_image(samples_denorm, save_path, normalize=False)
print(f"Saved large image to {save_path}")