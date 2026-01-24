import torch
import os
from torchvision.utils import save_image
from config import DiffNCAConfig, TrainConfig
from models import DiffNCA
from diffusion import sample_ddpm
from utils import denormalize, set_seed

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "../checkpoint_190000.pt"
save_path = "../generated_large_bcss.png"

# 2. Load the Model Architecture
model_cfg = DiffNCAConfig()
model = DiffNCA(model_cfg).to(device)

# 3. Load Weights (Updated for Option 2: EMA Priority)
print(f"Loading weights from {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, map_location=device)

# Logic: Check for EMA -> Check for 'model' key -> Fallback
if 'ema' in ckpt:
    print(">>> Success: Loading EMA weights (Smoother, Higher Quality)...")
    raw_state_dict = ckpt['ema']
elif 'model' in ckpt:
    print(">>> Warning: EMA not found. Loading standard weights...")
    raw_state_dict = ckpt['model']
elif 'model_state_dict' in ckpt:
    raw_state_dict = ckpt['model_state_dict']
else:

    raw_state_dict = ckpt


new_state_dict = {}
for k, v in raw_state_dict.items():
    new_key = k.replace("_orig_mod.", "")
    new_state_dict[new_key] = v

# Load and set to Eval mode
model.load_state_dict(new_state_dict)
model.eval()

# 4. Generate at 512x512
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