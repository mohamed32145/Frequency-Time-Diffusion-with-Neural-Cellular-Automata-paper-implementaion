import torch
import os
from tqdm import tqdm
from torchvision.utils import save_image
from config import DiffNCAConfig, FourierDiffNCAConfig
from models import DiffNCA, FourierDiffNCA
from diffusion import sample_ddpm
from utils import denormalize, set_seed


def generate_evaluation_set(model_type="DiffNCA", num_images=2048, batch_size=32, output_dir="data/generated_eval"):
    # 1. Setup
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_images} images with {model_type}...")

    # 2. Initialize Model
    if model_type == "DiffNCA":
        cfg = DiffNCAConfig()
        model = DiffNCA(cfg).to(device)
        # Update this to your specific path if needed
        ckpt_path = "../checkpoint_190000.pt"
    elif model_type == "FourierDiffNCA":
        cfg = FourierDiffNCAConfig()
        model = FourierDiffNCA(cfg).to(device)
        ckpt_path = "../final.pt"
    else:
        raise ValueError("Unknown model type")

    # 3. Load Weights (CORRECTED LOGIC)
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found! Please train the model first.")
        return

    ckpt = torch.load(ckpt_path, map_location=device)

    # Logic to find the correct weights dictionary
    if 'ema' in ckpt:
        print("Found EMA weights! Using them for best quality.")
        state_dict = ckpt['ema']
    elif 'model' in ckpt:
        print("Found standard model weights.")
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        # Fallback for raw saved state dicts
        state_dict = ckpt

    # Clean the keys if torch.compile was used (removes "_orig_mod.")
    clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load into model
    model.load_state_dict(clean_dict)

    # 4. Generation Loop
    generated_count = 0
    num_batches = (num_images + batch_size - 1) // batch_size

    model.eval()

    print(f"Starting generation loop on {device}...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            # Calculate actual batch size for the last batch
            current_batch_size = min(batch_size, num_images - generated_count)

            # Use Mixed Precision for speed
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                samples = sample_ddpm(
                    model,
                    shape=(current_batch_size, 3, 64, 64),
                    device=device,
                    T=1000
                )

            # Save individual images
            samples = denormalize(samples.float())
            for j, img in enumerate(samples):
                file_name = f"gen_{generated_count + j:05d}.png"
                save_image(img, os.path.join(output_dir, file_name))

            generated_count += current_batch_size

    print(f"Done! Saved {generated_count} images to {output_dir}")


if __name__ == "__main__":
    # Ensure this matches your folder name
    #generate_evaluation_set(model_type="DiffNCA", num_images=2048, output_dir="../data/generated_eval_DiffNCA")
    generate_evaluation_set(model_type="FourierDiffNCA", num_images=2048, output_dir="data/generated_eval_FourierDiffNCA")