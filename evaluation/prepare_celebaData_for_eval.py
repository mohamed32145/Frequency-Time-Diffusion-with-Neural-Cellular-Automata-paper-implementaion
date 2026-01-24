import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Configuration
DATA_ROOT = r"C:\Users\Lab2\PycharmProjects\final\data"
OUTPUT_DIR = os.path.join(DATA_ROOT, "celeba_real_eval")
NUM_IMAGES = 2048
IMAGE_SIZE = 64


def prepare_real_images():
    # 1. Setup Transform (Must match Training exactly!)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # 2. Load CelebA Test Split
    print("Loading CelebA Test split...")
    ds = datasets.CelebA(
        root=DATA_ROOT,
        split='test',
        download=False,
        transform=transform
    )

    # 3. Save Images
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Processing and saving {NUM_IMAGES} real test images to {OUTPUT_DIR}...")


    count = 0
    for i in tqdm(range(len(ds))):
        if count >= NUM_IMAGES:
            break

        img_tensor, attr = ds[i]

        # Save as png
        save_name = f"real_{count:05d}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        save_image(img_tensor, save_path)

        count += 1

    print("Done!")


if __name__ == "__main__":
    prepare_real_images()