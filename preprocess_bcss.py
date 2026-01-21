import os
import random
from PIL import Image
from tqdm import tqdm


def process_bcss(source_root, dest_root, patch_size=64, split_ratios=(0.8, 0.1, 0.1)):
    """
    Reads 224x224 BCSS tiles and extracts 64x64 patches for training.
    """
    # Create split directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_root, split, 'data'), exist_ok=True)

    # Gather all image files
    image_files = []
    # Walk through the directory to find files
    if not os.path.exists(source_root):
        print(f"ERROR: Source path does not exist: {source_root}")
        return

    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} source images in {source_root}")

    if len(image_files) == 0:
        print("No images found! Check the path again.")
        return

    print("Starting processing...")

    patch_count = 0

    for img_path in tqdm(image_files):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                w, h = img.size


                # Extract 64x64 Patches (Grid)
                # For 224x224 input, this yields a 3x3 grid (9 patches per image)
                # discarding the small edge borders.
                for y in range(0, h - patch_size + 1, patch_size):
                    for x in range(0, w - patch_size + 1, patch_size):
                        patch = img.crop((x, y, x + patch_size, y + patch_size))

                        # Random Split
                        r = random.random()
                        if r < split_ratios[0]:
                            split = 'train'
                        elif r < split_ratios[0] + split_ratios[1]:
                            split = 'val'
                        else:
                            split = 'test'

                        # Save
                        # Use original filename prefix + patch index
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        save_name = f"{base_name}_p{patch_count}.png"
                        save_path = os.path.join(dest_root, split, 'data', save_name)

                        patch.save(save_path)
                        patch_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Done! Created {patch_count} patches in {dest_root}")


if __name__ == "__main__":
    # --- FIXED PATHS ---
    # Input: The folder where your debug script found the files
    raw_input = r"C:\Users\Lab2\PycharmProjects\final\data\BCSS"

    # Output: Where the training patches will be saved
    processed_output = r"C:\Users\Lab2\PycharmProjects\final\data\BCSS_Patches_64"

    process_bcss(raw_input, processed_output)