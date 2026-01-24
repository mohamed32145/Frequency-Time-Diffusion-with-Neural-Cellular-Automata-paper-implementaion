import os
import random
from PIL import Image
from tqdm import tqdm


def prepare_bcss_eval_set(source_root, dest_root, target_count=2048, patch_size=64):
    """
    Extracts exactly 'target_count' patches (64x64) from BCSS images for FID evaluation.
    """
    os.makedirs(dest_root, exist_ok=True)

    # 1. Gather all image files
    image_files = []
    if not os.path.exists(source_root):
        print(f"ERROR: Source path does not exist: {source_root}")
        return

    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} source images. extracting {target_count} patches...")

    # Shuffle files to get a random sample of the dataset
    random.shuffle(image_files)

    count = 0

    # 2. Extract Patches until we hit target_count
    for img_path in tqdm(image_files):
        if count >= target_count:
            break

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                w, h = img.size

                # Extract grid of patches
                for y in range(0, h - patch_size + 1, patch_size):
                    for x in range(0, w - patch_size + 1, patch_size):
                        if count >= target_count:
                            break

                        patch = img.crop((x, y, x + patch_size, y + patch_size))

                        # Save with simple filename
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        save_name = f"real_{base_name}_p{count}.png"
                        save_path = os.path.join(dest_root, save_name)

                        patch.save(save_path)
                        count += 1

        except Exception as e:
            print(f"Skipping broken image {img_path}: {e}")

    print(f"Done! Saved {count} images to {dest_root}")


if __name__ == "__main__":
    # Input: Your raw BCSS images
    raw_input = r"C:\Users\Lab2\Desktop\BCSS\test"

    # Output: Folder for FID reference images
    eval_output = r"C:\Users\Lab2\PycharmProjects\final\data\bcss_real_eval"

    prepare_bcss_eval_set(raw_input, eval_output, target_count=2048)