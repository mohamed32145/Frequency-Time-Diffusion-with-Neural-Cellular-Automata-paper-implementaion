"""Remove corrupted images from CelebA dataset"""
from PIL import Image
import os

data_dir = './data/celeba/img_align_celeba'
corrupted_files = []

print("Scanning for corrupted images...")
all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
print(f"Total files to check: {len(all_files)}")

for i, filename in enumerate(all_files):
    if i % 10000 == 0:
        print(f"Checked {i}/{len(all_files)} files, found {len(corrupted_files)} corrupted")
    
    filepath = os.path.join(data_dir, filename)
    try:
        img = Image.open(filepath)
        img.verify()  # Verify it's a valid image
    except Exception as e:
        print(f"Corrupted: {filename} - {e}")
        corrupted_files.append(filepath)

print(f"\nFound {len(corrupted_files)} corrupted files:")
for f in corrupted_files:
    print(f"  {os.path.basename(f)}")

if corrupted_files:
    response = input(f"\nDelete {len(corrupted_files)} corrupted files? (yes/no): ").strip().lower()
    if response == 'yes':
        for f in corrupted_files:
            os.remove(f)
            print(f"Deleted {os.path.basename(f)}")
        print(f"\nDeleted {len(corrupted_files)} corrupted files successfully!")
    else:
        print("Cancelled deletion.")
else:
    print("No corrupted files found!")
