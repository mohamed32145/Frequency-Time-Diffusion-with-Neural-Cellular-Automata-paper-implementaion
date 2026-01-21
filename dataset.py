import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from config import DataConfig


class RobustDataset(Dataset):
    """Wrapper that retries if an image load fails (corruption handling)"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        max_attempts = 50
        tried_indices = set()
        for attempt in range(max_attempts):
            try:
                if attempt == 0:
                    actual_idx = self.valid_indices[idx % len(self.valid_indices)]
                else:
                    actual_idx = random.choice(self.valid_indices)

                if actual_idx in tried_indices: continue
                tried_indices.add(actual_idx)

                return self.dataset[actual_idx]
            except Exception as e:
                if attempt == 0:
                    print(f"\nWarning: load failed at index {actual_idx}: {e}")
                continue
        raise RuntimeError(f"Failed to load image after {max_attempts} attempts")


def get_dataloader(cfg: DataConfig):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    if cfg.dataset_name.lower() == "celeba":
        print(f"Loading CelebA from {cfg.data_root}...")
        ds = datasets.CelebA(
            root=cfg.data_root,
            split='train',
            download=True,
            transform=transform
        )

    elif cfg.dataset_name.lower() == "bcss":
        print(f"Loading BCSS patches from {cfg.bcss_path}...")
        if not os.path.exists(cfg.bcss_path):
            raise FileNotFoundError(f"BCSS path not found: {cfg.bcss_path}. Run preprocess_bcss.py first.")

        # ImageFolder requires structure: root/class/image.png
        # preprocess_bcss.py creates: root/data/image.png (so 'data' is the class)
        ds = datasets.ImageFolder(root=cfg.bcss_path, transform=transform)
        print(f"Found {len(ds)} BCSS patches.")

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset_name}")

    ds = RobustDataset(ds)

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True
    )