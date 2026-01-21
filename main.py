import torch
import os
from config import DiffNCAConfig, FourierDiffNCAConfig, TrainConfig, DataConfig
from models import DiffNCA, FourierDiffNCA
from dataset import get_dataloader
from train import train_runner
from utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================================
    # 1. SELECT YOUR MODEL TYPE HERE
    # ==========================================
    MODEL_TYPE = "DiffNCA"

    if MODEL_TYPE == "DiffNCA":
        print(">>> Configuration: DiffNCA + Breast Cancer (BCSS) Dataset")

        # 1. Configure Model
        model_cfg = DiffNCAConfig()
        model = DiffNCA(model_cfg).to(device)

        # 2. Configure Dataset (BCSS)
        data_cfg = DataConfig(
            dataset_name="bcss",
            # Pointing to the TRAIN folder created by the preprocess script
            bcss_path=r"C:\Users\Lab2\PycharmProjects\final\data\BCSS_Patches_64\train",
            batch_size=32,
            num_workers=8
        )

        # 3. Checkpoint Directory
        checkpoint_dir = "checkpoints_bcss_diffnca"

    elif MODEL_TYPE == "FourierDiffNCA":
        print(">>> Configuration: FourierDiffNCA + CelebA Dataset")

        # 1. Configure Model
        model_cfg = FourierDiffNCAConfig()
        model = FourierDiffNCA(model_cfg).to(device)

        # 2. Configure Dataset (CelebA)
        data_cfg = DataConfig(
            dataset_name="celeba",
            data_root="./data",
            batch_size=32
        )

        # 3. Checkpoint Directory
        checkpoint_dir = "checkpoints_celeba_fourier"

    else:
        raise ValueError("Invalid MODEL_TYPE")

    # ==========================================
    # 2. SETUP TRAINING
    # ==========================================
    train_cfg = TrainConfig(
        device=device,
        train_steps=200_000,
        checkpoint_dir=checkpoint_dir,
        # Sync batch size
        #batch_size=data_cfg.batch_size
    )

    # ==========================================
    # 3. LOAD DATA
    # ==========================================
    try:
        dl = get_dataloader(data_cfg)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("-" * 60)
        print("Did you run 'preprocess_bcss.py' yet?")
        print("-" * 60)
        exit(1)

    # ==========================================
    # 4. CHECKPOINT LOADING
    # ==========================================
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    load_cp = input("Do you want to load a checkpoint? (yes/no): ").strip().lower()

    if load_cp in ['yes', 'y']:
        path = input(f"Enter path (default: {checkpoint_dir}/final.pt): ").strip()
        if not path: path = os.path.join(checkpoint_dir, "final.pt")

        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device)
                state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['model']
                model.load_state_dict(state)
                print("✓ Checkpoint loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load checkpoint: {e}")
        else:
            print("✗ Checkpoint not found, starting fresh.")

    # ==========================================
    # 5. RUN TRAINING
    # ==========================================
    print(f"\nStarting training on {MODEL_TYPE}...")
    train_runner(model, dl, train_cfg)