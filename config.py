from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DiffNCAConfig:
    image_channels: int = 3
    pred_channels: int = 3
    #  (3 image + 3 noise + 90 state)
    state_channels: int = 90

    # "maps to a hidden vector of depth h = 512"
    hidden: int = 512

    # "fire rate, which is set to a probability of 90%"
    fire_rate: float = 0.9

    #  "20 steps in the image space" is the optimal setting
    nca_steps: int = 20

    # "linear layer of size 256" for embedding
    emb_dim: int = 256


@dataclass
class FourierDiffNCAConfig:
    diff_cfg: DiffNCAConfig = field(default_factory=DiffNCAConfig)

    #  "After 32 iterations... in Fourier space"
    fourier_steps: int = 32

    # "extract a 16x16 cell window" from 64x64 input [cite: 151]
    # 16 / 64 = 0.25. If your code uses this as a fraction of resolution, use 0.25.
    freq_crop: float = 0.25


@dataclass
class DataConfig:
    dataset_name: str = "celeba"
    data_root: str = "./data"
    bcss_path: str = r"C:\Users\Lab2\Datasets\BCSS_Patches_64\train"
    image_size: int = 64

    # Paper: "utilizing a batch size of 16"
    # Note: 128 might be too large for the 512-hidden-size model on some GPUs.
    batch_size: int = 16

    num_workers: int = 8


@dataclass
class TrainConfig:
    #  "learning rate of 1.6 x 10^-3"
    lr: float = 1.6e-3

    # "learning rate gamma of 0.9999"
    lr_gamma: float = 0.9999

    # "betas for the learning rate as (0.9, 0.99)"
    betas: Tuple[float, float] = (0.9, 0.99)

    #"epsilon... set at 1 x 10^-8"
    eps: float = 1e-8

    #  "models undergo training for 200,000 steps"
    train_steps: int = 200_000

    # Diffusion params (Standard DDPM settings implied by context, but not explicitly fixed in text snippet)
    T: int = 1000  # Standard DDPM usually uses 1000, user had 300.
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Checkpointing
    log_every: int = 200
    save_every: int = 10_000
    test_every: int = 1_000
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"