import torch
import torch.nn as nn
from typing import Tuple


class DDPMSchedule:
    def __init__(self, T: int = 300, beta_start: float = 1e-4, beta_end: float = 2e-2, device="cuda"):
        self.T = int(T)
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)
        return s1 * x0 + s2 * noise


@torch.no_grad()
def sample_ddpm(model: nn.Module, shape: Tuple[int, int, int, int], device: str, T: int = 300, beta_start=1e-4,
                beta_end=2e-2):
    model.eval()
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_theta = model(x, t_batch)

        a, ab, b = alphas[t], alpha_bar[t], betas[t]
        mu = (1.0 / torch.sqrt(a)) * (x - (b / torch.sqrt(1.0 - ab)) * eps_theta)

        if t > 0:
            z = torch.randn_like(x)
            x = mu + torch.sqrt(b) * z
        else:
            x = mu
    return x