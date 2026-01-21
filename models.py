import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from config import DiffNCAConfig, FourierDiffNCAConfig
import torch.utils.checkpoint as checkpoint

def sinusoidal_embedding(x: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=x.device, dtype=torch.float32) / (half - 1)
    )
    args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def make_xy_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx[None, None, ...], yy[None, None, ...]


class NCABlock(nn.Module):
    def __init__(self, channels: int, hidden: int = 128, cond_channels: int = 4, fire_rate: float = 1.0, groups: int = 8,initial_state=None):
        super().__init__()
        self.fire_rate = float(fire_rate)
        self.perc = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels + channels + cond_channels, hidden, 1)
        valid_groups = min(groups, hidden)
        while hidden % valid_groups != 0 and valid_groups > 1:
            valid_groups -= 1
        self.gn = nn.GroupNorm(num_groups=valid_groups, num_channels=hidden)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, state: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        p = self.perc(state)
        x = torch.cat([state, p, cond], dim=1)
        x = self.conv1(x)
        x = self.gn(x)
        x = self.act(x)
        delta = self.conv2(x)
        if self.fire_rate < 1.0:
            mask = (torch.rand(state.shape[0], 1, state.shape[2], state.shape[3], device=state.device) <= self.fire_rate)
            delta = delta * mask.to(delta.dtype)
        return state + delta


class DiffNCA(nn.Module):
    def __init__(self, cfg: DiffNCAConfig):
        super().__init__()
        self.cfg = cfg
        self.total_ch = cfg.pred_channels + cfg.state_channels
        self.init_proj = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.SiLU(), nn.Linear(cfg.emb_dim, cfg.pred_channels)
        )
        self.emb_lin1 = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.emb_lin2 = nn.Linear(cfg.emb_dim, 4 + cfg.image_channels)
        self.emb_act = nn.SiLU()
        self.nca = NCABlock(self.total_ch, hidden=cfg.hidden, cond_channels=4 + cfg.image_channels, fire_rate=cfg.fire_rate)

    def make_cond(self, H, W, x_t, t, s, device, dtype):
        xx, yy = make_xy_grid(H, W, device=device, dtype=dtype)
        B = t.shape[0]
        xx = xx.expand(B, -1, -1, -1)
        yy = yy.expand(B, -1, -1, -1)
        tt = t[:, None, None, None].expand(B, 1, H, W).to(dtype)
        ss = s[:, None, None, None].expand(B, 1, H, W).to(dtype)
        emb = (sinusoidal_embedding(xx.squeeze(1), self.cfg.emb_dim) + sinusoidal_embedding(yy.squeeze(1), self.cfg.emb_dim) +
               sinusoidal_embedding(tt.squeeze(1), self.cfg.emb_dim) + sinusoidal_embedding(ss.squeeze(1), self.cfg.emb_dim))
        emb = self.emb_act(self.emb_lin1(emb))
        out = self.emb_lin2(emb)
        out = out.permute(0, 3, 1, 2).contiguous()
        return torch.cat([out[:, :4], x_t], dim=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, initial_state=None) -> torch.Tensor:
        B, C, H, W = x_t.shape

        # 1. Standard Initialization (Used if no global info is provided)
        t_emb = sinusoidal_embedding(t.float(), self.cfg.emb_dim)
        noise_init = self.init_proj(t_emb)
        noise_slot = noise_init[:, :, None, None].expand(-1, -1, H, W)  # Default N
        mem = torch.zeros(B, self.cfg.state_channels, H, W, device=x_t.device, dtype=x_t.dtype)  # Default E

        # 2. Fourier Initialization (Method 3.B from Paper)
        if initial_state is not None:
            # The Fourier model outputs the full C = I + N + E (e.g., 96 channels)
            # We need to slice it to get the relevant parts for this NCA.
            # I (Image) = Channels 0:3  (Ignored here, x_t is used in make_cond)
            # N (Pred)  = Channels 3:6  (Use this as starting noise guess)
            # E (State) = Channels 6:96 (Use this as starting memory)

            # Indices assuming image_channels=3, pred_channels=3
            start_n = self.cfg.image_channels
            end_n = start_n + self.cfg.pred_channels

            # Extract N (Prediction) and E (Hidden State)
            # We overwrite the default initialization with the Fourier "Global Context"
            noise_slot = initial_state[:, start_n:end_n, :, :]
            mem = initial_state[:, end_n:, :, :]

        # 3. Construct Evolving State
        state = torch.cat([noise_slot, mem], dim=1)

        # 4. Run NCA Steps
        for step_idx in range(self.cfg.nca_steps):
            s = torch.full((B,), float(step_idx), device=x_t.device, dtype=x_t.dtype)
            cond = self.make_cond(H, W, x_t, t.to(x_t.dtype), s, x_t.device, x_t.dtype)
            state = self.nca(state, cond)

        return state[:, :self.cfg.pred_channels]





class FourierDiffNCA(nn.Module):
    def __init__(self, cfg: FourierDiffNCAConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Image Space NCA (The Refiner)
        self.image_nca = DiffNCA(cfg.diff_cfg)

        # Calculate the full state dimension (C = I + N + E)
        # [cite_start]Typically: 3 (Image) + 3 (Pred) + 90 (Hidden) = 96 channels [cite: 160]
        self.full_state_dim = (
                cfg.diff_cfg.image_channels +
                cfg.diff_cfg.pred_channels +
                cfg.diff_cfg.state_channels
        )

        # 2. Fourier Space NCA (The Initializer)
        # It must output the FULL state (Real + Imag), so output channels = 96 * 2 = 192
        fourier_inner_cfg = DiffNCAConfig(
            image_channels=cfg.diff_cfg.image_channels * 2,  # Input is FFT(x_t)

            # CHANGE: Output matches full state size (Real + Imag)
            pred_channels=self.full_state_dim * 2,

            state_channels=cfg.diff_cfg.state_channels,  # Internal memory for Fourier NCA
            hidden=cfg.diff_cfg.hidden,
            fire_rate=cfg.diff_cfg.fire_rate,
            nca_steps=cfg.fourier_steps,
            emb_dim=cfg.diff_cfg.emb_dim,
        )
        self.fourier_nca = DiffNCA(fourier_inner_cfg)

        # REMOVED: self.fusion layer is no longer needed

    @staticmethod
    def _fft2(x):
        X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.fftshift(X, dim=(-2, -1))

    @staticmethod
    def _ifft2(X):
        X = torch.fft.ifftshift(X, dim=(-2, -1))
        return torch.fft.ifft2(X, dim=(-2, -1), norm="ortho")

    def _center_crop(self, X):
        B, C, H, W = X.shape
        # Ensure we use the exact crop size logic
        ch = max(2, int(H * self.cfg.freq_crop))
        cw = max(2, int(W * self.cfg.freq_crop))
        y0, x0 = (H - ch) // 2, (W - cw) // 2
        ys, xs = slice(y0, y0 + ch), slice(x0, x0 + cw)
        return X[:, :, ys, xs], (ys, xs)

    def forward(self, x_t, t):
        # ---------------------------------------------------------
        # Phase 1: Global Communication in Fourier Space
        # ---------------------------------------------------------
        # 1. FFT
        X = self._fft2(x_t)

        # 2. Crop Center (Low Frequencies)
        Xc, (ys, xs) = self._center_crop(X)

        # 3. Run Fourier NCA
        # Input: (B, 6, Hc, Wc) -> Real+Imag of x_t
        Xc_ri = torch.cat([Xc.real, Xc.imag], dim=1)

        # Output: (B, 192, Hc, Wc) -> Real+Imag of FULL STATE (I+N+E)
        fourier_out_ri = self.fourier_nca(Xc_ri, t)

        # ---------------------------------------------------------
        # Phase 2: Translation to Image Space
        # ---------------------------------------------------------
        # 4. Reconstruct Complex Tensor
        # Split 192 channels back into 96 Real + 96 Imag
        fo_real, fo_imag = fourier_out_ri.chunk(2, dim=1)
        fourier_state_c = torch.complex(fo_real, fo_imag)  # (B, 96, Hc, Wc)

        # 5. Place back into Full Spectrum (Zero Padding)
        # We need a tensor of size (B, 96, H, W)
        full_spectrum = torch.zeros(
            x_t.shape[0], self.full_state_dim, x_t.shape[2], x_t.shape[3],
            device=x_t.device, dtype=torch.complex64
        )
        full_spectrum[:, :, ys, xs] = fourier_state_c

        # 6. Inverse FFT
        # This gives us the "Global Context" state in spatial domain
        spatial_state = self._ifft2(full_spectrum).real  # (B, 96, H, W)

        # ---------------------------------------------------------
        # Phase 3: Local Refinement in Image Space
        # ---------------------------------------------------------
        # 7. Run Image NCA sequentially
        # The Fourier result is passed as the starting point (initial_state)
        # Note: Your DiffNCA.forward() must support 'initial_state' argument
        pred_noise = self.image_nca(x_t, t, initial_state=spatial_state)

        return pred_noise