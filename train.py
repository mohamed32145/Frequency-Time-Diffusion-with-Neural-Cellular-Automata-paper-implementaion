import time
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.utils import save_image

from config import TrainConfig
from utils import normalize, denormalize, EMA
from diffusion import DDPMSchedule, sample_ddpm


@torch.no_grad()
def test_model(model, cfg: TrainConfig, step):
    model.eval()
    save_path = os.path.join(cfg.checkpoint_dir, 'samples', f'step_{step:07d}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    samples = sample_ddpm(
        model,
        shape=(4, 3, 64, 64),
        device=cfg.device,
        T=cfg.T,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end
    )

    samples_save = denormalize(samples)
    save_image(samples_save, save_path, nrow=2, normalize=False)

    model.train()
    return samples.mean().item(), samples.std().item()


def train_runner(model, dataloader, cfg: TrainConfig):
    model.train()
    opt = Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    lr_sched = ExponentialLR(opt, gamma=cfg.lr_gamma)
    ddpm = DDPMSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=cfg.device)
    ema = EMA(model, decay=0.9999)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    step = 180000
    start_time = time.time()
    it = iter(dataloader)

    while step < cfg.train_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)

        x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        x0 = normalize(x0.to(cfg.device))

        t = torch.randint(0, cfg.T, (x0.shape[0],), device=cfg.device).long()
        noise = torch.randn_like(x0)
        x_t = ddpm.q_sample(x0, t, noise)

        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise) + F.l1_loss(pred_noise, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update()
        lr_sched.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - start_time
            cur_lr = opt.param_groups[0]['lr']
            print(f"Step {step} | Loss {loss.item():.6f} | LR {cur_lr:.2e} | Time {elapsed:.1f}s")

        if step % cfg.test_every == 0 and step > 0:
            print(f"Testing at step {step}...")
            ema.apply_shadow()
            mean, std = test_model(model, cfg, step)
            print(f"Stats: Mean {mean:.4f} Std {std:.4f}")
            ema.restore()

        if step % cfg.save_every == 0 and step > 0:
            ema.apply_shadow()
            ckpt = {
                'step': step,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'ema': ema.shadow
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, f'checkpoint_{step}.pt'))
            print(f"Saved checkpoint {step}")
            ema.restore()

        step += 1

    ema.apply_shadow()
    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'final.pt'))