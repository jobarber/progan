"""Training loop: one D step and one G step per minibatch, EMA generator, resumable."""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from progan.data import ImageCache, fade_reals
from progan.losses import discriminator_loss, generator_loss
from progan.models import Discriminator, Generator, num_stages


@dataclass
class TrainConfig:
    data: Path
    """uint8 .npy cache written by `progan prepare`."""
    out_dir: Path = Path("runs/celeba128")
    max_resolution: int = 128
    latent_dim: int = 512
    fmap_base: int = 8192
    fmap_max: int = 256
    fade_kimg: int = 600
    """Thousands of real images shown while a new block fades in."""
    stable_kimg: int = 600
    """Thousands of real images shown at each resolution after fading."""
    final_kimg: int = 0
    """Extra thousands of images at the last resolution."""
    batch_sizes: dict[int, int] = field(
        default_factory=lambda: {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 32, 256: 16}
    )
    lr: float = 1e-3
    betas: tuple[float, float] = (0.0, 0.99)
    ema_beta: float = 0.999
    gp_weight: float = 10.0
    drift_weight: float = 1e-3
    log_every: int = 100
    sample_every_kimg: int = 50
    ckpt_every_kimg: int = 250
    seed: int = 0
    device: str = "cuda"


def phase_at(cfg: TrainConfig, kimg: float) -> tuple[int, float]:
    """Map images seen (in thousands) to (stage, alpha) under the fade/stable schedule."""
    n = num_stages(cfg.max_resolution)
    if kimg < cfg.stable_kimg:
        return 0, 1.0
    kimg -= cfg.stable_kimg
    for stage in range(1, n):
        if kimg < cfg.fade_kimg:
            return stage, kimg / cfg.fade_kimg
        kimg -= cfg.fade_kimg
        if kimg < cfg.stable_kimg:
            return stage, 1.0
        kimg -= cfg.stable_kimg
    return n - 1, 1.0


def total_kimg(cfg: TrainConfig) -> int:
    n = num_stages(cfg.max_resolution)
    return cfg.stable_kimg + (n - 1) * (cfg.fade_kimg + cfg.stable_kimg) + cfg.final_kimg


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, beta: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters(), strict=True):
        p_ema.lerp_(p, 1.0 - beta)


def build_models(cfg: TrainConfig) -> tuple[Generator, Discriminator]:
    kw = dict(max_resolution=cfg.max_resolution, fmap_base=cfg.fmap_base, fmap_max=cfg.fmap_max)
    return Generator(latent_dim=cfg.latent_dim, **kw), Discriminator(**kw)


@torch.no_grad()
def save_grid(G: Generator, z: torch.Tensor, stage: int, alpha: float, path: Path, size: int):
    imgs = G(z, stage, alpha)
    imgs = nn.functional.interpolate(imgs, size=size, mode="nearest")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(imgs.clamp(-1, 1) * 0.5 + 0.5, path, nrow=int(len(z) ** 0.5), padding=0)


def config_dict(cfg: TrainConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=str))


def _crossed(prev: float, now: float, every: float) -> bool:
    return every > 0 and int(prev // every) != int(now // every)


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg.device)
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    G, D = build_models(cfg)
    G, D = G.to(device), D.to(device)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)

    def make_optimizers():
        return (
            torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas, eps=1e-8),
            torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas, eps=1e-8),
        )

    g_opt, d_opt = make_optimizers()
    data = ImageCache(cfg.data, device=device, seed=cfg.seed)
    fixed_z = torch.randn(64, cfg.latent_dim, generator=torch.Generator().manual_seed(1234)).to(
        device
    )

    seen = 0
    ckpt_path = out / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        G_ema.load_state_dict(ckpt["G_ema"])
        g_opt.load_state_dict(ckpt["g_opt"])
        d_opt.load_state_dict(ckpt["d_opt"])
        seen = ckpt["seen"]
        print(f"resumed from {ckpt_path} at {seen / 1000:.0f} kimg")

    def checkpoint() -> None:
        state = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "G_ema": G_ema.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "seen": seen,
            "config": config_dict(cfg),
        }
        torch.save(state, ckpt_path.with_suffix(".tmp"))
        ckpt_path.with_suffix(".tmp").replace(ckpt_path)

    (out / "config.json").write_text(json.dumps(config_dict(cfg), indent=2))
    log = (out / "metrics.jsonl").open("a")
    end = total_kimg(cfg) * 1000
    step, t0, imgs_since = 0, time.perf_counter(), 0
    stage, alpha = phase_at(cfg, seen / 1000)
    opt_stage = stage

    while seen < end:
        stage, alpha = phase_at(cfg, seen / 1000)
        if stage != opt_stage:  # fresh Adam state for each resolution, as in the paper's code
            g_opt, d_opt = make_optimizers()
            opt_stage = stage
        res = 4 * 2**stage
        bs = cfg.batch_sizes[res]
        reals = fade_reals(data.sample(bs, res), alpha) if stage > 0 else data.sample(bs, res)

        D.requires_grad_(True)
        with torch.no_grad():
            fakes = G(torch.randn(bs, cfg.latent_dim, device=device), stage, alpha)
        d_loss, stats = discriminator_loss(
            D, reals, fakes, stage, alpha, cfg.gp_weight, cfg.drift_weight
        )
        d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        d_opt.step()

        D.requires_grad_(False)
        fake_scores = D(
            G(torch.randn(bs, cfg.latent_dim, device=device), stage, alpha), stage, alpha
        )
        g_loss = generator_loss(fake_scores)
        g_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        g_opt.step()
        update_ema(G_ema, G, cfg.ema_beta)

        prev, seen = seen, seen + bs
        step += 1
        imgs_since += bs

        if step % cfg.log_every == 0:
            dt = time.perf_counter() - t0
            row = {
                "kimg": round(seen / 1000, 3),
                "stage": stage,
                "resolution": res,
                "alpha": round(alpha, 4),
                "d_loss": round(d_loss.item(), 4),
                "g_loss": round(g_loss.item(), 4),
                "wdist": round(stats["wdist"], 4),
                "gp": round(stats["gp"], 4),
                "img_per_s": round(imgs_since / dt, 1),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            t0, imgs_since = time.perf_counter(), 0

        if _crossed(prev / 1000, seen / 1000, cfg.sample_every_kimg):
            path = out / "samples" / f"kimg{seen // 1000:06d}.png"
            save_grid(G_ema, fixed_z, stage, alpha, path, cfg.max_resolution)

        next_stage, _ = phase_at(cfg, seen / 1000)
        if next_stage != stage or seen >= end:
            # Same latents rendered at the end of every resolution: the growth strip.
            path = out / "growth" / f"stage{stage}_{res}px.png"
            save_grid(G_ema, fixed_z[:16], stage, alpha, path, cfg.max_resolution)
            checkpoint()  # keep one restart point per resolution
            shutil.copy(ckpt_path, out / f"checkpoint_{res}px.pt")

        if _crossed(prev / 1000, seen / 1000, cfg.ckpt_every_kimg):
            checkpoint()

    checkpoint()
    final = out / "generator_ema.pt"
    torch.save(
        {"state_dict": G_ema.state_dict(), "stage": stage, "config": config_dict(cfg)}, final
    )
    log.close()
    return final
