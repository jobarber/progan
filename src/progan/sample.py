"""Inference utilities: load a trained generator, sample grids, latent walks, FID, plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from progan.data import ImageCache
from progan.models import Generator


def load_generator(path: Path, device: str = "cpu") -> tuple[Generator, int]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    G = Generator(
        latent_dim=cfg["latent_dim"],
        max_resolution=cfg["max_resolution"],
        fmap_base=cfg["fmap_base"],
        fmap_max=cfg["fmap_max"],
    )
    G.load_state_dict(ckpt["state_dict"])
    return G.to(device).eval(), ckpt["stage"]


def to_unit(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def sample_grid(G: Generator, stage: int, n: int, seed: int, out: Path, device: str) -> Path:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, G.latent_dim, generator=g).to(device)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_unit(G(z, stage)), out, nrow=int(np.ceil(np.sqrt(n))), padding=2, pad_value=1.0)
    return out


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    a_n, b_n = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((a_n * b_n).sum(-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


@torch.no_grad()
def interpolation_gif(
    G: Generator,
    stage: int,
    out: Path,
    device: str,
    n_rows: int = 1,
    n_cols: int = 4,
    keyframes: int = 5,
    steps: int = 16,
    seed: int = 7,
) -> Path:
    """Loop through `keyframes` random latents with spherical interpolation."""
    import imageio.v3 as iio

    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(keyframes, n_rows * n_cols, G.latent_dim, generator=g).to(device)
    frames = []
    for k in range(keyframes):
        a, b = keys[k], keys[(k + 1) % keyframes]
        for i in range(steps):
            t = 0.5 - 0.5 * np.cos(np.pi * i / steps)  # ease in/out
            imgs = to_unit(G(slerp(a, b, float(t)), stage))
            grid = make_grid(imgs, nrow=n_cols, padding=2, pad_value=1.0)
            frames.append((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, duration=60, loop=0)
    return out


@torch.no_grad()
def compute_fid(
    G: Generator,
    stage: int,
    data: Path,
    n: int = 10_000,
    batch: int = 100,
    device: str = "cuda",
) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    cache = ImageCache(data, device=device, seed=123)
    res = 4 * 2**stage
    for _ in range(n // batch):
        fid.update(to_unit(cache.sample(batch, res, flip=False)), real=True)
        z = torch.randn(batch, G.latent_dim, device=device)
        fid.update(to_unit(G(z, stage)), real=False)
    return float(fid.compute())


def plot_training(run_dir: Path, out: Path) -> Path:
    """Wasserstein estimate over training, with each resolution shaded."""
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    kimg = np.array([r["kimg"] for r in rows])
    wdist = np.array([r["wdist"] for r in rows])
    res = np.array([r["resolution"] for r in rows])
    alpha = np.array([r["alpha"] for r in rows])

    smooth = np.convolve(wdist, np.ones(25) / 25, mode="same")
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
    ax.plot(kimg, wdist, color="#9db4d6", lw=0.6, alpha=0.6)
    ax.plot(kimg, smooth, color="#1f4e8c", lw=1.6, label="Wasserstein estimate (smoothed)")
    for r in np.unique(res):
        m = res == r
        start = kimg[m].min()
        fade_end = kimg[m & (alpha >= 1.0)].min() if (m & (alpha >= 1.0)).any() else kimg[m].max()
        ax.axvspan(start, fade_end, color="#f2d9b3", alpha=0.6, lw=0)
        ax.text(start, ax.get_ylim()[1], f" {r}px", va="top", fontsize=8, color="#555")
        ax.axvline(start, color="#bbb", lw=0.6)
    ax.set_xlabel("real images shown (thousands)")
    ax.set_ylabel("E[D(real)] - E[D(fake)]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def growth_strip(run_dir: Path, out: Path, n: int = 4) -> Path:
    """One row per latent, one column per resolution, from the growth/ snapshots."""
    from PIL import Image

    snaps = sorted(
        (run_dir / "growth").glob("stage*_*px.png"), key=lambda p: int(p.stem[5:].split("_")[0])
    )
    tiles = [Image.open(p) for p in snaps]
    full = tiles[0].width // 4  # snapshots are 4x4 grids at full resolution
    strip = Image.new("RGB", (full * len(tiles), full * n), "white")
    for col, img in enumerate(tiles):
        for row in range(n):
            tile = img.crop((row * full, 0, (row + 1) * full, full))
            strip.paste(tile, (col * full, row * full))
    out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out)
    return out
