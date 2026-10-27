"""Image data as a uint8 memmap at full resolution; lower stages are downsampled on GPU.

Decoding JPEGs every step dominates training time at low resolutions, so `prepare`
converts a dataset once into an (N, C, H, W) uint8 array and training just slices it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _center_square(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img).transpose(2, 0, 1)


def write_cache(images: Iterable[Image.Image], out: Path, resolution: int, limit: int) -> Path:
    """Write up to `limit` images into `out` (.npy, uint8, NCHW)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.uint8, shape=(limit, 3, resolution, resolution)
    )
    n = 0
    for img in tqdm(images, total=limit, desc="caching"):
        if n == limit:
            break
        arr[n] = _center_square(img, resolution)
        n += 1
    arr.flush()
    del arr
    if n < limit:  # fewer images than requested: rewrite with the true length
        trimmed = np.load(out, mmap_mode="r")[:n].copy()
        np.save(out, trimmed)
    return out


def images_from_folder(root: Path) -> Iterator[Image.Image]:
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(p) as img:
                yield img.copy()


def images_from_hub(
    name: str, split: str = "train", column: str = "image"
) -> Iterator[Image.Image]:
    from datasets import load_dataset  # optional extra

    for row in load_dataset(name, split=split, streaming=True):
        yield row[column]


class ImageCache:
    """Random minibatches from a uint8 cache, returned in [-1, 1] at any stage resolution."""

    def __init__(self, path: Path, device: torch.device | str = "cpu", seed: int = 0):
        self.data = np.load(path, mmap_mode="r")
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.resolution = self.data.shape[-1]

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, resolution: int, flip: bool = True) -> torch.Tensor:
        idx = np.sort(self.rng.choice(len(self.data), batch_size, replace=False))
        x = torch.from_numpy(self.data[idx]).to(self.device, non_blocking=True).float()
        x = x / 127.5 - 1.0
        if flip:
            mask = torch.rand(batch_size, device=self.device) < 0.5
            x = torch.where(mask[:, None, None, None], x.flip(3), x)
        return downsample(x, resolution)


def downsample(x: torch.Tensor, resolution: int) -> torch.Tensor:
    factor = x.shape[-1] // resolution
    return F.avg_pool2d(x, factor) if factor > 1 else x


def fade_reals(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend real images toward their upsampled half-resolution copy during a fade-in.

    The generator's output is itself a blend of the new block and the upsampled previous
    stage, so the discriminator should see reals degraded the same way. Without this,
    every transition hands the discriminator an easy, destabilizing signal.
    """
    if alpha >= 1.0:
        return x
    low = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
    return torch.lerp(low, x, alpha)
