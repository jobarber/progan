"""Progressively grown generator and discriminator.

All resolution blocks are built up front. `stage` selects how many are active
(stage 0 = 4x4, stage k = 4 * 2**k) and `alpha` blends the newest block in, as in
Figure 2 of the paper. Building everything up front keeps checkpoints as plain
state dicts and lets one optimizer live for the whole run.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from progan.layers import EqualizedConv2d, EqualizedLinear, MinibatchStd, PixelNorm


def num_stages(max_resolution: int) -> int:
    stages = int(math.log2(max_resolution)) - 1
    if 4 * 2 ** (stages - 1) != max_resolution:
        raise ValueError(f"max_resolution must be a power of two >= 4, got {max_resolution}")
    return stages


def channels_at(resolution: int, fmap_base: int, fmap_max: int) -> int:
    return min(fmap_base // resolution, fmap_max)


def _lrelu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.2)


class GeneratorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.norm = PixelNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.norm(_lrelu(self.conv1(x)))
        return self.norm(_lrelu(self.conv2(x)))


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        max_resolution: int = 128,
        fmap_base: int = 8192,
        fmap_max: int = 512,
        img_channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_stages = num_stages(max_resolution)
        ch = [channels_at(4 * 2**s, fmap_base, fmap_max) for s in range(self.n_stages)]

        self.input_norm = PixelNorm()
        # Paper: dense 512 -> 4x4x512 with gain sqrt(2)/4, then a 3x3 conv.
        self.dense = EqualizedLinear(latent_dim, ch[0] * 16, gain=math.sqrt(2) / 4)
        self.stem_conv = EqualizedConv2d(ch[0], ch[0], 3, padding=1)
        self.norm = PixelNorm()
        self.blocks = nn.ModuleList(
            GeneratorBlock(ch[s - 1], ch[s]) for s in range(1, self.n_stages)
        )
        self.to_rgb = nn.ModuleList(EqualizedConv2d(c, img_channels, 1, gain=1.0) for c in ch)

    def forward(self, z: torch.Tensor, stage: int, alpha: float = 1.0) -> torch.Tensor:
        x = self.dense(self.input_norm(z)).view(z.shape[0], -1, 4, 4)
        x = self.norm(_lrelu(x))
        x = self.norm(_lrelu(self.stem_conv(x)))
        for s in range(1, stage + 1):
            if s == stage and alpha < 1.0:
                skip = F.interpolate(self.to_rgb[s - 1](x), scale_factor=2, mode="nearest")
                x = self.blocks[s - 1](x)
                return torch.lerp(skip, self.to_rgb[s](x), alpha)
            x = self.blocks[s - 1](x)
        return self.to_rgb[stage](x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _lrelu(self.conv1(x))
        x = _lrelu(self.conv2(x))
        return F.avg_pool2d(x, 2)


class Discriminator(nn.Module):
    def __init__(
        self,
        max_resolution: int = 128,
        fmap_base: int = 8192,
        fmap_max: int = 512,
        img_channels: int = 3,
        mbstd_group_size: int = 4,
    ):
        super().__init__()
        self.n_stages = num_stages(max_resolution)
        ch = [channels_at(4 * 2**s, fmap_base, fmap_max) for s in range(self.n_stages)]

        self.from_rgb = nn.ModuleList(EqualizedConv2d(img_channels, c, 1) for c in ch)
        # blocks[s] maps resolution 4 * 2**s down to half that.
        self.blocks = nn.ModuleList(
            [nn.Identity()]
            + [DiscriminatorBlock(ch[s], ch[s - 1]) for s in range(1, self.n_stages)]
        )
        self.mbstd = MinibatchStd(mbstd_group_size)
        self.final_conv = EqualizedConv2d(ch[0] + 1, ch[0], 3, padding=1)
        self.final_dense = EqualizedLinear(ch[0] * 16, ch[0])
        self.out = EqualizedLinear(ch[0], 1, gain=1.0)

    def forward(self, img: torch.Tensor, stage: int, alpha: float = 1.0) -> torch.Tensor:
        x = _lrelu(self.from_rgb[stage](img))
        if stage > 0:
            x = self.blocks[stage](x)
            if alpha < 1.0:
                skip = _lrelu(self.from_rgb[stage - 1](F.avg_pool2d(img, 2)))
                x = torch.lerp(skip, x, alpha)
            for s in range(stage - 1, 0, -1):
                x = self.blocks[s](x)
        x = _lrelu(self.final_conv(self.mbstd(x)))
        x = _lrelu(self.final_dense(x.flatten(1)))
        return self.out(x).squeeze(1)
