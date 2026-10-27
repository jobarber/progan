"""Building blocks from Section 4 of the ProGAN paper."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EqualizedLinear(nn.Module):
    """Linear layer with the equalized learning rate trick.

    Weights are stored as N(0, 1) and scaled by the He constant at runtime, so every
    layer sees the same effective learning rate under Adam.
    """

    def __init__(self, in_features: int, out_features: int, gain: float = math.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = gain / math.sqrt(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConv2d(nn.Module):
    """3x3 / 1x1 / 4x4 convolution with the equalized learning rate trick."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        gain: float = math.sqrt(2),
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = gain / math.sqrt(in_channels * kernel_size * kernel_size)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class PixelNorm(nn.Module):
    """Normalize each pixel's feature vector to unit RMS (generator only)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)


class MinibatchStd(nn.Module):
    """Append the average per-group feature std as one extra channel.

    Lets the discriminator see batch-level statistics, which pushes the generator
    toward more varied samples.
    """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        g = min(self.group_size, n)
        while n % g:  # fall back to the largest group size that divides the batch
            g -= 1
        y = x.float().reshape(g, -1, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=(1, 2, 3)).to(x.dtype)  # [n/g]
        y = y.repeat(g).reshape(n, 1, 1, 1).expand(n, 1, h, w)
        return torch.cat([x, y], dim=1)
