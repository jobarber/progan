import math

import pytest
import torch
import torch.nn.functional as F

from progan.layers import EqualizedConv2d, EqualizedLinear, MinibatchStd, PixelNorm
from progan.models import Discriminator, Generator, channels_at, num_stages

SMALL = dict(max_resolution=32, fmap_base=256, fmap_max=32)


@pytest.fixture
def models():
    torch.manual_seed(0)
    return Generator(latent_dim=16, **SMALL), Discriminator(**SMALL)


def test_num_stages():
    assert num_stages(4) == 1
    assert num_stages(128) == 6
    with pytest.raises(ValueError):
        num_stages(96)


def test_channel_schedule_caps_at_fmap_max():
    assert channels_at(4, 8192, 256) == 256
    assert channels_at(128, 8192, 256) == 64


def test_equalized_layers_store_unit_weights_and_scale_at_runtime():
    torch.manual_seed(0)
    conv = EqualizedConv2d(64, 64, 3, padding=1)
    assert conv.weight.std().item() == pytest.approx(1.0, abs=0.05)
    assert conv.scale == pytest.approx(math.sqrt(2) / math.sqrt(64 * 9))
    lin = EqualizedLinear(100, 10, gain=1.0)
    x = torch.randn(4, 100)
    assert torch.allclose(lin(x), x @ (lin.weight * 0.1).T)


def test_pixelnorm_gives_unit_rms():
    y = PixelNorm()(torch.randn(2, 8, 4, 4) * 5)
    assert torch.allclose(y.pow(2).mean(1), torch.ones(2, 4, 4), atol=1e-4)


@pytest.mark.parametrize("batch", [8, 6, 5, 1])
def test_minibatch_std_adds_one_channel(batch):
    y = MinibatchStd(group_size=4)(torch.randn(batch, 3, 4, 4))
    assert y.shape == (batch, 4, 4, 4)
    assert torch.isfinite(y).all()


def test_minibatch_std_is_constant_per_group():
    x = torch.randn(8, 3, 4, 4)
    y = MinibatchStd(group_size=4)(x)[:, -1]
    # Samples i and i + n/g share a group, so they share the statistic.
    assert torch.allclose(y[0], y[2]) and torch.allclose(y[1], y[3])


@pytest.mark.parametrize("stage", range(4))
def test_shapes_at_every_stage(models, stage):
    G, D = models
    res = 4 * 2**stage
    imgs = G(torch.randn(3, 16), stage, alpha=0.5)
    assert imgs.shape == (3, 3, res, res)
    assert D(imgs, stage, alpha=0.5).shape == (3,)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_generator_fade_in_starts_from_previous_stage(models, stage):
    """At alpha=0 the new block contributes nothing: output = upsampled lower stage."""
    G, _ = models
    z = torch.randn(2, 16)
    low = F.interpolate(G(z, stage - 1), scale_factor=2, mode="nearest")
    assert torch.allclose(G(z, stage, alpha=0.0), low, atol=1e-5)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_discriminator_fade_in_starts_from_previous_stage(models, stage):
    _, D = models
    D.eval()
    img = torch.randn(4, 3, 4 * 2**stage, 4 * 2**stage)
    assert torch.allclose(D(img, stage, alpha=0.0), D(F.avg_pool2d(img, 2), stage - 1), atol=1e-5)


def test_inactive_blocks_get_no_gradient(models):
    G, _ = models
    G(torch.randn(2, 16), stage=1).mean().backward()
    assert G.blocks[0].conv1.weight.grad is not None
    assert G.blocks[1].conv1.weight.grad is None
