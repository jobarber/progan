import json

import numpy as np
import pytest
import torch
from PIL import Image

from progan.data import ImageCache, downsample, fade_reals, images_from_folder, write_cache
from progan.losses import discriminator_loss, gradient_penalty
from progan.models import Discriminator
from progan.sample import load_generator, slerp
from progan.train import TrainConfig, phase_at, total_kimg, train, update_ema


def test_schedule_fades_then_stabilizes():
    cfg = TrainConfig(data=None, max_resolution=16, fade_kimg=10, stable_kimg=20)
    assert phase_at(cfg, 0) == (0, 1.0)
    assert phase_at(cfg, 19.9) == (0, 1.0)
    assert phase_at(cfg, 25) == (1, 0.5)
    assert phase_at(cfg, 31) == (1, 1.0)
    assert phase_at(cfg, 55) == (2, 0.5)
    assert phase_at(cfg, total_kimg(cfg) + 5) == (2, 1.0)
    assert total_kimg(cfg) == 20 + 2 * 30


def test_gradient_penalty_is_zero_for_a_unit_gradient_critic():
    class Linear(torch.nn.Module):
        def forward(self, x, stage, alpha):
            w = torch.ones_like(x[0]) / x[0].numel() ** 0.5  # ||grad|| == 1 everywhere
            return (x * w).flatten(1).sum(1)

    x = torch.randn(4, 3, 8, 8)
    gp = gradient_penalty(Linear(), x, torch.randn_like(x), 1, 1.0)
    assert torch.allclose(gp, torch.zeros(4), atol=1e-6)


def test_discriminator_loss_backpropagates():
    D = Discriminator(max_resolution=8, fmap_base=64, fmap_max=16)
    x = torch.randn(4, 3, 8, 8)
    loss, stats = discriminator_loss(D, x, torch.randn_like(x), 1, 0.3)
    loss.backward()
    assert set(stats) == {"wdist", "gp"}
    assert all(p.grad is not None for p in D.from_rgb[1].parameters())


def test_ema_moves_toward_model():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = a.weight.clone()
    update_ema(a, b, beta=0.9)
    assert torch.allclose(a.weight, 0.9 * before + 0.1 * b.weight)


def test_slerp_endpoints():
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-5)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-5)


@pytest.fixture
def image_cache(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(12):
        arr = rng.integers(0, 255, (20, 24, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"{i}.png")
    return write_cache(images_from_folder(folder), tmp_path / "cache.npy", 16, limit=50)


def test_cache_is_trimmed_and_scaled(image_cache):
    cache = ImageCache(image_cache)
    assert len(cache) == 12 and cache.resolution == 16
    x = cache.sample(4, 8)
    assert x.shape == (4, 3, 8, 8)
    assert x.min() >= -1 and x.max() <= 1


def test_downsample_is_average_pooling():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    assert downsample(x, 2).flatten().tolist() == [2.5, 4.5, 10.5, 12.5]


def test_fade_reals_matches_generator_blend():
    x = torch.randn(2, 3, 8, 8)
    low = torch.nn.functional.interpolate(downsample(x, 4), scale_factor=2, mode="nearest")
    assert torch.allclose(fade_reals(x, 0.0), low)
    assert torch.equal(fade_reals(x, 1.0), x)


def test_end_to_end_training_and_resume(image_cache, tmp_path):
    cfg = TrainConfig(
        data=image_cache,
        out_dir=tmp_path / "run",
        max_resolution=16,
        latent_dim=16,
        fmap_base=64,
        fmap_max=16,
        fade_kimg=0.016,
        stable_kimg=0.016,
        batch_sizes={4: 4, 8: 4, 16: 4},
        log_every=1,
        sample_every_kimg=0.02,
        ckpt_every_kimg=0.04,
        device="cpu",
    )
    final = train(cfg)
    rows = [json.loads(r) for r in (cfg.out_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["resolution"] == 16
    assert len(list((cfg.out_dir / "growth").glob("*.png"))) == 3
    assert len(list(cfg.out_dir.glob("checkpoint_*px.pt"))) == 3

    G, stage = load_generator(final)
    assert stage == 2
    assert G(torch.randn(2, 16), stage).shape == (2, 3, 16, 16)

    # A second call resumes from the checkpoint and has nothing left to do.
    n_rows = len(rows)
    train(cfg)
    assert len((cfg.out_dir / "metrics.jsonl").read_text().splitlines()) == n_rows
