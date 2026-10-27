"""`progan` command line: prepare, train, sample, interpolate, fid, plot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from progan.train import TrainConfig, train


@dataclass
class Prepare:
    """Cache a dataset as a uint8 .npy array at the target resolution."""

    out: Path = Path("data/celeba128.npy")
    hub: str | None = "tglcourse/CelebA-faces-cropped-128"
    """Hugging Face dataset id (ignored if --folder is given)."""
    folder: Path | None = None
    resolution: int = 128
    limit: int = 50_000

    def run(self) -> None:
        from progan.data import images_from_folder, images_from_hub, write_cache

        images = images_from_folder(self.folder) if self.folder else images_from_hub(self.hub)
        write_cache(images, self.out, self.resolution, self.limit)


@dataclass
class Sample:
    """Write a grid of samples from a trained generator."""

    checkpoint: Path
    out: Path = Path("samples.png")
    n: int = 64
    seed: int = 0
    device: str = "cuda"

    def run(self) -> None:
        from progan.sample import load_generator, sample_grid

        G, stage = load_generator(self.checkpoint, self.device)
        print(sample_grid(G, stage, self.n, self.seed, self.out, self.device))


@dataclass
class Interpolate:
    """Write a looping GIF of spherical interpolations in latent space."""

    checkpoint: Path
    out: Path = Path("interpolation.gif")
    seed: int = 7
    device: str = "cuda"

    def run(self) -> None:
        from progan.sample import interpolation_gif, load_generator

        G, stage = load_generator(self.checkpoint, self.device)
        print(interpolation_gif(G, stage, self.out, self.device, seed=self.seed))


@dataclass
class Fid:
    """Frechet Inception Distance between generated and cached real images."""

    checkpoint: Path
    data: Path
    n: int = 10_000
    device: str = "cuda"

    def run(self) -> None:
        from progan.sample import compute_fid, load_generator

        G, stage = load_generator(self.checkpoint, self.device)
        fid = compute_fid(G, stage, self.data, self.n, device=self.device)
        print(f"FID@{self.n // 1000}k: {fid:.2f}")


@dataclass
class Plot:
    """Training curve and growth strip for a finished run."""

    run_dir: Path
    out_dir: Path = Path("assets")

    def run(self) -> None:
        from progan.sample import growth_strip, plot_training

        print(plot_training(self.run_dir, self.out_dir / "training.png"))
        print(growth_strip(self.run_dir, self.out_dir / "growth.png"))


@dataclass
class Train:
    """Train a progressively grown GAN."""

    config: Annotated[TrainConfig, tyro.conf.arg(name="")]

    def run(self) -> None:
        print(train(self.config))


def main() -> None:
    cmd = tyro.extras.subcommand_cli_from_dict(
        {
            "prepare": Prepare,
            "train": Train,
            "sample": Sample,
            "interpolate": Interpolate,
            "fid": Fid,
            "plot": Plot,
        }
    )
    cmd.run()


if __name__ == "__main__":
    main()
