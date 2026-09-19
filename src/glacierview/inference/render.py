"""Render predictions as GIFs and area plots."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # headless: these run on servers and in batch
import matplotlib.pyplot as plt

from glacierview import config
from glacierview.log import get_logger

logger = get_logger(__name__)

#: Keep every Nth frame; a 460-scene series is too long to animate whole.
DEFAULT_FRAME_STRIDE = 5


def write_gif(
    gif_path: Path,
    scratch_dir: Path,
    images,
    probabilities,
    file_names: list[str],
    stride: int = DEFAULT_FRAME_STRIDE,
    overlay: bool = False,
) -> Path:
    """Animate the prediction over the series.

    Args:
        overlay: if True, draw the mask on top of the RGB composite (the batch
            style); otherwise show them as separate panels (the single style).
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for stale in scratch_dir.iterdir():
        stale.unlink()

    masks = probabilities.numpy()
    for i in range(0, len(file_names), stride):
        rgb = images[i][:, :, [2, 1, 0]]  # red, green, blue
        fig, axs = plt.subplots(2, figsize=(10, 10))
        fig.suptitle(file_names[i])
        axs[0].imshow(rgb)
        if overlay:
            axs[1].imshow(rgb)
            axs[1].imshow(masks[i, 0], alpha=0.2)
        else:
            axs[1].imshow(masks[i, 0])
        fig.savefig(scratch_dir / f"{file_names[i]}.png", dpi=100)
        plt.close(fig)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(gif_path, mode="I") as writer:
        for frame in sorted(os.listdir(scratch_dir)):
            writer.append_data(imageio.imread(scratch_dir / frame))
    logger.info("wrote %s", gif_path)
    return gif_path


def plot_area_series(out_path: Path, dates, areas, title: str) -> Path:
    """Plot a km^2 time series using the project's shared style."""
    style = config.mpl_style()
    with plt.style.context(str(style)) if style.exists() else _null_context():
        fig, ax = plt.subplots()
        ax.plot(dates, areas)
        ax.set_title(title)
        ax.set_ylabel("km$^2$")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
