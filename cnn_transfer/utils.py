"""Shared helpers: device selection, seeding and figure output."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to disk; works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUTPUT_DIR = Path("outputs")


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_figure(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def save_image_samples(images: torch.Tensor, name: str, gray: bool = False) -> Path:
    """Save a row of sample images (C, H, W tensors) to outputs/."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 1.5))
    axes = np.atleast_1d(axes)
    for ax, image in zip(axes, images):
        array = np.transpose(image.cpu().numpy(), [1, 2, 0])
        if gray:
            ax.imshow(array.squeeze(-1), cmap="gray")
        else:
            ax.imshow(np.clip(array, 0, 1))
        ax.axis("off")
    return save_figure(fig, name)
