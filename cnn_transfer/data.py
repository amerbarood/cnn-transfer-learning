"""Dataset loading: CIFAR-10, MNIST and SVHN with an 80/20 train/validation split."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

DATA_DIR = "data/raw"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Loaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    num_classes: int


def _split_and_wrap(
    full: Dataset,
    test: Dataset,
    batch_size: int,
    eval_batch_size: int | None,
    num_classes: int,
) -> Loaders:
    train, validation = random_split(full, [0.8, 0.2])
    # The original notebooks evaluated validation/test in a single full-size batch.
    val_bs = eval_batch_size or len(validation)
    test_bs = eval_batch_size or len(test)
    loaders = Loaders(
        train=DataLoader(train, batch_size=batch_size, shuffle=False),
        validation=DataLoader(validation, batch_size=val_bs, shuffle=False),
        test=DataLoader(test, batch_size=test_bs, shuffle=False),
        num_classes=num_classes,
    )
    print(f"Train: {len(train)}  Validation: {len(validation)}  Test: {len(test)}")
    return loaders


def cifar10(
    batch_size: int,
    eval_batch_size: int | None = None,
    imagenet_style: bool = False,
) -> Loaders:
    """CIFAR-10. With imagenet_style, images are resized to 64px and ImageNet-normalised."""
    steps = [transforms.ToTensor()]
    if imagenet_style:
        steps += [transforms.Resize(64), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    transform = transforms.Compose(steps)
    full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)
    return _split_and_wrap(full, test, batch_size, eval_batch_size, len(full.classes))


def mnist(batch_size: int, eval_batch_size: int | None = None) -> Loaders:
    transform = transforms.ToTensor()
    full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return _split_and_wrap(full, test, batch_size, eval_batch_size, 10)


def svhn_as_mnist(batch_size: int, eval_batch_size: int | None = None) -> Loaders:
    """SVHN converted to 28x28 grayscale so an MNIST model can be applied to it."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop((28, 28)),
        ]
    )
    full = datasets.SVHN(DATA_DIR, split="train", download=True, transform=transform)
    test = datasets.SVHN(DATA_DIR, split="test", download=True, transform=transform)
    return _split_and_wrap(full, test, batch_size, eval_batch_size, 10)


def first_images(loader: DataLoader, n: int = 4) -> torch.Tensor:
    images, _ = next(iter(loader))
    return images[:n]
