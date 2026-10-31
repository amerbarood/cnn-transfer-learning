"""Network definitions used in the experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

ACTIVATIONS = {"leakyrelu": nn.LeakyReLU, "tanh": nn.Tanh}


class SmallCNN(nn.Module):
    """Two conv layers + two fully connected layers.

    CIFAR-10 (3x32x32) uses in_channels=3, flat_size=7; MNIST (1x28x28) uses
    in_channels=1, flat_size=6. The conv blocks use `activation`, the hidden
    fully connected layer always uses ReLU, as in the original notebooks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        flat_size: int = 7,
        activation: str = "leakyrelu",
        num_classes: int = 10,
    ):
        super().__init__()
        act = ACTIVATIONS.get(activation, nn.ReLU)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 2),
            act(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 2),
            act(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * flat_size * flat_size, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def cifar_cnn(activation: str) -> SmallCNN:
    return SmallCNN(in_channels=3, flat_size=7, activation=activation)


def mnist_cnn() -> SmallCNN:
    return SmallCNN(in_channels=1, flat_size=6, activation="relu")


def pretrained_alexnet(num_classes: int, freeze_backbone: bool, pretrained: bool = True) -> nn.Module:
    """ImageNet AlexNet with an extra ReLU + Linear(1000 -> num_classes) head.

    freeze_backbone=False is fine-tuning (all weights train);
    freeze_backbone=True is feature extraction (only the new head trains).
    """
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = nn.Sequential(model.classifier, nn.ReLU(), nn.Linear(1000, num_classes))
    return model
