"""Fast checks on random tensors; no dataset download needed."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_transfer import models
from cnn_transfer.training import accuracy, train_model


def _loader(channels: int, size: int, n: int = 32) -> DataLoader:
    x = torch.rand(n, channels, size, size)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_cifar_cnn_shapes():
    for activation in ("leakyrelu", "tanh"):
        out = models.cifar_cnn(activation)(torch.rand(4, 3, 32, 32))
        assert out.shape == (4, 10)


def test_mnist_cnn_shapes():
    assert models.mnist_cnn()(torch.rand(4, 1, 28, 28)).shape == (4, 10)


def test_alexnet_head_and_freezing():
    model = models.pretrained_alexnet(10, freeze_backbone=True, pretrained=False)
    assert model(torch.rand(2, 3, 64, 64)).shape == (2, 10)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 2  # only the new Linear layer's weight and bias


def test_training_loop_runs():
    device = torch.device("cpu")
    model = models.cifar_cnn("leakyrelu")
    loader = _loader(3, 32)
    best, history = train_model(
        model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), 1e-3), loader, loader, 2, device
    )
    assert len(history.training_loss) == 2
    assert 0.0 <= accuracy(best, loader, device) <= 1.0
