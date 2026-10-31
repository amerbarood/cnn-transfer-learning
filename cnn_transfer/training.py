"""Training loop with best-validation-loss checkpointing, plus evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import save_figure


@dataclass
class History:
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


@torch.no_grad()
def _validation_loss(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += criterion(model(images), labels).item()
        batches += 1
    return total / max(batches, 1)


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    device: torch.device,
    print_every: int = 10,
) -> tuple[nn.Module, History]:
    """Train for `epochs` and return a copy of the model with the lowest validation loss.

    As in the original notebooks, validation loss is measured at the start of
    each epoch (before that epoch's updates).
    """
    history = History()
    best_loss = float("inf")
    best_model = copy.deepcopy(model)

    for epoch in range(epochs):
        val_loss = _validation_loss(model, criterion, validation_loader, device)
        history.validation_loss.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            history.best_epoch = epoch

        model.train()
        total, batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
            batches += 1
        history.training_loss.append(total / max(batches, 1))

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch}, training loss: {history.training_loss[-1]:.4f}, "
                f"validation loss: {history.validation_loss[-1]:.4f}"
            )

    return best_model, history


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def report_accuracy(model: nn.Module, train_loader, validation_loader, test_loader, device) -> dict[str, float]:
    results = {
        "train": accuracy(model, train_loader, device),
        "validation": accuracy(model, validation_loader, device),
        "test": accuracy(model, test_loader, device),
    }
    for split, value in results.items():
        print(f"Accuracy, {split}: {value * 100:.2f}%")
    return results


def plot_history(history: History, name: str) -> None:
    epochs = range(1, len(history.training_loss) + 1)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(epochs, history.training_loss, "r", label="Training loss")
    plt.plot(epochs, history.validation_loss, "b", label="Validation loss")
    top = max(history.training_loss + history.validation_loss)
    plt.plot((history.best_epoch + 1,) * 2, (0, top), "k-", label="Best model")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    save_figure(fig, name)
