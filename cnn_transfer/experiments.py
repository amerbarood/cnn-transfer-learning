"""The three experiments from the original notebooks.

Hyperparameters default to the values used in the notebooks.
"""

from __future__ import annotations

from torch import nn, optim

from . import data, models
from .training import plot_history, report_accuracy, train_model
from .utils import get_device, save_image_samples

OPTIMIZERS = {"sgd": optim.SGD, "adam": optim.Adam}


def cifar_cnn(
    activation: str = "leakyrelu",
    optimizers: tuple[str, ...] = ("sgd", "adam"),
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int | None = None,
) -> dict:
    """0.1: small CNN on CIFAR-10, comparing SGD and Adam for one activation.

    Notebook batch sizes: 750 for LeakyReLU, 5000 for Tanh.
    """
    device = get_device()
    if batch_size is None:
        batch_size = 750 if activation == "leakyrelu" else 5000
    loaders = data.cifar10(batch_size)
    save_image_samples(data.first_images(loaders.train), f"cifar10_samples_{activation}")

    results = {}
    for name in optimizers:
        print(f"\n== CIFAR-10 CNN, {activation} + {name} ==")
        model = models.cifar_cnn(activation).to(device)
        optimizer = OPTIMIZERS[name](model.parameters(), lr, weight_decay=0)
        best, history = train_model(
            model, nn.CrossEntropyLoss(), optimizer, loaders.train, loaders.validation, epochs, device
        )
        plot_history(history, f"cifar10_cnn_{activation}_{name}_loss")
        results[name] = report_accuracy(best, loaders.train, loaders.validation, loaders.test, device)
    return results


def transfer_imagenet(
    modes: tuple[str, ...] = ("finetune", "feature_extraction"),
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 750,
    eval_batch_size: int | None = None,
) -> dict:
    """0.2.1: ImageNet AlexNet on CIFAR-10, fine-tuning vs feature extraction.

    Note: the notebook intended a smaller batch on GPU, but its check never
    matched, so 750 (with full-size evaluation batches) was what actually ran.
    Pass a smaller --batch-size / --eval-batch-size if you run out of memory.
    """
    device = get_device()
    loaders = data.cifar10(batch_size, eval_batch_size, imagenet_style=True)

    results = {}
    for mode in modes:
        print(f"\n== AlexNet on CIFAR-10, {mode} ==")
        model = models.pretrained_alexnet(loaders.num_classes, freeze_backbone=(mode == "feature_extraction"))
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr, weight_decay=0)
        best, history = train_model(
            model, nn.CrossEntropyLoss(), optimizer, loaders.train, loaders.validation, epochs, device,
            print_every=1,
        )
        plot_history(history, f"alexnet_{mode}_loss")
        results[mode] = report_accuracy(best, loaders.train, loaders.validation, loaders.test, device)
    return results


def transfer_mnist_svhn(epochs: int = 100, lr: float = 1e-3, batch_size: int = 5000) -> dict:
    """0.2.2: train a CNN on MNIST, then evaluate the same model on SVHN."""
    device = get_device()
    mnist = data.mnist(batch_size)
    save_image_samples(data.first_images(mnist.train), "mnist_samples", gray=True)

    print("\n== CNN trained on MNIST ==")
    model = models.mnist_cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=0)
    best, history = train_model(model, nn.CrossEntropyLoss(), optimizer, mnist.train, mnist.validation, epochs, device)
    plot_history(history, "mnist_cnn_loss")
    results = {"mnist": report_accuracy(best, mnist.train, mnist.validation, mnist.test, device)}

    print("\n== Same model applied to SVHN ==")
    svhn = data.svhn_as_mnist(batch_size)
    save_image_samples(data.first_images(svhn.train), "svhn_samples", gray=True)
    results["svhn"] = report_accuracy(best, svhn.train, svhn.validation, svhn.test, device)
    return results
