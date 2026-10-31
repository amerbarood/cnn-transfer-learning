"""Command-line entry point.

Examples:
    python main.py cifar-cnn --activation leakyrelu
    python main.py cifar-cnn --activation tanh --epochs 20
    python main.py transfer-imagenet
    python main.py transfer-mnist-svhn
    python main.py all
"""

from __future__ import annotations

import argparse

from cnn_transfer import experiments
from cnn_transfer.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CNNs and transfer learning in PyTorch")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible splits")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cifar-cnn", help="0.1: small CNN on CIFAR-10, SGD vs Adam")
    p.add_argument("--activation", choices=["leakyrelu", "tanh"], default="leakyrelu")
    p.add_argument("--optimizers", nargs="+", choices=["sgd", "adam"], default=["sgd", "adam"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=None, help="default: 750 (LeakyReLU) / 5000 (Tanh)")

    p = sub.add_parser("transfer-imagenet", help="0.2.1: AlexNet fine-tuning vs feature extraction")
    p.add_argument("--modes", nargs="+", choices=["finetune", "feature_extraction"],
                   default=["finetune", "feature_extraction"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=750)
    p.add_argument("--eval-batch-size", type=int, default=None, help="default: whole split in one batch")

    p = sub.add_parser("transfer-mnist-svhn", help="0.2.2: MNIST model applied to SVHN")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=5000)

    sub.add_parser("all", help="run every experiment with default settings")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.command == "cifar-cnn":
        experiments.cifar_cnn(args.activation, tuple(args.optimizers), args.epochs, args.lr, args.batch_size)
    elif args.command == "transfer-imagenet":
        experiments.transfer_imagenet(tuple(args.modes), args.epochs, args.lr, args.batch_size, args.eval_batch_size)
    elif args.command == "transfer-mnist-svhn":
        experiments.transfer_mnist_svhn(args.epochs, args.lr, args.batch_size)
    elif args.command == "all":
        experiments.cifar_cnn("leakyrelu")
        experiments.cifar_cnn("tanh")
        experiments.transfer_imagenet()
        experiments.transfer_mnist_svhn()


if __name__ == "__main__":
    main()
