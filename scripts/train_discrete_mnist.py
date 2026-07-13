"""Train the weight-shared discrete (Euler) ResNet baseline on MNIST.

This is the parameter-matched discrete counterpart to the MLP Neural-ODE (both
204,650 params at hidden_dim=160). Reproducibility fixes vs the coursework:
- explicit ``--seed`` (python/numpy/torch/cuda + dataloader shuffle);
- logging via the pluggable backend (CSV default, no W&B account);
- robust ``memory_mb`` access so the script no longer crashes on CPU;
- per-run summary appended to ``results/mnist/discrete_summary.csv``.

Also supports the memory-scaling diagnostic (larger ``--num_layers``).
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from data.dataloaders import get_mnist_dataloaders
from models.networks import DiscreteResNet
from training.engine import eval_epoch, train_epoch
from training.logging_backend import get_logger
from training.utils import set_seed


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discrete ResNet MNIST baseline.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results/mnist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} | L={args.num_layers} | seed {args.seed}")

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size, seed=args.seed
    )

    model = DiscreteResNet(
        data_dim=784,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        num_layers=args.num_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    run_name = f"mnist_discrete_L{args.num_layers}_seed{args.seed}"
    logger = get_logger(
        run_name=run_name,
        project="neural-odes-30562",
        config={
            "model": "DiscreteResNet",
            "dataset": "MNIST",
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "epochs": args.epochs,
            "seed": args.seed,
            "num_parameters": n_params,
        },
    )

    best_test_acc = 0.0
    train_metrics: Dict[str, float] = {}
    test_metrics: Dict[str, float] = {}
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = eval_epoch(model, test_loader, criterion, device)
        best_test_acc = max(best_test_acc, test_metrics["accuracy"])

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "best_test_accuracy": best_test_acc,
                "num_parameters": n_params,
                "memory_mb": train_metrics.get("memory_mb", 0.0),
            }
        )
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch} | train_acc: {train_metrics['accuracy']:.4f} | "
                f"test_acc: {test_metrics['accuracy']:.4f} | "
                f"best: {best_test_acc:.4f} | "
                f"mem: {train_metrics.get('memory_mb', 0.0):.1f} MB"
            )

    _append_summary_row(
        Path(args.results_dir) / "discrete_summary.csv",
        {
            "num_layers": args.num_layers,
            "seed": args.seed,
            "num_parameters": n_params,
            "epochs": args.epochs,
            "final_test_accuracy": test_metrics.get("accuracy", float("nan")),
            "final_test_loss": test_metrics.get("loss", float("nan")),
            "best_test_accuracy": best_test_acc,
            "final_peak_memory_mb": train_metrics.get("memory_mb", 0.0),
        },
    )

    logger.finish()


if __name__ == "__main__":
    main()
