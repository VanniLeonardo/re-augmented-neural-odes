"""Train an MLP or convolutional Neural-ODE classifier on MNIST.

NOTE (ReScience scope): these MNIST rows are our OWN seeded baselines. They are
NOT a reproduction of Chen et al. 2018 Table 1 (different architecture, params,
epochs, solver, and error rate). See REPLICATION_PLAN.md (C5) and §9.

Reproducibility fixes vs the coursework version:
- explicit ``--seed`` seeds python/numpy/torch/cuda and the dataloader shuffle;
- logging goes through the pluggable backend (CSV default, no W&B account needed);
- the expensive tolerance diagnostic is opt-in (``--tol-diagnostic``);
- a per-run summary row is appended to ``results/mnist/mnist_summary.csv`` so the
  table is regenerable from committed CSVs.
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
from models.networks import ODENet, ConvODENet
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
    parser = argparse.ArgumentParser(description="Neural-ODE MNIST classifier.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument(
        "--network_type", type=str, default="cnn", choices=["mlp", "cnn"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results/mnist")
    parser.add_argument(
        "--tol-diagnostic",
        action="store_true",
        help="Run the (expensive) post-training solver-tolerance diagnostic (Chen Fig 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} | Network: {args.network_type.upper()} | seed {args.seed}")

    flatten_img = args.network_type == "mlp"
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size, flatten=flatten_img, seed=args.seed
    )

    if args.network_type == "mlp":
        model = ODENet(
            data_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=10,
            solver_type=args.solver,
        ).to(device)
    else:
        model = ConvODENet(
            in_channels=1,
            num_filters=args.hidden_dim,
            num_classes=10,
            solver_type=args.solver,
        ).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    run_name = f"mnist_{args.network_type}_seed{args.seed}"
    logger = get_logger(
        run_name=run_name,
        project="neural-odes-30562",
        config={
            "model": "ODENet",
            "network_type": args.network_type,
            "dataset": "MNIST",
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "epochs": args.epochs,
            "solver": args.solver,
            "seed": args.seed,
            "num_parameters": n_params,
        },
    )

    train_metrics: Dict[str, float] = {}
    test_metrics: Dict[str, float] = {}
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = eval_epoch(model, test_loader, criterion, device)

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "num_parameters": n_params,
                "forward_nfe": train_metrics.get("forward_nfe_mean", 0.0),
                "peak_memory_mb": train_metrics.get("memory_mb", 0.0),
            }
        )
        print(
            f"Epoch {epoch} | train_acc: {train_metrics['accuracy']:.4f} | "
            f"test_acc: {test_metrics['accuracy']:.4f} | "
            f"NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
            f"Mem: {train_metrics.get('memory_mb', 0.0):.1f} MB"
        )

    _append_summary_row(
        Path(args.results_dir) / "mnist_summary.csv",
        {
            "network_type": args.network_type,
            "seed": args.seed,
            "num_parameters": n_params,
            "epochs": args.epochs,
            "solver": args.solver,
            "final_test_accuracy": test_metrics.get("accuracy", float("nan")),
            "final_test_loss": test_metrics.get("loss", float("nan")),
            "final_train_forward_nfe": train_metrics.get("forward_nfe_mean", 0.0),
            "final_train_backward_nfe": train_metrics.get("backward_nfe_mean", 0.0),
            "final_peak_memory_mb": train_metrics.get("memory_mb", 0.0),
        },
    )

    if args.tol_diagnostic:
        # Imported lazily so the default path has no dependency on the diagnostic.
        from scripts.plot_fig3 import evaluate_tolerances, plot_figure_3

        print("Running post-training solver-tolerance diagnostic (Chen Fig 3)...")
        x_val, y_val = next(iter(test_loader))
        x_val, y_val = x_val.to(device), y_val.to(device)
        results = evaluate_tolerances(model, x_val, y_val)
        plot_figure_3(results, epoch=args.epochs)
        print("Tolerance diagnostic figure written to plots/.")

    logger.finish()


if __name__ == "__main__":
    main()
