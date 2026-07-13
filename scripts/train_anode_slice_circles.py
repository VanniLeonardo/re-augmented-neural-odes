import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console

from data.synthetic import make_circles
from models.networks import ODENet
from training.engine import eval_epoch, train_epoch
from training.logging_backend import get_logger
from training.utils import set_seed

console = Console()


@dataclass
class SliceCirclesConfig:
    """Configuration for missing angular slice generalization on circles."""

    n_samples: int = 1000
    n_val_samples: int = 1000
    noise: float = 0.05
    missing_angle_start: float = 0.0
    missing_angle_width: float = math.pi / 5.0

    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3

    hidden_dim: int = 2
    ode_hidden_dim: int = 64
    augment_dims: Tuple[int, ...] = (0, 2)
    seeds: Tuple[int, ...] = (0, 1, 2)

    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 200
    log_every: int = 20

    project: str = "neural-odes-30562"
    results_dir: str = "results/anode"
    write_results: bool = True


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    """Parses comma-separated integers from a CLI argument."""
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _count_parameters(model: nn.Module) -> int:
    """Counts trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _angle_mask(
    x: torch.Tensor,
    angle_start: float,
    angle_width: float,
) -> torch.Tensor:
    """Returns mask for points whose polar angle lies in the removed sector."""
    if angle_width <= 0.0:
        raise ValueError("angle_width must be positive.")

    two_pi = 2.0 * math.pi
    theta = torch.atan2(x[:, 1], x[:, 0])
    theta = torch.remainder(theta, two_pi)

    angle_start = angle_start % two_pi
    if angle_width >= two_pi:
        return torch.ones_like(theta, dtype=torch.bool)

    angle_end = angle_start + angle_width
    if angle_end <= two_pi:
        return (theta >= angle_start) & (theta <= angle_end)

    wrapped_end = angle_end - two_pi
    return (theta >= angle_start) | (theta <= wrapped_end)


def _make_slice_loaders(
    cfg: SliceCirclesConfig,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train, full-validation, and removed-slice validation loaders.

    The training set excludes a fixed angular sector. The two validation loaders
    are generated from an independent sample: one keeps the full validation set,
    and the other keeps only points in the removed sector.
    """
    x_train_pool, y_train_pool = make_circles(
        n_samples=cfg.n_samples,
        noise=cfg.noise,
        seed=seed,
    )
    train_slice_mask = _angle_mask(
        x_train_pool,
        cfg.missing_angle_start,
        cfg.missing_angle_width,
    )
    x_train = x_train_pool[~train_slice_mask]
    y_train = y_train_pool[~train_slice_mask]

    x_val, y_val = make_circles(
        n_samples=cfg.n_val_samples,
        noise=cfg.noise,
        seed=seed + 10_000,
    )
    val_slice_mask = _angle_mask(
        x_val,
        cfg.missing_angle_start,
        cfg.missing_angle_width,
    )
    x_slice = x_val[val_slice_mask]
    y_slice = y_val[val_slice_mask]

    if len(x_train) == 0:
        raise ValueError("The angular mask removed the entire training set.")
    if len(x_slice) == 0:
        raise ValueError("The angular mask produced an empty slice validation set.")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    full_val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    slice_val_loader = DataLoader(
        TensorDataset(x_slice, y_slice),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    return train_loader, full_val_loader, slice_val_loader


def _get_gpu_memory_stats(device: torch.device) -> Dict[str, float]:
    """Returns peak CUDA memory in MB, or an empty dict on CPU."""
    if device.type != "cuda":
        return {}

    gpu_id = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_mem_peak_allocated_mb": torch.cuda.max_memory_allocated(gpu_id)
        / (1024**2),
    }


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    """Appends one run summary to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_args() -> SliceCirclesConfig:
    """Parses CLI overrides."""
    cfg = SliceCirclesConfig()
    parser = argparse.ArgumentParser(
        description="NODE vs ANODE generalization on missing angular slices."
    )

    parser.add_argument("--n_samples", type=int, default=cfg.n_samples)
    parser.add_argument("--n_val_samples", type=int, default=cfg.n_val_samples)
    parser.add_argument("--noise", type=float, default=cfg.noise)
    parser.add_argument(
        "--missing_angle_start",
        type=float,
        default=cfg.missing_angle_start,
    )
    parser.add_argument(
        "--missing_angle_width",
        type=float,
        default=cfg.missing_angle_width,
    )
    parser.add_argument("--solver_type", type=str, default=cfg.solver_type)
    parser.add_argument("--atol", type=float, default=cfg.atol)
    parser.add_argument("--rtol", type=float, default=cfg.rtol)
    parser.add_argument("--hidden_dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--ode_hidden_dim", type=int, default=cfg.ode_hidden_dim)
    parser.add_argument(
        "--augment_dims",
        type=_parse_int_tuple,
        default=cfg.augment_dims,
    )
    parser.add_argument("--seeds", type=_parse_int_tuple, default=cfg.seeds)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--log_every", type=int, default=cfg.log_every)
    parser.add_argument("--project", type=str, default=cfg.project)
    parser.add_argument("--results_dir", type=str, default=cfg.results_dir)
    parser.add_argument(
        "--write_results",
        action=argparse.BooleanOptionalAction,
        default=cfg.write_results,
    )

    args = parser.parse_args()
    for field in vars(cfg):
        if hasattr(args, field):
            setattr(cfg, field, getattr(args, field))
    return cfg


def _run_single(
    cfg: SliceCirclesConfig,
    augment_dim: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs one missing-slice NODE/ANODE experiment."""
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_name = "NODE" if augment_dim == 0 else f"ANODE-p{augment_dim}"
    run_name = f"slice_{model_name}_seed={seed}"

    train_loader, full_val_loader, slice_val_loader = _make_slice_loaders(cfg, seed)

    model = ODENet(
        data_dim=2,
        hidden_dim=cfg.hidden_dim,
        num_classes=2,
        solver_type=cfg.solver_type,
        atol=cfg.atol,
        rtol=cfg.rtol,
        augment_dim=augment_dim,
        ode_hidden_dim=cfg.ode_hidden_dim,
    ).to(device)

    num_parameters = _count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    logger = get_logger(
        run_name=run_name,
        project=cfg.project,
        config={
            **asdict(cfg),
            "augment_dim": augment_dim,
            "ode_hidden_dim": cfg.ode_hidden_dim,
            "seed": seed,
            "model_name": model_name,
            "num_parameters": num_parameters,
            "train_samples_after_slice": len(train_loader.dataset),
            "full_val_samples": len(full_val_loader.dataset),
            "slice_val_samples": len(slice_val_loader.dataset),
        },
    )

    final_train_metrics: Dict[str, float] = {}
    final_full_val_metrics: Dict[str, float] = {}
    final_slice_val_metrics: Dict[str, float] = {}
    final_epoch_time = 0.0

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        final_epoch_time = time.perf_counter() - epoch_start
        full_val_metrics = eval_epoch(model, full_val_loader, criterion, device)
        slice_val_metrics = eval_epoch(model, slice_val_loader, criterion, device)

        final_train_metrics = train_metrics
        final_full_val_metrics = full_val_metrics
        final_slice_val_metrics = slice_val_metrics

        logger.log(
            {
                "epoch": epoch,
                "epoch_time_s": final_epoch_time,
                "num_parameters": num_parameters,
                "train_samples_after_slice": len(train_loader.dataset),
                "slice_val_samples": len(slice_val_loader.dataset),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"full_val_{key}": value for key, value in full_val_metrics.items()},
                **{
                    f"slice_val_{key}": value
                    for key, value in slice_val_metrics.items()
                },
                **_get_gpu_memory_stats(device),
            }
        )

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            console.log(
                f"[bold]{run_name}[/bold] | Epoch {epoch:>3} | "
                f"train acc: {train_metrics['accuracy']:.3f} | "
                f"full val acc: {full_val_metrics['accuracy']:.3f} | "
                f"slice val acc: {slice_val_metrics['accuracy']:.3f} | "
                f"Fwd NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
                f"Bwd NFE: {train_metrics.get('backward_nfe_mean', 0):.1f} | "
                f"time: {final_epoch_time:.2f}s"
            )

    logger.finish()

    return {
        "model_name": model_name,
        "augment_dim": augment_dim,
        "seed": seed,
        "n_samples": cfg.n_samples,
        "n_val_samples": cfg.n_val_samples,
        "train_samples_after_slice": len(train_loader.dataset),
        "full_val_samples": len(full_val_loader.dataset),
        "slice_val_samples": len(slice_val_loader.dataset),
        "missing_angle_start": cfg.missing_angle_start,
        "missing_angle_width": cfg.missing_angle_width,
        "hidden_dim": cfg.hidden_dim,
        "solver_type": cfg.solver_type,
        "atol": cfg.atol,
        "rtol": cfg.rtol,
        "epochs": cfg.epochs,
        "num_parameters": num_parameters,
        "final_epoch_time_s": final_epoch_time,
        "final_train_loss": final_train_metrics["loss"],
        "final_train_accuracy": final_train_metrics["accuracy"],
        "final_full_val_loss": final_full_val_metrics["loss"],
        "final_full_val_accuracy": final_full_val_metrics["accuracy"],
        "final_slice_val_loss": final_slice_val_metrics["loss"],
        "final_slice_val_accuracy": final_slice_val_metrics["accuracy"],
        "final_train_forward_nfe": final_train_metrics.get("forward_nfe_mean", 0.0),
        "final_train_backward_nfe": final_train_metrics.get("backward_nfe_mean", 0.0),
        "final_train_total_nfe": final_train_metrics.get("total_nfe_mean", 0.0),
        "final_full_val_forward_nfe": final_full_val_metrics.get(
            "forward_nfe_mean", 0.0
        ),
        "final_slice_val_forward_nfe": final_slice_val_metrics.get(
            "forward_nfe_mean", 0.0
        ),
    }


def main() -> None:
    """Entry point for the missing-slice ANODE experiment."""
    cfg = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_path = Path(cfg.results_dir) / "slice_circles_summary.csv"

    console.log(f"Running on device: [bold]{device}[/bold]")
    console.log(
        f"Starting missing-slice sweep: "
        f"{len(cfg.augment_dims)} augment dims x {len(cfg.seeds)} seeds"
    )

    for augment_dim in cfg.augment_dims:
        for seed in cfg.seeds:
            summary = _run_single(cfg, augment_dim, seed, device)
            if cfg.write_results:
                _append_summary_row(summary_path, summary)

    console.log("[bold green]Missing-slice ANODE sweep complete.[/bold green]")
    if cfg.write_results:
        console.log(f"Summary written to [bold]{summary_path}[/bold]")


if __name__ == "__main__":
    main()
