import argparse
import csv
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from rich.console import Console

from config import ANODECirclesConfig
from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import eval_epoch, train_epoch
from training.logging_backend import get_logger
from training.utils import set_seed

console = Console()


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    """Parses comma-separated integers from a CLI argument."""
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _count_parameters(model: nn.Module) -> int:
    """Counts trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


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


def _add_config_arguments(
    parser: argparse.ArgumentParser, cfg: ANODECirclesConfig
) -> None:
    """Adds scalar dataclass fields to the CLI parser."""
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, bool):
            parser.add_argument(
                f"--{field.name}",
                action=argparse.BooleanOptionalAction,
                default=value,
            )
        elif isinstance(value, (str, int, float)):
            parser.add_argument(f"--{field.name}", type=type(value), default=value)

    parser.add_argument(
        "--augment_dims",
        type=_parse_int_tuple,
        default=cfg.augment_dims,
        help="Comma-separated augmentation dimensions, e.g. 0,1,2,5.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default=cfg.seeds,
        help="Comma-separated random seeds, e.g. 0,1,2.",
    )


def _parse_args() -> ANODECirclesConfig:
    """Parses CLI overrides into an ANODECirclesConfig."""
    cfg = ANODECirclesConfig()
    parser = argparse.ArgumentParser(
        description="NODE vs ANODE sweep on concentric circles."
    )
    _add_config_arguments(parser, cfg)
    args = parser.parse_args()

    for field in fields(cfg):
        if hasattr(args, field.name):
            setattr(cfg, field.name, getattr(args, field.name))

    return cfg


def _run_single(
    cfg: ANODECirclesConfig,
    augment_dim: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs one NODE/ANODE training configuration.

    Args:
        cfg (ANODECirclesConfig): Shared experiment configuration.
        augment_dim (int): Number of zero dimensions appended to the ODE state.
        seed (int): Random seed for dataset split and model initialization.
        device (torch.device): Target compute device.

    Returns:
        Dict[str, Any]: Final run summary.
    """
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_name = "NODE" if augment_dim == 0 else f"ANODE-p{augment_dim}"
    run_name = f"{model_name}_seed={seed}"

    train_loader, val_loader = get_dataloaders(
        dataset=cfg.dataset,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        noise=cfg.noise,
        seed=seed,
    )

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
        },
    )

    final_train_metrics: Dict[str, float] = {}
    final_val_metrics: Dict[str, float] = {}
    final_epoch_time = 0.0

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        final_epoch_time = time.perf_counter() - epoch_start
        val_metrics = eval_epoch(model, val_loader, criterion, device)

        final_train_metrics = train_metrics
        final_val_metrics = val_metrics

        logger.log(
            {
                "epoch": epoch,
                "epoch_time_s": final_epoch_time,
                "num_parameters": num_parameters,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **_get_gpu_memory_stats(device),
            }
        )

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            console.log(
                f"[bold]{run_name}[/bold] | Epoch {epoch:>3} | "
                f"train loss: {train_metrics['loss']:.4f} | "
                f"train acc: {train_metrics['accuracy']:.3f} | "
                f"val acc: {val_metrics['accuracy']:.3f} | "
                f"Fwd NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
                f"Bwd NFE: {train_metrics.get('backward_nfe_mean', 0):.1f} | "
                f"time: {final_epoch_time:.2f}s"
            )

    logger.finish()

    summary = {
        "model_name": model_name,
        "augment_dim": augment_dim,
        "seed": seed,
        "dataset": cfg.dataset,
        "n_samples": cfg.n_samples,
        "hidden_dim": cfg.hidden_dim,
        "solver_type": cfg.solver_type,
        "atol": cfg.atol,
        "rtol": cfg.rtol,
        "epochs": cfg.epochs,
        "num_parameters": num_parameters,
        "final_epoch_time_s": final_epoch_time,
        "final_train_loss": final_train_metrics["loss"],
        "final_train_accuracy": final_train_metrics["accuracy"],
        "final_val_loss": final_val_metrics["loss"],
        "final_val_accuracy": final_val_metrics["accuracy"],
        "final_train_forward_nfe": final_train_metrics.get("forward_nfe_mean", 0.0),
        "final_train_backward_nfe": final_train_metrics.get("backward_nfe_mean", 0.0),
        "final_train_total_nfe": final_train_metrics.get("total_nfe_mean", 0.0),
        "final_val_forward_nfe": final_val_metrics.get("forward_nfe_mean", 0.0),
    }

    return summary


def main() -> None:
    """Entry point for the ANODE concentric-circles sweep."""
    cfg = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.log(f"Running on device: [bold]{device}[/bold]")
    console.log(
        f"Starting ANODE circles sweep: "
        f"{len(cfg.augment_dims)} augment dims x {len(cfg.seeds)} seeds"
    )

    summary_path = Path(cfg.results_dir) / "circles_summary.csv"

    for augment_dim in cfg.augment_dims:
        for seed in cfg.seeds:
            summary = _run_single(cfg, augment_dim, seed, device)
            if cfg.write_results:
                _append_summary_row(summary_path, summary)

    console.log("[bold green]ANODE circles sweep complete.[/bold green]")
    if cfg.write_results:
        console.log(f"Summary written to [bold]{summary_path}[/bold]")


if __name__ == "__main__":
    main()
