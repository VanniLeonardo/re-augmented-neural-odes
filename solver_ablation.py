import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from dataclasses import fields
from rich.console import Console

from config import ODEConfig, SolverAblationConfig
from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import eval_epoch, train_epoch
from training.logging_backend import get_logger
from training.utils import set_seed

console = Console()


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _build_run_configs(
    cfg: SolverAblationConfig,
) -> List[Tuple[str, float, float]]:
    r"""Generates the full sweep matrix as (solver, atol, rtol) tuples.

    Fixed-step solvers are not tolerance-controlled; (0.0, 0.0) is used as a
    sentinel. Adaptive solvers are crossed with the tolerance grid using
    $\text{atol} = \text{rtol}$, which is standard practice.

    Args:
        cfg (SolverAblationConfig): The ablation configuration.
    """
    runs: List[Tuple[str, float, float]] = [(s, 0.0, 0.0) for s in cfg.fixed_solvers]
    for s in cfg.adaptive_solvers:
        for tol in cfg.tolerances:
            runs.append((s, tol, tol))
    if getattr(cfg, "max_configs", 0) and cfg.max_configs > 0:
        runs = runs[: cfg.max_configs]
    return runs


def _run_single(
    solver: str,
    atol: float,
    rtol: float,
    cfg: SolverAblationConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Trains and evaluates one (solver, tolerance) configuration.

    Args:
        solver (str): torchdiffeq solver name (e.g. 'dopri5', 'rk4').
        atol (float): Absolute tolerance; 0.0 for fixed-step solvers (ignored).
        rtol (float): Relative tolerance; 0.0 for fixed-step solvers (ignored).
        cfg (SolverAblationConfig): Shared training configuration.
        train_loader (torch.utils.data.DataLoader): Pre-built training loader.
        val_loader (torch.utils.data.DataLoader): Pre-built validation loader.
        device (torch.device): Target compute device.
    """
    is_adaptive = solver not in set(cfg.fixed_solvers)
    run_name = (
        f"solver={solver}_tol={atol:.0e}_seed={cfg.seed}"
        if is_adaptive
        else f"solver={solver}_seed={cfg.seed}"
    )

    logger = get_logger(
        run_name=run_name,
        project="neural-odes-30562",
        config={
            "solver": solver,
            "atol": atol,
            "rtol": rtol,
            "is_adaptive": is_adaptive,
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "hidden_dim": cfg.hidden_dim,
            "seed": cfg.seed,
        },
    )

    # Fixed-step solvers ignore atol/rtol; pass ODEConfig defaults as safe values
    effective_atol = atol if is_adaptive else ODEConfig.atol
    effective_rtol = rtol if is_adaptive else ODEConfig.rtol

    model = ODENet(
        data_dim=2,
        hidden_dim=cfg.hidden_dim,
        num_classes=2,
        solver_type=solver,
        atol=effective_atol,
        rtol=effective_rtol,
        use_stem=False,  # toy circles: integrate in data space (d=2), vf width = hidden_dim.
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.perf_counter() - t0

        val_metrics = eval_epoch(model, val_loader, criterion, device)

        logger.log(
            {
                "epoch": epoch,
                "epoch_time_s": epoch_time,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if epoch % 5 == 0:
            console.log(
                f"[bold]{run_name}[/bold] | Epoch {epoch:>3} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f} | "
                f"Fwd NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
                f"Time: {epoch_time:.2f}s"
            )

    logger.finish()

    _append_summary_row(
        Path(cfg.results_dir) / "solver_ablation_summary.csv",
        {
            "solver": solver,
            "is_adaptive": is_adaptive,
            "atol": effective_atol,
            "rtol": effective_rtol,
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "final_val_accuracy": val_metrics.get("accuracy", float("nan")),
            "final_val_loss": val_metrics.get("loss", float("nan")),
            "final_train_forward_nfe": train_metrics.get("forward_nfe_mean", 0.0),
            "final_train_backward_nfe": train_metrics.get("backward_nfe_mean", 0.0),
        },
    )


def main() -> None:
    """Entry point for the solver ablation sweep."""

    config = SolverAblationConfig()
    parser = argparse.ArgumentParser(description="Neural ODE solver ablation")
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, (str, int, float, bool)):
            parser.add_argument(f"--{f.name}", type=type(val), default=val)
    args = parser.parse_args()
    for f in fields(config):
        if hasattr(args, f.name):
            setattr(config, f.name, getattr(args, f.name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Running on device: [bold]{device}[/bold]")

    set_seed(config.seed)
    run_configs = _build_run_configs(config)
    train_loader, val_loader = get_dataloaders(
        dataset=config.dataset,
        n_samples=config.n_samples,
        batch_size=config.batch_size,
        val_split=config.val_split,
        noise=config.noise,
        seed=config.seed,
    )
    console.log(f"Starting ablation: [bold]{len(run_configs)} runs[/bold]")

    for solver, atol, rtol in run_configs:
        _run_single(solver, atol, rtol, config, train_loader, val_loader, device)

    console.log("[bold green]Ablation complete.[/bold green]")


if __name__ == "__main__":
    main()
