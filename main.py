from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, random_split

import wandb

from config import ODEConfig, LatentODEConfig
from data.dataloaders import get_dataloaders
from data.synthetic import TimeSeriesDataset
from models.continuous import LatentODE
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import plot_ode_flows, visualize_2d_features

console = Console()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def arg_type_from_default(value: Any):
    if isinstance(value, bool):
        return str_to_bool
    if isinstance(value, int) and not isinstance(value, bool):
        return int
    if isinstance(value, float):
        return float
    return str


def add_prefixed_config_args(
    parser: argparse.ArgumentParser,
    *,
    config: Any,
    prefix: str,
    title: str,
) -> None:
    group = parser.add_argument_group(title)
    for field, value in config.__dict__.items():
        option = f"--{prefix}-{field.replace('_', '-')}"
        destination = f"{prefix.replace('-', '_')}_{field}"
        group.add_argument(
            option, dest=destination, type=arg_type_from_default(value), default=None
        )


def apply_prefixed_config_args(
    config: Any, args: argparse.Namespace, prefix: str
) -> None:
    destination_prefix = prefix.replace("-", "_")
    for field in config.__dict__:
        value = getattr(args, f"{destination_prefix}_{field}", None)
        if value is not None:
            setattr(config, field, value)


def to_torch_device(device_like: Any) -> torch.device:
    if isinstance(device_like, torch.device):
        return device_like
    return torch.device(device_like)


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup_after_phase(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def get_gpu_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}

    gpu_id = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated(gpu_id) / (1024**2),
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved(gpu_id) / (1024**2),
        "gpu_mem_peak_allocated_mb": torch.cuda.max_memory_allocated(gpu_id)
        / (1024**2),
    }


def build_neural_ode_config(args: argparse.Namespace) -> ODEConfig:
    config = ODEConfig()

    config.hidden_dim = 2
    config.epochs = 200

    apply_prefixed_config_args(config, args, "ode")
    return config


def run_neural_ode(
    config: ODEConfig, *, run_name: Optional[str] = None
) -> Dict[str, float]:
    console.rule("Phase 1: Neural ODE")

    wandb.init(
        project="neural-odes-30562",
        name=run_name,
        config=config.__dict__,
        reinit=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Running Neural ODE on device: [bold]{device}[/bold]")

    train_loader, _val_loader = get_dataloaders(
        dataset=config.dataset,
        n_samples=config.n_samples,
        batch_size=config.batch_size,
        val_split=config.val_split,
        noise=config.noise,
    )

    model = ODENet(
        data_dim=config.in_features,
        hidden_dim=config.hidden_dim,
        num_classes=2,
        solver_type=config.solver_type,
        augment_dim=config.augment_dim,
        atol=config.atol,
        rtol=config.rtol,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    last_train_metrics: Dict[str, float] = {}

    try:
        for epoch in range(config.epochs):
            if device.type == "cuda":
                gpu_id = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                torch.cuda.reset_peak_memory_stats(gpu_id)

            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            last_train_metrics = dict(train_metrics)
            gpu_mem = get_gpu_memory_stats(device)

            wandb.log(
                {"phase": "neural_ode", "epoch": epoch, **train_metrics, **gpu_mem}
            )

            if config.hidden_dim == 2 and epoch % 10 == 0:
                visualize_2d_features(model, train_loader, device, epoch)

            if epoch % 20 == 0:
                plot_ode_flows(
                    model, train_loader, device, epoch, is_anode=config.augment_dim > 0
                )

            if epoch % 5 == 0:
                msg = (
                    f"Epoch {epoch:>3} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Acc: {train_metrics['accuracy']:.3f} | "
                    f"Fwd NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
                    f"Bwd NFE: {train_metrics.get('backward_nfe_mean', 0):.1f}"
                )

                if gpu_mem:
                    msg += (
                        f" | GPU Alloc: {gpu_mem['gpu_mem_allocated_mb']:.1f} MB"
                        f" | GPU Reserved: {gpu_mem['gpu_mem_reserved_mb']:.1f} MB"
                        f" | GPU Peak: {gpu_mem['gpu_mem_peak_allocated_mb']:.1f} MB"
                    )

                console.log(msg)

        return last_train_metrics
    finally:
        wandb.finish()
        cleanup_after_phase(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def move_batch_to_device(
    batch: Dict[str, Tensor], device: torch.device
) -> Dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_latent_ode(
    model: LatentODE, loader: DataLoader, config: LatentODEConfig
) -> Dict[str, float]:
    model.eval()

    total_interp_num = 0.0
    total_interp_den = 0.0
    total_extrap_num = 0.0
    total_extrap_den = 0.0
    total_recon_num = 0.0
    total_recon_den = 0.0
    total_kl = 0.0
    total_nfe = 0.0
    total_nfe_encoder = 0.0
    total_nfe_latent = 0.0
    num_batches = 0

    context_len = config.seq_len // 2
    device = to_torch_device(config.device)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            x=batch["observed_context"],
            t_obs=batch["context_times"],
            mask=batch["context_mask"],
            t_query=batch["full_times"],
            sample_latent=False,
        )

        preds = out["predictions"]
        interp_pred = preds[:, :context_len, :]
        future_pred = preds[:, context_len:, :]

        interp_sqerr = (
            (interp_pred - batch["ground_truth"][:, :context_len, :]) ** 2
        ) * batch["interp_mask"]
        extrap_sqerr = (
            (future_pred - batch["ground_truth"][:, context_len:, :]) ** 2
        ) * batch["future_mask"]
        recon_sqerr = ((interp_pred - batch["context_values"]) ** 2) * batch[
            "context_mask"
        ]

        total_interp_num += interp_sqerr.sum().item()
        total_interp_den += batch["interp_mask"].sum().item()
        total_extrap_num += extrap_sqerr.sum().item()
        total_extrap_den += batch["future_mask"].sum().item()
        total_recon_num += recon_sqerr.sum().item()
        total_recon_den += batch["context_mask"].sum().item()
        total_kl += kl_divergence(out["mu"], out["logvar"]).item()
        total_nfe += out["nfe"].item()
        total_nfe_encoder += out["nfe_encoder"].item()
        total_nfe_latent += out["nfe_latent"].item()
        num_batches += 1

    return {
        "recon_mse": total_recon_num / max(total_recon_den, 1.0),
        "interpolation_mse": total_interp_num / max(total_interp_den, 1.0),
        "extrapolation_mse": total_extrap_num / max(total_extrap_den, 1.0),
        "kl": total_kl / max(num_batches, 1),
        "nfe": total_nfe / max(num_batches, 1),
        "nfe_encoder": total_nfe_encoder / max(num_batches, 1),
        "nfe_latent": total_nfe_latent / max(num_batches, 1),
    }


def train_latent_ode_one_epoch(
    model: LatentODE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    config: LatentODEConfig,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_nfe = 0.0
    total_nfe_encoder = 0.0
    total_nfe_latent = 0.0
    num_batches = 0
    device = to_torch_device(config.device)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        out = model(
            x=batch["observed_context"],
            t_obs=batch["context_times"],
            mask=batch["context_mask"],
            t_query=batch["context_times"],
            sample_latent=None,
        )

        recon_loss = masked_mse(
            pred=out["predictions"],
            target=batch["context_values"],
            mask=batch["context_mask"],
        )
        kl_loss = (
            kl_divergence(out["mu"], out["logvar"])
            if config.is_variational
            else torch.zeros((), device=device)
        )
        loss = recon_loss + config.kl_coeff * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_nfe += out["nfe"].item()
        total_nfe_encoder += out["nfe_encoder"].item()
        total_nfe_latent += out["nfe_latent"].item()
        num_batches += 1

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_recon": total_recon / max(num_batches, 1),
        "train_kl": total_kl / max(num_batches, 1),
        "train_nfe": total_nfe / max(num_batches, 1),
        "train_nfe_encoder": total_nfe_encoder / max(num_batches, 1),
        "train_nfe_latent": total_nfe_latent / max(num_batches, 1),
    }


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.6f}")
    console.print(table)


@torch.no_grad()
def evaluate_persistence(
    loader: DataLoader, config: LatentODEConfig
) -> Dict[str, float]:
    total_interp_num = 0.0
    total_interp_den = 0.0
    total_extrap_num = 0.0
    total_extrap_den = 0.0
    context_len = config.seq_len // 2

    for batch in loader:
        observed = batch["observed_context"]
        context_mask = batch["context_mask"]
        ground_truth = batch["ground_truth"]
        interp_mask = batch["interp_mask"]
        future_mask = batch["future_mask"]

        batch_size, context_steps, dim = observed.shape
        full_steps = ground_truth.shape[1]

        pred_ctx = torch.zeros_like(observed)
        last_value = torch.zeros(
            batch_size, dim, device=observed.device, dtype=observed.dtype
        )
        for i in range(context_steps):
            is_obs = context_mask[:, i, :].bool()
            last_value = torch.where(is_obs, observed[:, i, :], last_value)
            pred_ctx[:, i, :] = last_value

        pred_future = last_value.unsqueeze(1).expand(
            batch_size, full_steps - context_steps, dim
        )

        interp_sqerr = (
            (pred_ctx - ground_truth[:, :context_len, :]) ** 2
        ) * interp_mask
        extrap_sqerr = (
            (pred_future - ground_truth[:, context_len:, :]) ** 2
        ) * future_mask

        total_interp_num += interp_sqerr.sum().item()
        total_interp_den += interp_mask.sum().item()
        total_extrap_num += extrap_sqerr.sum().item()
        total_extrap_den += future_mask.sum().item()

    return {
        "persistence_interpolation_mse": total_interp_num / max(total_interp_den, 1.0),
        "persistence_extrapolation_mse": total_extrap_num / max(total_extrap_den, 1.0),
    }


def build_latent_ode_config(args: argparse.Namespace) -> LatentODEConfig:
    config = LatentODEConfig()
    apply_prefixed_config_args(config, args, "latent")

    if args.latent_use_ode_rnn is not None:
        config.use_ode_rnn = args.latent_use_ode_rnn
    if args.latent_is_variational is not None:
        config.is_variational = args.latent_is_variational
    if args.latent_skip_validation:
        config.skip_validation = True
    if args.latent_encoder_type is not None:
        config.encoder_type = args.latent_encoder_type
        config.use_ode_rnn = args.latent_encoder_type == "odernn"

    device = to_torch_device(config.device)
    config.device = device

    if not hasattr(config, "seed"):
        config.seed = 42
    if not hasattr(config, "eval_every"):
        config.eval_every = 1
    if not hasattr(config, "skip_validation"):
        config.skip_validation = False
    if not hasattr(config, "profile_dataloader"):
        config.profile_dataloader = True
    if not hasattr(config, "profile_num_batches"):
        config.profile_num_batches = 20
    if not hasattr(config, "num_workers"):
        config.num_workers = 0
    if not hasattr(config, "pin_memory"):
        config.pin_memory = device.type == "cuda"

    return config


def run_latent_ode(
    config: LatentODEConfig, *, run_name: Optional[str] = None
) -> Dict[str, float]:
    console.rule("Phase 2: Latent ODE")

    device = to_torch_device(config.device)
    config.device = device
    set_seed(config.seed)

    wandb.init(
        project=config.project_name,
        name=run_name,
        config={
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in config.__dict__.items()
        },
        reinit=True,
    )

    dataset = TimeSeriesDataset(config=config)
    assert (
        len(dataset) == config.num_samples
    ), f"dataset size {len(dataset)} != num_train+num_val+num_test = {config.num_samples}"

    split_generator = torch.Generator().manual_seed(config.seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [config.num_train, config.num_val, config.num_test],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model = LatentODE(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    console.log(
        f"phase 4 run | "
        f"encoder_type={getattr(config, 'encoder_type', 'odernn' if config.use_ode_rnn else 'gru_time')} | "
        f"is_variational={config.is_variational} | "
        f"device={device}"
    )
    console.log(
        f"dataset sizes | train={len(train_set)} | val={len(val_set)} | test={len(test_set)}"
    )

    for attr_name in ("num_context", "num_future"):
        if hasattr(dataset, attr_name):
            console.log(f"{attr_name}={getattr(dataset, attr_name)}")
    if hasattr(dataset, "num_context") and hasattr(dataset, "num_future"):
        console.log(f"t_full={dataset.num_context + dataset.num_future}")

    if config.profile_dataloader:
        start = time.time()
        for batch_idx, _ in enumerate(train_loader):
            if batch_idx + 1 >= config.profile_num_batches:
                break
        loader_time = time.time() - start
        console.log(
            f"dataloader probe | first {config.profile_num_batches} batches in {loader_time:.2f}s"
        )

    best_val_interp = float("inf")
    best_state_dict = None

    try:
        for epoch in range(1, config.epochs + 1):
            sync_cuda(device)
            train_start = time.time()
            train_metrics = train_latent_ode_one_epoch(
                model, train_loader, optimizer, config
            )
            sync_cuda(device)
            train_time = time.time() - train_start

            val_metrics = {
                "recon_mse": float("nan"),
                "interpolation_mse": float("nan"),
                "extrapolation_mse": float("nan"),
                "kl": float("nan"),
                "nfe": float("nan"),
                "nfe_encoder": float("nan"),
                "nfe_latent": float("nan"),
            }
            val_time = 0.0

            do_validation = (not config.skip_validation) and (
                epoch % config.eval_every == 0 or epoch == 1 or epoch == config.epochs
            )

            if do_validation:
                sync_cuda(device)
                val_start = time.time()
                val_metrics = evaluate_latent_ode(model, val_loader, config)
                sync_cuda(device)
                val_time = time.time() - val_start

                if val_metrics["interpolation_mse"] < best_val_interp:
                    best_val_interp = val_metrics["interpolation_mse"]
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

            row = {
                "phase": "latent_ode",
                "epoch": epoch,
                "train_loss": float(train_metrics["train_loss"]),
                "train_recon": float(train_metrics["train_recon"]),
                "train_kl": float(train_metrics["train_kl"]),
                "train_nfe": float(train_metrics["train_nfe"]),
                "train_nfe_encoder": float(train_metrics["train_nfe_encoder"]),
                "train_nfe_latent": float(train_metrics["train_nfe_latent"]),
                "train_time_sec": train_time,
                "val_recon": float(val_metrics["recon_mse"]),
                "val_interp": float(val_metrics["interpolation_mse"]),
                "val_extrap": float(val_metrics["extrapolation_mse"]),
                "val_kl": float(val_metrics["kl"]),
                "val_nfe": float(val_metrics["nfe"]),
                "val_nfe_encoder": float(val_metrics["nfe_encoder"]),
                "val_nfe_latent": float(val_metrics["nfe_latent"]),
                "val_time_sec": val_time,
            }

            wandb.log(row)

            if do_validation:
                console.log(
                    f"epoch {epoch:03d} | "
                    f"train_time={train_time:.2f}s | "
                    f"val_time={val_time:.2f}s | "
                    f"loss={row['train_loss']:.6f} | "
                    f"val_interp={row['val_interp']:.6f} | "
                    f"val_extrap={row['val_extrap']:.6f} | "
                    f"train_encoder_nfe={row['train_nfe_encoder']:.2f} | "
                    f"train_latent_nfe={row['train_nfe_latent']:.2f} | "
                    f"train_total_nfe={row['train_nfe']:.2f} | "
                    f"val_encoder_nfe={row['val_nfe_encoder']:.2f} | "
                    f"val_latent_nfe={row['val_nfe_latent']:.2f} | "
                    f"val_total_nfe={row['val_nfe']:.2f}"
                )
            else:
                console.log(
                    f"epoch {epoch:03d} | "
                    f"train_time={train_time:.2f}s | "
                    f"loss={row['train_loss']:.6f} | "
                    f"train_encoder_nfe={row['train_nfe_encoder']:.2f} | "
                    f"train_latent_nfe={row['train_nfe_latent']:.2f} | "
                    f"train_total_nfe={row['train_nfe']:.2f} | "
                    f"validation=skipped"
                )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        sync_cuda(device)
        test_start = time.time()
        test_metrics = evaluate_latent_ode(model, test_loader, config)
        sync_cuda(device)
        test_time = time.time() - test_start

        console.log(
            f"test | "
            f"time={test_time:.2f}s | "
            f"interp={test_metrics['interpolation_mse']:.6f} | "
            f"extrap={test_metrics['extrapolation_mse']:.6f} | "
            f"recon={test_metrics['recon_mse']:.6f} | "
            f"encoder_nfe={test_metrics['nfe_encoder']:.2f} | "
            f"latent_nfe={test_metrics['nfe_latent']:.2f} | "
            f"total_nfe={test_metrics['nfe']:.2f}"
        )

        persistence_metrics = evaluate_persistence(test_loader, config)
        console.log(
            f"persistence baseline | "
            f"interp={persistence_metrics['persistence_interpolation_mse']:.6f} | "
            f"extrap={persistence_metrics['persistence_extrapolation_mse']:.6f}"
        )

        wandb.log(
            {
                "phase": "latent_ode_test",
                "test_interp": float(test_metrics["interpolation_mse"]),
                "test_extrap": float(test_metrics["extrapolation_mse"]),
                "test_recon": float(test_metrics["recon_mse"]),
                "test_kl": float(test_metrics["kl"]),
                "test_encoder_nfe": float(test_metrics["nfe_encoder"]),
                "test_latent_nfe": float(test_metrics["nfe_latent"]),
                "test_nfe": float(test_metrics["nfe"]),
                "test_time_sec": test_time,
                "best_val_interp": best_val_interp,
                "persistence_interp": float(
                    persistence_metrics["persistence_interpolation_mse"]
                ),
                "persistence_extrap": float(
                    persistence_metrics["persistence_extrapolation_mse"]
                ),
            }
        )

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        run_tag = (
            run_name
            or f"seed{config.seed}_odernn{config.use_ode_rnn}_var{config.is_variational}_{config.method}"
        )
        out_path = results_dir / f"{run_tag}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": {
                        key: str(value) for key, value in config.__dict__.items()
                    },
                    "test_metrics": {
                        key: float(value) for key, value in test_metrics.items()
                    },
                    "best_val_interp": float(best_val_interp),
                    "test_time_sec": float(test_time),
                },
                f,
                indent=2,
            )
        console.log(f"saved metrics to {out_path}")

        return test_metrics
    finally:
        wandb.finish()
        cleanup_after_phase(device)


def parse_integrated_args() -> argparse.Namespace:
    ode_defaults = ODEConfig()
    ode_defaults.hidden_dim = 2
    ode_defaults.epochs = 200

    latent_defaults = LatentODEConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)

    add_prefixed_config_args(
        parser,
        config=ode_defaults,
        prefix="ode",
        title="Neural ODE options, originally from main.py",
    )
    add_prefixed_config_args(
        parser,
        config=latent_defaults,
        prefix="latent",
        title="Latent ODE options, originally from main_latentODE.py",
    )

    parser.add_argument("--latent-skip-validation", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_integrated_args()

    neural_config = build_neural_ode_config(args)
    latent_config = build_latent_ode_config(args)

    neural_run_name = f"{args.run_name}-neural_ode" if args.run_name else None
    latent_run_name = f"{args.run_name}-latent_ode" if args.run_name else None

    run_neural_ode(neural_config, run_name=neural_run_name)
    run_latent_ode(latent_config, run_name=latent_run_name)


if __name__ == "__main__":
    main()
