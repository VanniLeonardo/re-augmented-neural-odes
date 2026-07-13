from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


BatchDict = Mapping[str, torch.Tensor]


def move_batch_to_device(
    batch: BatchDict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Moves a dictionary batch to the requested device."""
    return {key: value.to(device) for key, value in batch.items()}


def masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Computes mean squared error over a mask.

    Supported mask shapes:
    - [batch, time]
    - [batch, time, dim]
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if mask.ndim == 2:
        expanded_mask = mask.unsqueeze(-1).expand_as(predictions)
    elif mask.ndim == 3:
        expanded_mask = mask.expand_as(predictions)
    else:
        raise ValueError("mask must have shape [batch, time] or [batch, time, dim].")

    if not bool(expanded_mask.any().item()):
        return predictions.new_tensor(0.0)

    squared_error = (predictions - targets).pow(2)
    return torch.masked_select(squared_error, expanded_mask).mean()


def compute_context_training_metrics(
    context_predictions: torch.Tensor,
    batch: BatchDict,
) -> Dict[str, torch.Tensor]:
    """Metrics fortraining: query context only, optimize observed context only."""
    batch_size, context_steps, output_dim = context_predictions.shape

    if batch["context_values"].shape != (batch_size, context_steps, output_dim):
        raise ValueError(
            "context_values shape is inconsistent with context_predictions."
        )

    if batch["context_mask"].shape != (batch_size, context_steps, 1):
        raise ValueError("context_mask must have shape [batch, T_ctx, 1].")

    if batch["interp_mask"].shape != (batch_size, context_steps, 1):
        raise ValueError("interp_mask must have shape [batch, T_ctx, 1].")

    gt_context = batch["ground_truth"][:, :context_steps, :]

    observed_mse = masked_mse(
        context_predictions,
        batch["context_values"],
        batch["context_mask"],
    )
    interpolation_mse = masked_mse(
        context_predictions,
        gt_context,
        batch["interp_mask"],
    )

    return {
        "loss": observed_mse,
        "observed_mse": observed_mse,
        "interpolation_mse": interpolation_mse,
    }


def compute_timeseries_metrics(
    predictions: torch.Tensor,
    batch: BatchDict,
) -> Dict[str, torch.Tensor]:
    """Computes reconstruction/interpolation/extrapolation metrics."""
    ground_truth = batch["ground_truth"]

    batch_size, total_steps, output_dim = predictions.shape
    context_steps = batch["context_times"].size(1)
    future_steps = total_steps - context_steps

    if ground_truth.shape != predictions.shape:
        raise ValueError("ground_truth and predictions must have identical shape.")

    if batch["context_values"].shape != (batch_size, context_steps, output_dim):
        raise ValueError("context_values shape is inconsistent with predictions.")

    if batch["context_mask"].shape != (batch_size, context_steps, 1):
        raise ValueError("context_mask must have shape [batch, T_ctx, 1].")

    if batch["interp_mask"].shape != (batch_size, context_steps, 1):
        raise ValueError("interp_mask must have shape [batch, T_ctx, 1].")

    if batch["future_mask"].shape != (batch_size, future_steps, 1):
        raise ValueError("future_mask must have shape [batch, T_future, 1].")

    pred_context = predictions[:, :context_steps, :]
    pred_future = predictions[:, context_steps:, :]
    gt_context = ground_truth[:, :context_steps, :]
    gt_future = ground_truth[:, context_steps:, :]

    full_mask = torch.ones_like(ground_truth, dtype=torch.bool)

    return {
        "loss": masked_mse(predictions, ground_truth, full_mask),
        "observed_mse": masked_mse(
            pred_context,
            batch["context_values"],
            batch["context_mask"],
        ),
        "interpolation_mse": masked_mse(
            pred_context,
            gt_context,
            batch["interp_mask"],
        ),
        "extrapolation_mse": masked_mse(
            pred_future,
            gt_future,
            batch["future_mask"],
        ),
    }


def train_timeseries_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_mode: str = "observed_context",
) -> Dict[str, float]:
    """Runs one training epoch for time-series forecasting."""
    model.train()

    total_loss = 0.0
    total_observed_mse = 0.0
    total_interpolation_mse = 0.0
    total_extrapolation_mse = 0.0
    total_extrapolation_count = 0
    total_nfe = 0.0
    total_samples = 0
    total_batches = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["context_times"].size(0)

        optimizer.zero_grad()

        if loss_mode == "observed_context":
            predictions = model(
                context_times=batch["context_times"],
                observed_context=batch["observed_context"],
                context_mask=batch["context_mask"],
                full_times=batch["context_times"],
            )
            metrics = compute_context_training_metrics(predictions, batch)
            extrapolation_value = None

        elif loss_mode == "full_trajectory":
            predictions = model(
                context_times=batch["context_times"],
                observed_context=batch["observed_context"],
                context_mask=batch["context_mask"],
                full_times=batch["full_times"],
            )
            metrics = compute_timeseries_metrics(predictions, batch)
            extrapolation_value = metrics["extrapolation_mse"].item()

        else:
            raise ValueError(
                f"Unknown loss_mode={loss_mode!r}. "
                "Expected 'observed_context' or 'full_trajectory'."
            )

        loss = metrics["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_observed_mse += metrics["observed_mse"].item() * batch_size
        total_interpolation_mse += metrics["interpolation_mse"].item() * batch_size

        if extrapolation_value is not None:
            total_extrapolation_mse += extrapolation_value * batch_size
            total_extrapolation_count += batch_size

        total_nfe += float(getattr(model, "get_nfe", lambda: 0)())
        total_samples += batch_size
        total_batches += 1

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "loss": total_loss / total_samples,
        "observed_mse": total_observed_mse / total_samples,
        "interpolation_mse": total_interpolation_mse / total_samples,
        "extrapolation_mse": (
            total_extrapolation_mse / total_extrapolation_count
            if total_extrapolation_count > 0
            else float("nan")
        ),
        "nfe_per_sample": total_nfe / total_samples if total_samples > 0 else 0.0,
        "nfe_per_batch": total_nfe / total_batches if total_batches > 0 else 0.0,
        "memory_mb": peak_memory_mb,
    }


@torch.no_grad()
def evaluate_timeseries(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates a time-series model."""
    model.eval()

    total_loss = 0.0
    total_observed_mse = 0.0
    total_interpolation_mse = 0.0
    total_extrapolation_mse = 0.0
    total_nfe = 0.0
    total_samples = 0
    total_batches = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["context_times"].size(0)

        predictions = model(
            context_times=batch["context_times"],
            observed_context=batch["observed_context"],
            context_mask=batch["context_mask"],
            full_times=batch["full_times"],
        )
        metrics = compute_timeseries_metrics(predictions, batch)

        total_loss += metrics["loss"].item() * batch_size
        total_observed_mse += metrics["observed_mse"].item() * batch_size
        total_interpolation_mse += metrics["interpolation_mse"].item() * batch_size
        total_extrapolation_mse += metrics["extrapolation_mse"].item() * batch_size
        total_nfe += float(getattr(model, "get_nfe", lambda: 0)())
        total_samples += batch_size
        total_batches += 1

    return {
        "loss": total_loss / total_samples,
        "observed_mse": total_observed_mse / total_samples,
        "interpolation_mse": total_interpolation_mse / total_samples,
        "extrapolation_mse": total_extrapolation_mse / total_samples,
        "nfe_per_sample": total_nfe / total_samples if total_samples > 0 else 0.0,
        "nfe_per_batch": total_nfe / total_batches if total_batches > 0 else 0.0,
    }
