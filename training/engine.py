import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from training.utils import NFEStats


def _max_batches(explicit: Optional[int] = None) -> Optional[int]:
    """Batch cap for fast smoke/CI runs.

    Priority: explicit argument > ``$NODE_MAX_BATCHES`` env var > None (no cap).
    Lets ``make smoke`` exercise the full training pipeline in seconds without a GPU.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get("NODE_MAX_BATCHES")
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Trains the model for a single epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    nfe_stats = NFEStats()
    has_ode_func = hasattr(model, "ode_func")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cap = _max_batches()
    for batch_idx, (x, y) in enumerate(dataloader):
        if cap is not None and batch_idx >= cap:
            break
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        forward_nfe = model.ode_func.nfe if has_ode_func else 0

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if has_ode_func:
            nfe_stats.record(
                forward=forward_nfe,
                backward=model.ode_func.nfe - forward_nfe,
            )

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total_samples += x.size(0)

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples,
    }
    if device.type == "cuda":
        metrics["memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    if nfe_stats:
        metrics.update(nfe_stats.summary())

    return metrics


def eval_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the model for a single epoch without gradient computation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    has_ode_func = hasattr(model, "ode_func")
    fwd_nfe: List[int] = []

    cap = _max_batches()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if cap is not None and batch_idx >= cap:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)

            if has_ode_func:
                fwd_nfe.append(model.ode_func.nfe)

            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total_samples += x.size(0)

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples,
    }
    if fwd_nfe:
        metrics["forward_nfe_mean"] = sum(fwd_nfe) / len(fwd_nfe)

    return metrics
