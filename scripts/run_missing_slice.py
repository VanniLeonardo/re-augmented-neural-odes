"""D2 — missing-slice generalisation (Dupont §5.1 / Fig 9), at ACCURATE tolerance.

Dupont: remove an angular wedge (his example [0, pi/5]) from the TRAINING set; NODEs then
show a *large generalisation gap* on the held-out wedge -- *"because the flow moves through
the gaps in the training set"* -- while *"ANODEs generalise much better and achieve near zero
validation loss."* We test this faithfully: train NODE vs ANODE-p1 at accurate tol (recon
checked), and report accuracy/loss on the held-out wedge (`slice`) separately from the full
independent validation set. The slice metric is the sharp generalisation quantity.

The historical `train_anode_slice_circles.py` measured this at atol=rtol=1e-3 (non-integrating);
this script is the accurate-tolerance replacement. Every row carries `recon_ok`.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse
import csv
import math
import sys
import platform
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

from data.synthetic import make_spheres, make_circles
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import set_seed

DEVICE = torch.device("cpu")
NOISE, LR = 0.05, 3e-3
_GEOM = {"spheres": (1200, make_spheres), "circles": (1000, make_circles)}
CRIT = nn.CrossEntropyLoss()


def augment_of(model_name: str) -> int:
    """'NODE' -> 0, 'ANODE-p<k>' -> k (any k, so the §6 grid can sweep augmentation)."""
    return 0 if model_name == "NODE" else int(model_name.split("-p")[1])


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as h:
            for line in h:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _angle_mask(x: torch.Tensor, start: float, width: float) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    theta = torch.remainder(torch.atan2(x[:, 1], x[:, 0]), two_pi)
    start = start % two_pi
    end = start + width
    if end <= two_pi:
        return (theta >= start) & (theta <= end)
    return (theta >= start) | (theta <= end - two_pi)


@torch.no_grad()
def _flow_rk4(model: ODENet, X: torch.Tensor, nsteps: int = 1000) -> torch.Tensor:
    """Accurate, batch-independent terminal state (append ANODE zeros as forward() does)."""
    x = X.to(DEVICE)
    if model.augment_dim > 0:
        x = torch.cat([x, torch.zeros(x.size(0), model.augment_dim)], dim=1)
    out = []
    for i in range(0, len(x), 4000):
        out.append(odeint(model.ode_func, x[i:i + 4000], torch.tensor([0.0, 1.0]),
                          method="rk4", options={"step_size": 1.0 / nsteps})[1])
    return torch.cat(out)


@torch.no_grad()
def _acc_loss(model: ODENet, X: torch.Tensor, Y: torch.Tensor) -> Tuple[float, float]:
    logits = model.fc(_flow_rk4(model, X))
    acc = (logits.argmax(1).cpu() == Y).float().mean().item()
    loss = CRIT(logits, Y.to(DEVICE)).item()
    return acc, loss


@torch.no_grad()
def _recon_and_nfe(model: ODENet, X: torch.Tensor, tol: float) -> Tuple[float, float]:
    x = X.to(DEVICE)
    if model.augment_dim > 0:
        x = torch.cat([x, torch.zeros(x.size(0), model.augment_dim)], dim=1)
    model.ode_func.nfe = 0
    fwd = odeint(model.ode_func, x, torch.tensor([0.0, 1.0]), method="dopri5",
                 atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    nfe = model.ode_func.nfe
    back = odeint(model.ode_func, fwd, torch.tensor([1.0, 0.0]), method="dopri5",
                  atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    return (back - x).norm(dim=1).max().item(), float(nfe)


def run(model_name: str, augment_dim: int, seed: int, cfg) -> Dict[str, Any]:
    """Train one cell from scratch and return its result row (the caller writes it)."""
    set_seed(seed)
    n, gen = _GEOM[cfg.geometry]
    Xtr, Ytr = gen(n_samples=n, noise=NOISE, seed=seed)
    keep = ~_angle_mask(Xtr, cfg.start, cfg.width)          # remove the wedge from TRAIN
    Xtr, Ytr = Xtr[keep], Ytr[keep]
    # independent validation pools
    Xv, Yv = gen(n_samples=cfg.n_val, noise=NOISE, seed=seed + 10_000)
    smask = _angle_mask(Xv, cfg.start, cfg.width)
    Xs, Ys = Xv[smask], Yv[smask]                            # held-out wedge = generalisation test

    tr = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=True)
    model = ODENet(data_dim=2, hidden_dim=2, num_classes=2, augment_dim=augment_dim,
                   ode_hidden_dim=32, use_stem=False, head_hidden_dim=None,
                   solver_type="dopri5", atol=cfg.train_tol, rtol=cfg.train_tol,
                   solver_options={"max_num_steps": 10_000_000}).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(cfg.epochs):
        train_epoch(model, tr, opt, CRIT, DEVICE)
    model.eval()

    tr_acc, tr_loss = _acc_loss(model, Xtr, Ytr)
    fv_acc, fv_loss = _acc_loss(model, Xv, Yv)
    sl_acc, sl_loss = _acc_loss(model, Xs, Ys)
    # Observed-region-only validation (§6): the full validation set CONTAINS the removed
    # wedge, which blurs in-distribution generalisation with held-out extrapolation.
    ob_acc, ob_loss = _acc_loss(model, Xv[~smask], Yv[~smask])
    recon, nfe = _recon_and_nfe(model, Xv, cfg.eval_tol)
    row = {
        "model": model_name, "augment_dim": augment_dim, "geometry": cfg.geometry, "seed": seed,
        "epochs": cfg.epochs, "width_over_pi": round(cfg.width / math.pi, 4),
        "n_train_after_slice": len(Xtr), "n_slice_val": len(Xs),
        "train_acc": tr_acc, "train_loss": tr_loss,
        "full_val_acc": fv_acc, "full_val_loss": fv_loss,
        "slice_val_acc": sl_acc, "slice_val_loss": sl_loss,
        "obs_val_acc": ob_acc, "obs_val_loss": ob_loss,
        "gen_gap_loss": sl_loss - tr_loss, "fwd_nfe": nfe,
        "recon_max": recon, "recon_ok": int(recon < cfg.recon_thresh),
        "train_tol": cfg.train_tol, "eval_tol": cfg.eval_tol, "hardware": cpu_name(),
    }
    print(f"[{cfg.geometry[:3]} {model_name} s{seed}] train_loss {tr_loss:.3f} | "
          f"SLICE acc {sl_acc:.3f} loss {sl_loss:.3f} | full acc {fv_acc:.3f} | "
          f"gap {row['gen_gap_loss']:+.3f} | NFE {nfe:.0f} | "
          f"recon {recon:.1e} {'OK' if row['recon_ok'] else 'LOOSE'}", flush=True)
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", default="circles", choices=list(_GEOM))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--width", type=float, default=math.pi / 5.0)  # Dupont's [0, pi/5] example
    p.add_argument("--n_val", type=int, default=3000)
    p.add_argument("--train_tol", type=float, default=1e-6)
    p.add_argument("--eval_tol", type=float, default=1e-6)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/slice")
    p.add_argument("--models", default="NODE,ANODE-p1")
    return p.parse_args()


def main():
    cfg = parse_args()
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    models = [m for m in cfg.models.split(",") if m.strip()]
    out = Path(cfg.results_dir) / "slice_raw.csv"
    if out.exists():
        out.unlink()
    print(f"D2 missing-slice | geometry {cfg.geometry} | wedge [0,{cfg.width/math.pi:.3f}pi] | "
          f"models {models} | seeds {seeds} | {cfg.epochs} ep | tol {cfg.train_tol}", flush=True)
    for model_name in models:
        for seed in seeds:
            _append(out, run(model_name, augment_of(model_name), seed, cfg))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
