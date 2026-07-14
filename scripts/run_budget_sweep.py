"""Budget-dependence of the topological obstruction (spheres, accurate tolerance).

Tests the hypothesis that Dupont's reported NODE *failure* on the nested spheres is
BUDGET-dependent, not architectural: a homeomorphic flow stretches the inner disk
into an ever-thinner tendril threading the annulus gap, so as training proceeds the
obstruction CONVERTS from an accuracy failure into an unbounded NFE cost -- accuracy
climbs toward (never reaching) 100% while the forward NFE required to integrate the
(increasingly stiff) field grows.

Everything is measured at an ACCURATE tolerance and validated with a forward->backward
reconstruction check (a flow that does not reconstruct is not being integrated). NODE
(augment_dim=0) and ANODE-p1 (augment_dim=1) are swept over training budgets and seeds
from a SINGLE invocation; each (model, seed) trains once to the max budget and is
snapshotted at every budget. Rows are appended immediately so partial runs survive.

Outputs (results/budget/):
  budget_raw.csv        one row per (model, seed, budget): acc/loss/NFE(median,IQR,max)/recon
  budget_trajectory.csv one row per (model, seed, epoch): train fwd NFE + train acc
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU: faster for tiny 2-D, no GPU contention

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from data.synthetic import make_spheres
from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import set_seed

DEVICE = torch.device("cpu")
N, NOISE, LR = 1200, 0.05, 3e-3


def _append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def make_model(augment_dim: int, tol: float) -> ODENet:
    return ODENet(
        data_dim=2, hidden_dim=2, num_classes=2, augment_dim=augment_dim,
        ode_hidden_dim=32, use_stem=False, head_hidden_dim=None,
        solver_type="dopri5", atol=tol, rtol=tol,
        solver_options={"max_num_steps": 10_000_000},
    ).to(DEVICE)


@torch.no_grad()
def nfe_distribution(model: ODENet, X: torch.Tensor, bs: int = 64) -> Dict[str, float]:
    """Per-batch forward NFE over X (adaptive NFE is a per-batch quantity)."""
    vals: List[int] = []
    for i in range(0, len(X), bs):
        model.ode_func.nfe = 0
        model(X[i:i + bs].to(DEVICE))
        vals.append(model.ode_func.nfe)
    a = np.array(vals, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"fwd_nfe_median": med, "fwd_nfe_q1": q1, "fwd_nfe_q3": q3, "fwd_nfe_max": a.max()}


@torch.no_grad()
def accurate_acc(model: ODENet, X: torch.Tensor, Y: torch.Tensor, nsteps: int = 1000) -> float:
    """Continuum accuracy via a fine fixed-step rk4 flow (batch-independent, accurate)."""
    preds = []
    for i in range(0, len(X), 4000):
        x = X[i:i + 4000].to(DEVICE)
        if model.augment_dim > 0:  # append the ANODE augmentation zeros, as forward() does
            x = torch.cat([x, torch.zeros(x.size(0), model.augment_dim)], dim=1)
        phi = odeint(model.ode_func, x, torch.tensor([0.0, 1.0]),
                     method="rk4", options={"step_size": 1.0 / nsteps})[1]
        preds.append(model.fc(phi).argmax(1).cpu())
    return (torch.cat(preds) == Y).float().mean().item()


@torch.no_grad()
def recon_error(model: ODENet, X: torch.Tensor, tol: float) -> Dict[str, float]:
    """Forward 0->1 then backward 1->0 at the eval tolerance; a genuine flow reconstructs."""
    x = X.to(DEVICE)
    # augmentation is appended as zeros inside forward(); do the same here for ANODE
    if model.augment_dim > 0:
        x = torch.cat([x, torch.zeros(x.size(0), model.augment_dim)], dim=1)
    fwd = odeint(model.ode_func, x, torch.tensor([0.0, 1.0]), method="dopri5",
                 atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    back = odeint(model.ode_func, fwd, torch.tensor([1.0, 0.0]), method="dopri5",
                  atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    e = (back - x).norm(dim=1)
    return {"recon_max": e.max().item(), "recon_mean": e.mean().item()}


@torch.no_grad()
def val_metrics(model: ODENet, loader, crit) -> Dict[str, float]:
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x.to(DEVICE))
        tot_loss += crit(logits, y.to(DEVICE)).item() * x.size(0)
        correct += (logits.argmax(1).cpu() == y).sum().item()
        n += x.size(0)
    return {"val_acc": correct / n, "val_loss": tot_loss / n}


def run(model_name: str, augment_dim: int, seed: int, budgets: List[int], cfg) -> None:
    set_seed(seed)
    tr, va = get_dataloaders("spheres", n_samples=N, batch_size=64,
                             val_split=0.2, noise=NOISE, seed=seed)
    model = make_model(augment_dim, cfg.train_tol)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    # fixed fresh test set for continuum accuracy + NFE distribution (same for all snapshots)
    Xtest, Ytest = make_spheres(n_samples=12_000, noise=NOISE, seed=100_000 + seed)
    Xnfe, _ = make_spheres(n_samples=3_000, noise=NOISE, seed=200_000 + seed)

    traj = Path(cfg.results_dir) / "budget_trajectory.csv"
    raw = Path(cfg.results_dir) / "budget_raw.csv"

    done = 0
    t0 = time.perf_counter()
    for budget in budgets:
        for ep in range(done, budget):
            m = train_epoch(model, tr, opt, crit, DEVICE)
            _append(traj, {"model": model_name, "seed": seed, "epoch": ep + 1,
                           "train_acc": m["accuracy"],
                           "train_fwd_nfe": m.get("forward_nfe_mean", float("nan"))})
        done = budget

        Xval = torch.stack([va.dataset[i][0] for i in range(len(va.dataset))])
        vm = val_metrics(model, va, crit)
        nd = nfe_distribution(model, Xnfe)
        rc = recon_error(model, Xval, cfg.eval_tol)
        dense = accurate_acc(model, Xtest, Ytest)
        row = {
            "model": model_name, "augment_dim": augment_dim, "seed": seed, "budget": budget,
            "train_tol": cfg.train_tol, "eval_tol": cfg.eval_tol,
            **vm, "dense_acc": dense, **nd, **rc,
            "recon_ok": int(rc["recon_max"] < cfg.recon_thresh),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        _append(raw, row)
        print(f"[{model_name} s{seed} b{budget:4d}] val {vm['val_acc']:.3f} "
              f"dense {dense:.4f} | NFE med {nd['fwd_nfe_median']:.0f} "
              f"[{nd['fwd_nfe_q1']:.0f},{nd['fwd_nfe_q3']:.0f}] max {nd['fwd_nfe_max']:.0f} "
              f"| recon {rc['recon_max']:.1e} {'OK' if row['recon_ok'] else 'LOOSE'} "
              f"| {row['elapsed_s']:.0f}s", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--budgets", type=str, default="25,50,100,200,500,1000")
    p.add_argument("--train_tol", type=float, default=1e-6)
    p.add_argument("--eval_tol", type=float, default=1e-6)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", type=str, default="results/budget")
    p.add_argument("--models", type=str, default="NODE,ANODE-p1")
    return p.parse_args()


def main():
    cfg = parse_args()
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    budgets = [int(b) for b in cfg.budgets.split(",") if b.strip()]
    spec = {"NODE": 0, "ANODE-p1": 1}
    models = [m for m in cfg.models.split(",") if m.strip()]

    out = Path(cfg.results_dir)
    for fn in ("budget_raw.csv", "budget_trajectory.csv"):
        if (out / fn).exists():
            (out / fn).unlink()
    print(f"Budget sweep on {DEVICE} | models {models} | seeds {seeds} | budgets {budgets} "
          f"| train_tol {cfg.train_tol} eval_tol {cfg.eval_tol}", flush=True)

    for model_name in models:
        for seed in seeds:
            run(model_name, spec[model_name], seed, budgets, cfg)
    print(f"Wrote {out}/budget_raw.csv and budget_trajectory.csv", flush=True)


if __name__ == "__main__":
    main()
