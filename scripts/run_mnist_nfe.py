"""C1/C3 — MNIST conv ODE-Net: forward-NFE dynamics over training, recon-checked.

Chen 2018: the number of function evaluations grows as training proceeds. We measure this on
the conv ODE-Net at a FAITHFUL tolerance (per-epoch reconstruction check on a fixed test-feature
batch; a tolerance where the flow does not reconstruct is not measuring an ODE). Uses the adjoint
(O(1) memory). Per-epoch trajectory + per-seed summary CSVs; every row carries recon_ok.

`--max_train_batches` caps batches per epoch (feasibility probe / bounded runs). `--epochs` and
`--seeds` control the sweep. This is our own seeded baseline, NOT a reproduction of Chen Table 1
(different arch/params/epochs; see DEVIATIONS C4).
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torchdiffeq import odeint

from data.dataloaders import get_mnist_dataloaders
from models.networks import ConvODENet
from training.engine import eval_epoch, train_epoch
from training.utils import set_seed


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def recon_check(model, x_img, tol, device):
    """Reconstruct the conv-ODE flow on a fixed test feature-map batch at `tol`."""
    h = model.downsampling(x_img.to(device))
    model.ode_func.nfe = 0
    fwd = odeint(model.ode_func, h, torch.tensor([0.0, 1.0], device=device), method="dopri5",
                 atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    nfe_fwd = model.ode_func.nfe
    back = odeint(model.ode_func, fwd, torch.tensor([1.0, 0.0], device=device), method="dopri5",
                  atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    rec = (back - h).flatten(1).norm(dim=1).max().item() / (h.flatten(1).norm(dim=1).max().item() + 1e-9)
    return rec, nfe_fwd


def run_seed(seed, cfg, device):
    set_seed(seed)
    train_loader, test_loader = get_mnist_dataloaders(batch_size=cfg.batch_size, flatten=False, seed=seed)
    model = ConvODENet(in_channels=1, num_filters=cfg.filters, num_classes=10,
                       solver_type="dopri5").to(device)
    # set the faithful tolerance on the ODE block
    model.ode_block.atol = cfg.tol
    model.ode_block.rtol = cfg.tol
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    x_fixed = next(iter(test_loader))[0][:64]      # fixed batch for the recon check

    import os
    if cfg.max_train_batches:
        os.environ["NODE_MAX_BATCHES"] = str(cfg.max_train_batches)

    traj = Path(cfg.results_dir) / "mnist_nfe_trajectory.csv"
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        tm = train_epoch(model, train_loader, opt, crit, device)
        dt = time.perf_counter() - t0
        rec, rec_nfe = recon_check(model, x_fixed, cfg.eval_tol, device)
        te = eval_epoch(model, test_loader, crit, device)
        row = {"seed": seed, "epoch": epoch + 1, "tol": cfg.tol, "eval_tol": cfg.eval_tol,
               "train_fwd_nfe": tm.get("forward_nfe_mean", float("nan")),
               "train_bwd_nfe": tm.get("backward_nfe_mean", float("nan")),
               "faithful_fwd_nfe": rec_nfe,  # forward NFE at the recon-checked eval tol
               "test_acc": te["accuracy"], "recon_rel": rec,
               "recon_ok": int(rec < cfg.recon_thresh), "epoch_s": round(dt, 1)}
        _append(traj, row)
        print(f"[s{seed} e{epoch+1}] fwd NFE {row['train_fwd_nfe']:.1f} bwd {row['train_bwd_nfe']:.0f} "
              f"| test {te['accuracy']:.4f} | recon {rec:.1e} {'OK' if row['recon_ok'] else 'LOOSE'} "
              f"| {dt:.0f}s/epoch", flush=True)
    os.environ.pop("NODE_MAX_BATCHES", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--eval_tol", type=float, default=1e-5)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--max_train_batches", type=int, default=0, help="0 = full epoch")
    p.add_argument("--results_dir", default="results/mnist_nfe")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    traj = Path(args.results_dir) / "mnist_nfe_trajectory.csv"
    if traj.exists():
        traj.unlink()
    print(f"MNIST NFE on {device} | filters {args.filters} | tol {args.tol} eval_tol {args.eval_tol} "
          f"| seeds {seeds} | epochs {args.epochs} | cap {args.max_train_batches or 'full'}", flush=True)
    for seed in seeds:
        run_seed(seed, args, device)
    print(f"Wrote {traj}", flush=True)


if __name__ == "__main__":
    main()
