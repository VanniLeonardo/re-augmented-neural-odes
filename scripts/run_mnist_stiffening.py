"""C1/C3 confirmation — the MNIST conv field STIFFENS with training (mechanism, not defect).

The epochs-1-3 faithful window at eval-tol 1e-5 (from the committed C1/C3 run) is a finding: the
conv field gets stiffer as it trains until 1e-5 no longer reconstructs. This confirms the
mechanism by measuring, per epoch, the recon error AND forward NFE at a LADDER of eval tolerances
{1e-5, 1e-6, 1e-7}. Two things should show:
  (a) faithful-NFE (at the tightest recon_ok tol) grows monotonically over training;
  (b) the LOOSEST tol that stays recon-faithful TIGHTENS as training proceeds.
Train tol fixed at 1e-3 (as C1/C3). Recon solves capped so a stiff tight-tol solve can't hang.
Writes results/mnist_stiffening/stiffening_trajectory.csv (one row per epoch x eval_tol; recon_ok).
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
from training.engine import train_epoch
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
def recon_at(model, h0, tol, device, cap):
    model.ode_func.nfe = 0
    try:
        f = odeint(model.ode_func, h0, torch.tensor([0.0, 1.0], device=device), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        nfe = model.ode_func.nfe
        b = odeint(model.ode_func, f, torch.tensor([1.0, 0.0], device=device), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        rel = ((b - h0).flatten(1).norm(dim=1).max() / (h0.flatten(1).norm(dim=1).max() + 1e-9)).item()
        return rel, nfe
    except Exception:
        return float("nan"), -1  # capped -> too stiff at this tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--eval_tols", default="1e-5,1e-6,1e-7")
    p.add_argument("--cap", type=int, default=200_000)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/mnist_stiffening")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tols = [float(t) for t in args.eval_tols.split(",")]
    traj = Path(args.results_dir) / "stiffening_trajectory.csv"
    if traj.exists():
        traj.unlink()
    print(f"MNIST stiffening on {device} | filters {args.filters} | train tol {args.tol} "
          f"| eval tols {tols} | seeds {seeds} | {args.epochs} ep", flush=True)
    for seed in seeds:
        set_seed(seed)
        tr, te = get_mnist_dataloaders(batch_size=args.batch_size, flatten=False, seed=seed)
        model = ConvODENet(in_channels=1, num_filters=args.filters, num_classes=10).to(device)
        model.ode_block.atol = args.tol; model.ode_block.rtol = args.tol
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = nn.CrossEntropyLoss()
        x_fixed = next(iter(te))[0][:64].to(device)
        for epoch in range(args.epochs):
            t0 = time.perf_counter()
            tm = train_epoch(model, tr, opt, crit, device)
            dt = time.perf_counter() - t0
            with torch.no_grad():
                h0 = model.downsampling(x_fixed)
            best_ok_tol = None
            for tol in tols:
                rel, nfe = recon_at(model, h0, tol, device, args.cap)
                ok = int(rel == rel and rel < args.recon_thresh)  # nan-safe
                if ok and (best_ok_tol is None or tol > best_ok_tol):
                    best_ok_tol = tol
                _append(traj, {"seed": seed, "epoch": epoch + 1, "train_tol": args.tol,
                               "eval_tol": tol, "train_fwd_nfe": tm.get("forward_nfe_mean", float("nan")),
                               "eval_fwd_nfe": nfe, "recon_rel": rel, "recon_ok": ok,
                               "train_acc": tm["accuracy"], "epoch_s": round(dt, 1)})
            print(f"[s{seed} e{epoch+1}] train NFE {tm.get('forward_nfe_mean',0):.1f} | "
                  f"loosest recon_ok tol = {best_ok_tol} | {dt:.0f}s", flush=True)
    print(f"Wrote {traj}", flush=True)


if __name__ == "__main__":
    main()
