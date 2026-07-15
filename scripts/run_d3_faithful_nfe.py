"""D3 faithfulness fix — re-measure NODE (and ANODE) NFE at a tolerance LADDER, recon-checked.

The committed D3 measured NFE at eval-tol 1e-5, where the NODE flow is recon_ok only 2/5 at ep8
(3 seeds' NODE fields stiffen past 1e-5). So the "NODE faithful NFE 86 / ANODE 1.7x cheaper"
comparison sits at a tol that is NOT integrating those NODE seeds. This retrains BOTH D3 models
(deterministic, same seeds => the exact committed fields) and, per epoch, measures forward NFE +
reconstruction at {1e-5, 1e-6, 1e-7} (capped). We then report the ANODE/NODE NFE ratio at the
LOOSEST tol where BOTH models are recon_ok 5/5 through the budget -- an apples-to-apples faithful
comparison. If even 1e-7 can't integrate the stiffest NODE seeds, that is recorded as data.

Writes results/d3_faithful/d3_faithful_trajectory.csv (one row per model x seed x epoch x tol).
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
from training.engine import train_epoch
from training.utils import set_seed
from scripts.run_d3_anode_mnist import DupontConvODE, build, n_params


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def nfe_recon_at(model, x_img, tol, device, cap):
    """forward NFE and forward->backward reconstruction (relative) at eval tol `tol`."""
    x = x_img.to(device)
    if model.augment > 0:
        x = torch.cat([x, torch.zeros(x.size(0), model.augment, x.size(2), x.size(3), device=device)], dim=1)
    model.ode_func.nfe = 0
    try:
        f = odeint(model.ode_func, x, torch.tensor([0.0, 1.0], device=device), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        nfe = model.ode_func.nfe
        b = odeint(model.ode_func, f, torch.tensor([1.0, 0.0], device=device), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        rel = ((b - x).flatten(1).norm(dim=1).max() / (x.flatten(1).norm(dim=1).max() + 1e-9)).item()
        return nfe, rel
    except Exception:
        return -1, float("nan")  # capped -> too stiff to integrate at this tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--eval_tols", default="1e-5,1e-6,1e-7")
    p.add_argument("--cap", type=int, default=500_000)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/d3_faithful")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tols = [float(t) for t in args.eval_tols.split(",")]
    traj = Path(args.results_dir) / "d3_faithful_trajectory.csv"
    if traj.exists():
        traj.unlink()
    print(f"D3 faithful re-measure on {device} | eval tols {tols} | seeds {seeds} | {args.epochs} ep",
          flush=True)
    for spec in ("NODE", "ANODE-p5"):
        for seed in seeds:
            set_seed(seed)  # identical to the committed D3 run -> same fields
            tr, te = get_mnist_dataloaders(batch_size=args.batch_size, flatten=False, seed=seed)
            model = build(spec, args.tol).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            crit = nn.CrossEntropyLoss()
            x_fixed = next(iter(te))[0][:64]
            for epoch in range(args.epochs):
                t0 = time.perf_counter()
                train_epoch(model, tr, opt, crit, device)
                dt = time.perf_counter() - t0
                cells = []
                for tol in tols:
                    nfe, rel = nfe_recon_at(model, x_fixed, tol, device, args.cap)
                    ok = int(rel == rel and rel < args.recon_thresh)
                    _append(traj, {"model": spec, "seed": seed, "epoch": epoch + 1, "eval_tol": tol,
                                   "fwd_nfe": nfe, "recon_rel": rel, "recon_ok": ok,
                                   "capped": int(nfe < 0)})
                    cells.append(f"{tol:.0e}:nfe{nfe}{'ok' if ok else 'X'}")
                print(f"[{spec} s{seed} e{epoch+1}] " + " ".join(cells) + f" | {dt:.0f}s", flush=True)
    print(f"Wrote {traj}", flush=True)


if __name__ == "__main__":
    main()
