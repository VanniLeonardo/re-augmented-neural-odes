"""Guard the C2 headline (backward NFE ~ forward NFE, NOT Chen's 1/2) across tolerance.

The fixed-step unit test (tests/test_nfe_split.py) proves bwd==fwd==4N EXACTLY and is
tolerance-free. This script guards the ADAPTIVE (dopri5) claim, which is the one that
was previously only measured at atol=rtol=1e-3 -- a tolerance we showed can be
NON-INTEGRATING for the near-singular fields this task induces. It trains a NODE, then
sweeps the eval tolerance and reports, per tolerance: forward NFE, backward NFE, their
ratio, and a forward->backward reconstruction error. A tolerance whose reconstruction
fails is flagged: its NFE ratio is not a property of the flow. The C2 finding is only
safe if the ratio is ~1 (>> Chen's 0.5) in the regime where reconstruction passes.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torchdiffeq import odeint

from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import set_seed

DEVICE = torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", default="spheres")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_tol", type=float, default=1e-6)
    p.add_argument("--tolerances", default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8")
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/c2")
    args = p.parse_args()

    set_seed(args.seed)
    n = 1200 if args.geometry == "spheres" else 1000
    tr, va = get_dataloaders(args.geometry, n_samples=n, batch_size=64,
                             val_split=0.2, noise=0.05, seed=args.seed)
    model = ODENet(data_dim=2, hidden_dim=2, num_classes=2, augment_dim=0,
                   ode_hidden_dim=32, use_stem=False, head_hidden_dim=None,
                   solver_type="dopri5", atol=args.train_tol, rtol=args.train_tol,
                   solver_options={"max_num_steps": 10_000_000}).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    print(f"Training NODE on {args.geometry} at tol {args.train_tol}, {args.epochs} ep ...", flush=True)
    for _ in range(args.epochs):
        train_epoch(model, tr, opt, crit, DEVICE)

    Xval = torch.stack([va.dataset[i][0] for i in range(len(va.dataset))]).to(DEVICE)
    xb = Xval[:128].clone().requires_grad_(True)

    rows = []
    print(f"\n{'tol':>8s} {'fwd NFE':>8s} {'bwd NFE':>8s} {'bwd/fwd':>8s} "
          f"{'recon max':>11s} {'integrating?':>13s}")
    print("-" * 64)
    for tol in [float(t) for t in args.tolerances.split(",")]:
        model.ode_block.atol = tol
        model.ode_block.rtol = tol
        # forward+backward through the adjoint, snapshot the split like the engine does
        model.ode_func.nfe = 0
        logits = model(xb)
        fwd = model.ode_func.nfe
        logits.sum().backward()
        bwd = model.ode_func.nfe - fwd
        xb.grad = None
        # reconstruction at this tolerance
        with torch.no_grad():
            f1 = odeint(model.ode_func, Xval, torch.tensor([0.0, 1.0]), method="dopri5",
                        atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
            b1 = odeint(model.ode_func, f1, torch.tensor([1.0, 0.0]), method="dopri5",
                        atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
            recon = (b1 - Xval).norm(dim=1).max().item()
        integrating = recon < args.recon_thresh
        ratio = bwd / fwd if fwd else float("nan")
        print(f"{tol:8.0e} {fwd:8d} {bwd:8d} {ratio:8.2f} {recon:11.2e} "
              f"{'YES' if integrating else 'NO (loose)':>13s}")
        rows.append({"tol": tol, "fwd_nfe": fwd, "bwd_nfe": bwd, "bwd_over_fwd": ratio,
                     "recon_max": recon, "integrating": int(integrating)})

    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "c2_tolerance.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    integ = [r for r in rows if r["integrating"]]
    if integ:
        rr = [r["bwd_over_fwd"] for r in integ]
        print(f"\nIn the INTEGRATING regime (recon < {args.recon_thresh}): bwd/fwd in "
              f"[{min(rr):.2f}, {max(rr):.2f}] -- Chen reports 0.5. "
              f"C2 {'HOLDS (ratio >> 0.5, tolerance-stable)' if min(rr) > 0.8 else 'is UNSAFE'}.")
    print(f"Wrote {out}/c2_tolerance.csv")


if __name__ == "__main__":
    main()
