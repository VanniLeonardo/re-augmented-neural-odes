"""C2 consolidation — add MNIST conv to the bwd/fwd NFE tolerance surface.

Measures the adaptive backward/forward NFE ratio of the MNIST conv ODE field across a tolerance
axis, for an UNTRAINED field and a lightly TRAINED one (a few epochs), with a per-tolerance
reconstruction check. This is the third field for the C2 tolerance x field surface (the toy
spheres/circles data is in results/c2/c2_recharacterise.csv). GPU (peak-mem not needed; adjoint).

Small batch (conv NFE is expensive); tolerances default 1e-3..1e-6 (1e-7 backward on a conv field
is very slow -- skipped unless requested). Writes results/c2/c2_mnist.csv.
"""
from __future__ import annotations
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

from data.dataloaders import get_mnist_dataloaders
from models.networks import ConvODENet
from training.engine import train_epoch
from training.utils import set_seed


def measure(model, h0, tol, device, cap=50_000):
    """One fwd+bwd through the adjoint on feature map h0; snapshot NFE split; recon at tol.
    max_num_steps capped so a pathological tight-tol solve is bounded (recorded, not a hang)."""
    model.ode_block.atol = tol; model.ode_block.rtol = tol
    model.ode_block.options = {"max_num_steps": cap}
    x = h0.detach().requires_grad_(True)
    model.ode_func.nfe = 0
    out = model.ode_block(x)
    fwd = model.ode_func.nfe
    out.pow(2).sum().backward()
    bwd = model.ode_func.nfe - fwd
    with torch.no_grad():
        f = odeint(model.ode_func, h0, torch.tensor([0.0, 1.0], device=device), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        b = odeint(model.ode_func, f, torch.tensor([1.0, 0.0], device=device), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        rec = ((b - h0).flatten(1).norm(dim=1).max() / (h0.flatten(1).norm(dim=1).max() + 1e-9)).item()
    return fwd, bwd, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--train_epochs", type=int, default=3, help="lightly-trained state (faithful window)")
    p.add_argument("--tolerances", default="1e-3,1e-4,1e-5")
    p.add_argument("--cap", type=int, default=50_000, help="max_num_steps cap per solve")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/c2")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA"); return
    device = torch.device("cuda")
    set_seed(args.seed)
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128, flatten=False, seed=args.seed)
    model = ConvODENet(in_channels=1, num_filters=args.filters, num_classes=10).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_img = next(iter(test_loader))[0][:args.batch].to(device)
    tols = [float(t) for t in args.tolerances.split(",")]

    rows = []
    print(f"C2 MNIST field | batch {args.batch} filters {args.filters}", flush=True)
    print(f"{'state':>10} {'tol':>7} {'fwd':>6} {'bwd':>8} {'ratio':>7} {'recon':>10} {'ok':>3}", flush=True)
    for state, epochs in [("untrained", 0), (f"trained{args.train_epochs}", args.train_epochs)]:
        if epochs:
            import os
            os.environ["NODE_MAX_BATCHES"] = "200"  # bound the training cost
            for _ in range(epochs):
                train_epoch(model, train_loader, opt, crit, device)
            os.environ.pop("NODE_MAX_BATCHES", None)
        with torch.no_grad():
            h0 = model.downsampling(x_img)
        for tol in tols:
            try:
                fwd, bwd, rec = measure(model, h0, tol, device, cap=args.cap)
                ratio = bwd / fwd if fwd else float("nan")
                ok = int(rec < args.recon_thresh)
                print(f"{state:>10} {tol:7.0e} {fwd:6d} {bwd:8d} {ratio:7.2f} {rec:10.2e} {ok:3d}", flush=True)
                rows.append({"field": "mnist_conv", "state": state, "seed": args.seed, "tol": tol,
                             "fwd_nfe": fwd, "bwd_nfe": bwd, "bwd_over_fwd": ratio,
                             "recon_max": rec, "recon_ok": ok})
            except Exception as e:  # hit max_num_steps cap -> too stiff to integrate at this tol
                print(f"{state:>10} {tol:7.0e}  CAPPED ({type(e).__name__}) -> too stiff at cap {args.cap}", flush=True)
                rows.append({"field": "mnist_conv", "state": state, "seed": args.seed, "tol": tol,
                             "fwd_nfe": -1, "bwd_nfe": -1, "bwd_over_fwd": float("nan"),
                             "recon_max": float("nan"), "recon_ok": 0})
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "c2_mnist.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out}/c2_mnist.csv")


if __name__ == "__main__":
    main()
