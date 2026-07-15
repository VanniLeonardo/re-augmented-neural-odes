"""C2 recharacterisation: adaptive backward/forward NFE ratio vs tolerance x field x seed.

Tonight's committed finding withdrew the "backward ~ forward (ratio ~1)" C2 headline: at
accurate tolerance the adjoint's reverse (augmented, stiffer, higher-dim) system costs far more
than the forward. This script characterises that fully -- >=5 seeds, both toy fields, a tolerance
axis, with a reconstruction check per tolerance so the ratio is only trusted where the solver
integrates. Also measures an UNTRAINED control (epochs=0) to separate a pure-solver effect from
training-induced stiffness. FRAMING (headline/secondary/cut) is left to the awake review; this
only produces the table. Emits raw per-row numbers; any summary is printed alongside the raws.
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

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import set_seed

DEVICE = torch.device("cpu")
_N = {"spheres": 1200, "circles": 1000}


def measure(model, Xval, tol):
    """Return (fwd_nfe, bwd_nfe, recon_max) at a given eval tolerance.

    One forward+backward through the adjoint, snapshotting the NFE split exactly like the
    training engine (fwd = counter after forward; bwd = counter after backward - fwd)."""
    model.ode_block.atol = tol
    model.ode_block.rtol = tol
    xb = Xval[:128].clone().requires_grad_(True)
    model.ode_func.nfe = 0
    logits = model(xb)
    fwd = model.ode_func.nfe
    logits.sum().backward()
    bwd = model.ode_func.nfe - fwd
    with torch.no_grad():
        f = odeint(model.ode_func, Xval, torch.tensor([0.0, 1.0]), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
        b = odeint(model.ode_func, f, torch.tensor([1.0, 0.0]), method="dopri5",
                   atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
        recon = (b - Xval).norm(dim=1).max().item()
    return fwd, bwd, recon


def build_and_train(geometry, seed, epochs, train_tol):
    set_seed(seed)
    tr, va = get_dataloaders(geometry, n_samples=_N[geometry], batch_size=64,
                             val_split=0.2, noise=0.05, seed=seed)
    model = ODENet(data_dim=2, hidden_dim=2, num_classes=2, augment_dim=0,
                   ode_hidden_dim=32, use_stem=False, head_hidden_dim=None,
                   solver_type="dopri5", atol=train_tol, rtol=train_tol,
                   solver_options={"max_num_steps": 10_000_000}).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_epoch(model, tr, opt, crit, DEVICE)
    Xval = torch.stack([va.dataset[i][0] for i in range(len(va.dataset))])
    return model, Xval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometries", default="spheres,circles")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train_tol", type=float, default=1e-6)
    p.add_argument("--tolerances", default="1e-3,1e-4,1e-5,1e-6,1e-7")
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/c2")
    args = p.parse_args()
    geoms = [g for g in args.geometries.split(",") if g]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tols = [float(t) for t in args.tolerances.split(",")]
    out = Path(args.results_dir) / "c2_recharacterise.csv"
    if out.exists():
        out.unlink()

    rows = []
    # untrained control (seed 0, both geometries)
    tasks = [(g, 0, 0) for g in geoms] + [(g, s, args.epochs) for g in geoms for s in seeds]
    for geom, seed, epochs in tasks:
        model, Xval = build_and_train(geom, seed, epochs, args.train_tol)
        for tol in tols:
            fwd, bwd, recon = measure(model, Xval, tol)
            ratio = bwd / fwd if fwd else float("nan")
            integ = int(recon < args.recon_thresh)
            row = {"geometry": geom, "seed": seed, "epochs": epochs, "tol": tol,
                   "fwd_nfe": fwd, "bwd_nfe": bwd, "bwd_over_fwd": ratio,
                   "recon_max": recon, "recon_ok": integ}
            rows.append(row)
            print(f"[{geom[:3]} s{seed} e{epochs} tol {tol:.0e}] fwd {fwd:5d} bwd {bwd:7d} "
                  f"ratio {ratio:7.2f} recon {recon:.1e} {'OK' if integ else 'loose'}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # summary printed ALONGSIDE raws (never a bare verdict): integrating-regime ratio per field
    print("\n--- integrating-regime (recon_ok, trained) bwd/fwd ratio, median[min,max] over seeds/tol ---")
    import statistics
    for geom in geoms:
        rr = [r["bwd_over_fwd"] for r in rows
              if r["geometry"] == geom and r["recon_ok"] and r["epochs"] > 0]
        if rr:
            print(f"  {geom}: n={len(rr)} median {statistics.median(rr):.1f} "
                  f"[{min(rr):.1f}, {max(rr):.1f}]  (pre-declared refutation band for 'ratio~1' = [0.5,2.0])")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
