"""D4 — matched-param NODE vs ANODE on CIFAR-10 (Dupont Table 1), under a compute cap.

Same faithful machinery as D3: App F.1.2 conv field + Dupont image augmentation (p zero channels
on the 3x32x32 input) + flatten->linear head (models.run_d3_anode_mnist.DupontConvODE). Matched
params per App F.2.2: NODE k=125 (~172,358) vs ANODE k=64 aug=10 (~171,799) -- ASSERTED.

Faithfulness is the risk on CIFAR (harder field stiffens more), so:
  --probe : train a few epochs and report recon at a tol ladder -> pick the recon-faithful tol
            BEFORE the headline run. A cell that can't be integrated within the step cap is DATA
            (recon_ok=false, NFE lower-bounded), never reported at loose tol as faithful.
  headline: train at --tol, per epoch log test acc/loss, fwd NFE, and recon at --eval_tol; carry
            recon_ok. NFE-vs-loss (D6) + train/test gap (D7) fall out.

HARD CAP: caller enforces the ~18 GPU-h D4 budget (this script prints elapsed wall-clock; stop it
if projected to exceed). `--check_params` prints counts and exits.
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

from data.dataloaders import get_cifar10_dataloaders
from training.engine import eval_epoch, train_epoch
from training.utils import set_seed
from scripts.run_d3_anode_mnist import DupontConvODE, n_params

SPECS = {"NODE": (0, 125), "ANODE-p10": (10, 64)}  # (augment, k) -- Dupont CIFAR Table 1
DUPONT = {"NODE": 172_358, "ANODE-p10": 171_799}


def build(spec, tol):
    aug, k = SPECS[spec]
    return DupontConvODE(augment=aug, k=k, num_classes=10, hw=32, in_channels=3, atol=tol, rtol=tol)


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
        return -1, float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--eval_tol", type=float, default=1e-5)
    p.add_argument("--eval_tols", default="", help="probe mode: ladder, e.g. 1e-4,1e-5,1e-6")
    p.add_argument("--cap", type=int, default=500_000)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--models", default="NODE,ANODE-p10")
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--results_dir", default="results/d4")
    p.add_argument("--check_params", action="store_true")
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()

    if args.check_params:
        for spec in SPECS:
            aug, k = SPECS[spec]
            print(f"{spec}: augment={aug} k={k} -> params={n_params(build(spec, args.tol)):,} "
                  f"(Dupont ~{DUPONT[spec]:,})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m for m in args.models.split(",") if m.strip()]
    import os
    if args.max_train_batches:
        os.environ["NODE_MAX_BATCHES"] = str(args.max_train_batches)
    counts = {s: n_params(build(s, args.tol)) for s in SPECS}
    print(f"param counts: {counts} | Dupont {DUPONT}", flush=True)
    assert abs(counts["NODE"] - counts["ANODE-p10"]) < 0.02 * counts["NODE"], "params not matched"

    t_start = time.perf_counter()
    tag = "probe" if args.probe else "d4"
    traj = Path(args.results_dir) / f"{tag}_trajectory.csv"
    if traj.exists():
        traj.unlink()
    ladder = [float(t) for t in args.eval_tols.split(",")] if args.eval_tols else [args.eval_tol]

    for spec in models:
        for seed in seeds:
            set_seed(seed)
            tr, te = get_cifar10_dataloaders(batch_size=args.batch_size, seed=seed)
            model = build(spec, args.tol).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            crit = nn.CrossEntropyLoss()
            x_fixed = next(iter(te))[0][:64]
            for epoch in range(args.epochs):
                t0 = time.perf_counter()
                tm = train_epoch(model, tr, opt, crit, device)
                dt = time.perf_counter() - t0
                em = eval_epoch(model, te, crit, device) if not args.probe else {"accuracy": float("nan"), "loss": float("nan")}
                for tol in ladder:
                    nfe, rel = nfe_recon_at(model, x_fixed, tol, device, args.cap)
                    ok = int(rel == rel and rel < args.recon_thresh)
                    _append(traj, {"model": spec, "params": counts[spec], "seed": seed, "epoch": epoch + 1,
                                   "train_tol": args.tol, "eval_tol": tol,
                                   "train_acc": tm["accuracy"], "train_loss": tm["loss"],
                                   "test_acc": em["accuracy"], "test_loss": em["loss"],
                                   "train_fwd_nfe": tm.get("forward_nfe_mean", float("nan")),
                                   "eval_fwd_nfe": nfe, "recon_rel": rel, "recon_ok": ok,
                                   "gap_acc": tm["accuracy"] - em["accuracy"], "epoch_s": round(dt, 1)})
                el = (time.perf_counter() - t_start) / 3600
                print(f"[{spec} s{seed} e{epoch+1}] test {em['accuracy']:.4f} | train NFE "
                      f"{tm.get('forward_nfe_mean',0):.1f} | {dt:.0f}s | elapsed {el:.2f}h", flush=True)
    print(f"Wrote {traj} | total {(time.perf_counter()-t_start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
