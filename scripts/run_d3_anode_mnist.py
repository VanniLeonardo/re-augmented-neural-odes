"""D3 — matched-param NODE vs ANODE on MNIST (Dupont Table 1 / App F.1.2, F.2.2).

Faithful to the paper's DESCRIPTION (not their code): the ODE field is the App F.1.2 conv block
(1x1 k filters -> 3x3 k filters -> 1x1 c filters, a time channel concatenated before each conv --
our models.continuous.ConvODEFunc). Dupont augments the INPUT IMAGE with p zero channels
(R^{c x h x w} -> R^{(c+p) x h x w}); NO Chen-style downsampling. Classifier = flatten -> linear.

Matched param budgets (App F.2.2): NODE k=92 (~84,395) vs ANODE k=64, aug=5 (~84,816). We ASSERT
our counts are matched to each other and REPORT them against Dupont's targets (a small offset from
a head/padding detail is a documented deviation, not tuned).

Logs per epoch: train/test acc+loss, forward NFE (train-tol AND recon-checked eval-tol), a per-
epoch reconstruction check (recon_ok), and NFE-vs-loss points (D6) + train/test gap (D7). Every
row carries recon_ok. Single invocation over seeds. `--check_params` just prints counts and exits.
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
from models.continuous import ConvODEFunc, ODEBlock
from training.engine import eval_epoch, train_epoch
from training.utils import set_seed


class DupontConvODE(nn.Module):
    """Dupont image (A)NODE: augment input image with p zero channels, integrate the App F.1.2
    conv field, flatten, linear-classify. augment=0 => NODE."""

    def __init__(self, augment: int, k: int, num_classes: int = 10, hw: int = 28,
                 in_channels: int = 1, solver: str = "dopri5", atol: float = 1e-3, rtol: float = 1e-3):
        super().__init__()
        self.augment = augment
        c = in_channels + augment
        self.ode_func = ConvODEFunc(num_channels=c, hidden_channels=k)  # 1x1 k -> 3x3 k -> 1x1 c
        self.ode_block = ODEBlock(self.ode_func, solver_type=solver, atol=atol, rtol=rtol,
                                  options={"max_num_steps": 10_000_000})
        self.fc = nn.Linear(c * hw * hw, num_classes)

    def forward(self, x):
        self.ode_func.nfe = 0
        if self.augment > 0:
            z = torch.zeros(x.size(0), self.augment, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, z], dim=1)
        h = self.ode_block(x)
        return self.fc(h.flatten(1))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(spec, tol):
    aug, k = {"NODE": (0, 92), "ANODE-p5": (5, 64)}[spec]
    return DupontConvODE(augment=aug, k=k, atol=tol, rtol=tol)


@torch.no_grad()
def recon_check(model, x_img, tol, device):
    x = x_img.to(device)
    if model.augment > 0:
        x = torch.cat([x, torch.zeros(x.size(0), model.augment, x.size(2), x.size(3), device=device)], dim=1)
    model.ode_func.nfe = 0
    f = odeint(model.ode_func, x, torch.tensor([0.0, 1.0], device=device), method="dopri5",
               atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    nfe = model.ode_func.nfe
    b = odeint(model.ode_func, f, torch.tensor([1.0, 0.0], device=device), method="dopri5",
               atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    rel = ((b - x).flatten(1).norm(dim=1).max() / (x.flatten(1).norm(dim=1).max() + 1e-9)).item()
    return rel, nfe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--eval_tol", type=float, default=1e-5)
    p.add_argument("--eval_tols", default="",
                   help="tolerance ladder; INCLUDE the train tol (1e-3) so the tolerance the "
                        "accuracy is measured at is itself recon-checked")
    p.add_argument("--tag", default="", help="suffix for per-seed shards (HPC array jobs)")
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/d3")
    p.add_argument("--models", default="NODE,ANODE-p5")
    p.add_argument("--max_train_batches", type=int, default=0, help="0=full; else cap (probe)")
    p.add_argument("--check_params", action="store_true")
    args = p.parse_args()

    if args.check_params:
        for spec in ("NODE", "ANODE-p5"):
            m = build(spec, args.tol)
            aug, k = {"NODE": (0, 92), "ANODE-p5": (5, 64)}[spec]
            print(f"{spec}: augment={aug} k={k} -> params={n_params(m):,}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    traj = Path(args.results_dir) / f"d3_trajectory{args.tag}.csv"
    ladder = [float(t) for t in args.eval_tols.split(",") if t.strip()] or [args.eval_tol]
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if traj.exists():
        traj.unlink()
    # assert matched params once
    counts = {s: n_params(build(s, args.tol)) for s in ("NODE", "ANODE-p5")}
    print(f"param counts: {counts} | Dupont targets NODE 84,395 / ANODE 84,816", flush=True)
    assert abs(counts["NODE"] - counts["ANODE-p5"]) < 0.02 * counts["NODE"], "params not matched"

    import os
    if args.max_train_batches:
        os.environ["NODE_MAX_BATCHES"] = str(args.max_train_batches)
    models = [m for m in args.models.split(",") if m.strip()]
    for spec in models:
        for seed in seeds:
            set_seed(seed)
            tr, te = get_mnist_dataloaders(batch_size=args.batch_size, flatten=False, seed=seed)
            model = build(spec, args.tol).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            crit = nn.CrossEntropyLoss()
            x_fixed = next(iter(te))[0][:64]
            for epoch in range(args.epochs):
                t0 = time.perf_counter()
                tm = train_epoch(model, tr, opt, crit, device)
                dt = time.perf_counter() - t0
                em = eval_epoch(model, te, crit, device)
                is_last = (epoch + 1 == args.epochs)
                for etol in ladder:
                    rel, rnfe = recon_check(model, x_fixed, etol, device)
                    # Re-measure accuracy at this tolerance, at the final epoch only. test_acc comes
                    # from the training tolerance, and a full test pass at 1e-7 is
                    # far more expensive than one at 1e-3.
                    if is_last:
                        keep = (model.ode_block.atol, model.ode_block.rtol)
                        model.ode_block.atol = model.ode_block.rtol = etol
                        em_t = eval_epoch(model, te, crit, device)
                        model.ode_block.atol, model.ode_block.rtol = keep
                    else:
                        em_t = {"accuracy": float("nan"), "loss": float("nan")}
                    row = {"model": spec, "params": counts[spec], "seed": seed, "epoch": epoch + 1,
                           "tol": args.tol, "eval_tol": etol,
                           "train_acc": tm["accuracy"], "train_loss": tm["loss"],
                           "test_acc": em["accuracy"], "test_loss": em["loss"],
                           "train_fwd_nfe": tm.get("forward_nfe_mean", float("nan")),
                           "faithful_fwd_nfe": rnfe, "recon_rel": rel,
                           "recon_ok": int(rel < args.recon_thresh),
                           "gap_acc": tm["accuracy"] - em["accuracy"], "epoch_s": round(dt, 1),
                           "test_acc_at_tol": em_t["accuracy"],
                           "test_loss_at_tol": em_t["loss"], "hardware": hardware}
                    _append(traj, row)
                print(f"[{spec} s{seed} e{epoch+1}] test {em['accuracy']:.4f} loss {em['loss']:.3f} "
                      f"| NFE {tm.get('forward_nfe_mean', float('nan')):.1f} | {dt:.0f}s", flush=True)
    print(f"Wrote {traj}", flush=True)


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
