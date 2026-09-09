"""C1 — solver dynamics on a trained MNIST conv ODE-Net (Chen 2018, Fig 3a-b).

Chen's claim: as the solver tolerance tightens, numerical error falls and cost rises,
with forward time roughly proportional to NFE. Fig 3a-b sweep tolerance on a *trained*
ODE-Net; this reproduces that, and extends it to the fixed-step solvers, whose analogous
cost axis is the step count rather than a tolerance.

Method (one trained model per seed -- the sweep is an EVALUATION sweep, not a retrain):
  1. train a conv ODE-Net at --train_tol, then freeze it;
  2. compute a high-accuracy REFERENCE endpoint (dopri8 at --ref_tol) for a fixed test
     feature batch -- every error below is measured against this, so "numerical error"
     means error of the integrator, not of the classifier;
  3. sweep adaptive solvers x tolerance and fixed-step solvers x step count, recording
     endpoint error, forward wall-clock (median of repeats, CUDA-synchronised), forward
     and backward NFE, and a forward->backward reconstruction check.

Every row carries recon_ok and the hardware it ran on. A row whose reconstruction fails
is DATA (the integrator is not integrating the field there) and is reported as such --
it is never used as a faithful cost or error number. See DEVIATIONS.md A4.
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
from torchdiffeq import odeint, odeint_adjoint

from data.dataloaders import get_mnist_dataloaders
from models.networks import ConvODENet
from training.engine import train_epoch
from training.utils import set_seed

ADAPTIVE = ["bosh3", "dopri5", "dopri8"]
FIXED = ["euler", "midpoint", "rk4"]

FIELDS = ["seed", "solver", "adaptive", "tol", "n_steps", "fwd_nfe", "bwd_nfe",
          "fwd_time_s", "abs_err", "rel_err", "recon_rel", "recon_ok", "capped",
          "train_tol", "train_epochs", "test_acc", "hardware"]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _opts(solver: str, n_steps: int | None) -> dict:
    """Fixed-step solvers are driven by step size, adaptive ones by tolerance."""
    return {"step_size": 1.0 / n_steps} if n_steps else {"max_num_steps": 10_000_000}


@torch.no_grad()
def _solve(func, h, t, solver, tol, n_steps):
    func.nfe = 0
    out = odeint(func, h, t, method=solver, atol=tol, rtol=tol, options=_opts(solver, n_steps))
    return out[1], func.nfe


def _timed_forward(func, h, t, solver, tol, n_steps, repeats, device) -> float:
    """Median wall-clock of the forward solve. Chen Fig 3b plots time against NFE, so
    the timing must be the solve alone and CUDA work must be synchronised."""
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _solve(func, h, t, solver, tol, n_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _backward_nfe(func, h, t, solver, tol, n_steps) -> int:
    """NFE spent in the adjoint backward solve (total minus forward)."""
    func.nfe = 0
    h_ = h.detach().clone().requires_grad_(True)
    out = odeint_adjoint(func, h_, t, method=solver, atol=tol, rtol=tol,
                         options=_opts(solver, n_steps))
    fwd = func.nfe
    out[1].sum().backward()
    return int(func.nfe - fwd)


def sweep_one(func, h, ref, t, solver, tol, n_steps, cfg, device) -> dict:
    """One (solver, cost-setting) cell: error vs reference, cost, and a recon check."""
    try:
        y, fwd_nfe = _solve(func, h, t, solver, tol, n_steps)
        back, _ = _solve(func, y, torch.flip(t, [0]), solver, tol, n_steps)
        recon = ((back - h).flatten(1).norm(dim=1).max()
                 / (h.flatten(1).norm(dim=1).max() + 1e-9)).item()
        abs_err = (y - ref).flatten(1).norm(dim=1).max().item()
        rel_err = abs_err / (ref.flatten(1).norm(dim=1).max().item() + 1e-12)
        secs = _timed_forward(func, h, t, solver, tol, n_steps, cfg.repeats, device)
        bwd_nfe = _backward_nfe(func, h, t, solver, tol, n_steps)
        capped = 0
    except Exception as exc:  # step cap / non-convergence is DATA, not a crash
        print(f"    [capped] {solver} tol={tol} steps={n_steps}: {type(exc).__name__}", flush=True)
        return {"solver": solver, "tol": tol, "n_steps": n_steps or "", "fwd_nfe": -1,
                "bwd_nfe": -1, "fwd_time_s": float("nan"), "abs_err": float("nan"),
                "rel_err": float("nan"), "recon_rel": float("nan"), "recon_ok": 0, "capped": 1}
    return {"solver": solver, "tol": tol, "n_steps": n_steps or "", "fwd_nfe": fwd_nfe,
            "bwd_nfe": bwd_nfe, "fwd_time_s": secs, "abs_err": abs_err, "rel_err": rel_err,
            "recon_rel": recon, "recon_ok": int(recon == recon and recon < cfg.recon_thresh),
            "capped": capped}


def run_seed(seed: int, cfg, device) -> list[dict]:
    set_seed(seed)
    train_loader, test_loader = get_mnist_dataloaders(batch_size=cfg.batch_size,
                                                      flatten=False, seed=seed)
    model = ConvODENet(in_channels=1, num_filters=cfg.filters, num_classes=10,
                       solver_type="dopri5").to(device)
    model.ode_block.atol = model.ode_block.rtol = cfg.train_tol

    # Train once per seed and cache. The sweep is an EVALUATION sweep, so re-running or
    # re-tuning it must not cost another full training run (and a reviewer re-making the
    # figure should not have to retrain either).
    ckpt = Path(cfg.ckpt_dir) / f"c1_seed{seed}_f{cfg.filters}_e{cfg.epochs}.pt"
    if ckpt.exists() and not cfg.retrain:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  [seed {seed}] loaded cached model {ckpt.name}", flush=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        crit = nn.CrossEntropyLoss()
        for ep in range(cfg.epochs):
            m = train_epoch(model, train_loader, opt, crit, device)
            print(f"  [seed {seed}] epoch {ep+1}/{cfg.epochs} train_acc {m['accuracy']:.4f}",
                  flush=True)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"  [seed {seed}] cached model -> {ckpt}", flush=True)

    model.eval()
    xb, yb = next(iter(test_loader))
    xb = xb[: cfg.eval_batch].to(device)
    with torch.no_grad():
        logits = model(xb)
        test_acc = float((logits.argmax(1).cpu() == yb[: cfg.eval_batch]).float().mean())
        h = model.downsampling(xb)
    t = torch.tensor([0.0, 1.0], device=device)

    # Reference endpoint: the most accurate integration we can afford. Everything below
    # is measured against this, so the error axis is the integrator's, not the model's.
    if cfg.ref_bench:
        print(f"  [seed {seed}] reference benchmark (dopri8, batch {cfg.eval_batch}):", flush=True)
        for rt in [float(x) for x in cfg.ref_bench.split(",") if x.strip()]:
            tb = time.perf_counter()
            with torch.no_grad():
                _, nfe_b = _solve(model.ode_func, h, t, "dopri8", rt, None)
            print(f"    tol {rt:.0e}: NFE {nfe_b:7d} | {time.perf_counter()-tb:8.1f}s", flush=True)
        return []

    t_ref = time.perf_counter()
    with torch.no_grad():
        ref, ref_nfe = _solve(model.ode_func, h, t, "dopri8", cfg.ref_tol, None)
    print(f"  [seed {seed}] reference dopri8 @ {cfg.ref_tol:.0e}: NFE {ref_nfe} | "
          f"{time.perf_counter()-t_ref:.1f}s | test_acc {test_acc:.4f}", flush=True)

    tols = [float(x) for x in cfg.tols.split(",") if x.strip()]
    steps = [int(x) for x in cfg.steps.split(",") if x.strip()]
    rows = []
    for solver in ADAPTIVE:
        for tol in tols:
            r = sweep_one(model.ode_func, h, ref, t, solver, tol, None, cfg, device)
            rows.append(r)
            print(f"    {solver:8s} tol {tol:.0e}: NFE {r['fwd_nfe']:5} bwd {r['bwd_nfe']:6} "
                  f"| {r['fwd_time_s']*1e3:8.2f} ms | rel_err {r['rel_err']:.3e} "
                  f"| recon_ok {r['recon_ok']}", flush=True)
    for solver in FIXED:
        for n in steps:
            r = sweep_one(model.ode_func, h, ref, t, solver, 0.0, n, cfg, device)
            rows.append(r)
            print(f"    {solver:8s} steps {n:4d}: NFE {r['fwd_nfe']:5} bwd {r['bwd_nfe']:6} "
                  f"| {r['fwd_time_s']*1e3:8.2f} ms | rel_err {r['rel_err']:.3e} "
                  f"| recon_ok {r['recon_ok']}", flush=True)

    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    for r in rows:
        r.update({"seed": seed, "adaptive": int(r["solver"] in ADAPTIVE),
                  "train_tol": cfg.train_tol, "train_epochs": cfg.epochs,
                  "test_acc": test_acc, "hardware": hardware})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="C1 solver dynamics (Chen Fig 3a-b).")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch", type=int, default=32)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_tol", type=float, default=1e-3)
    p.add_argument("--ref_tol", type=float, default=1e-8,
                   help="reference integration; 2+ orders tighter than the tightest "
                        "swept tol, so the error axis is not reference-limited")
    p.add_argument("--tols", default="1e-1,1e-2,1e-3,1e-4,1e-5")
    p.add_argument("--steps", default="1,2,4,8,16,32,64,128")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/c1")
    p.add_argument("--ckpt_dir", default="results/c1/ckpt",
                   help="trained models are cached here so the evaluation sweep can be "
                        "re-run without retraining (gitignored)")
    p.add_argument("--retrain", action="store_true", help="ignore any cached model")
    p.add_argument("--ref_bench", default="",
                   help="time the dopri8 reference at these tols and exit (design probe)")
    p.add_argument("--tag", default="", help="suffix for per-seed shards (HPC array jobs)")
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    print(f"C1 solver dynamics on {device} | seeds {seeds}", flush=True)

    t0 = time.perf_counter()
    rows: list[dict] = []
    for seed in seeds:
        rows += run_seed(seed, cfg, device)
    name = f"c1_solver_dynamics{cfg.tag}.csv"
    _write(Path(cfg.results_dir) / name, rows)
    print(f"Wrote {Path(cfg.results_dir) / name} | {len(rows)} rows | "
          f"{(time.perf_counter()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
