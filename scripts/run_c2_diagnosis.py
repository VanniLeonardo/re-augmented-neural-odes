"""C2 diagnosis — WHY does backward NFE ~ 1/2 forward NFE (Chen Fig. 3c) not reproduce?

Our C2 result is that the adaptive backward/forward NFE ratio is not 0.5 and not a single
number: in the integrating regime it runs ~13-121x, driven by tolerance and by how trained
the field is. This script asks what explains the gap to Chen's 0.5. Three candidate causes,
each an axis here:

  SOLVER FAMILY  Chen used implicit Adams (a stiff-capable, adaptive, implicit method).
                 We used dopri5, an EXPLICIT Runge-Kutta pair. The adjoint integrates an
                 augmented reverse system (state + adjoint + parameter sensitivities) that
                 is stiffer than the forward one, and explicit methods pay for stiffness in
                 step count while implicit ones do not. torchdiffeq's `implicit_adams` is
                 FIXED-step, so it is not a fair proxy; `scipy_solver` (LSODA, which switches
                 between Adams and BDF, and BDF) is the closest adaptive stiff-capable
                 analogue available, and is what we use to stand in for Chen's solver.

  ADJOINT TOLERANCE  odeint_adjoint can solve the reverse system at a different tolerance
                 from the forward pass. If the backward tolerance is looser, the ratio falls.

  FIELD STIFFNESS  a trained field is stiffer than an untrained one (our C1/C3 finding).

Every cell also checks that the gradients are actually CORRECT, against direct backprop at
high accuracy: a configuration that is cheap because it is solving the adjoint badly is not
evidence about cost. Cells failing that check, or the forward/backward reconstruction check,
are recorded and excluded from the conclusions rather than dropped.

This can overturn our own negative result. See the pre-declared REVISION TRIGGER in
OVERNIGHT_LOG.md: if a configuration faithful to Chen's description reproduces his ratio,
the paper must say the ratio is solver-specific rather than say the claim does not replicate.
"""
from __future__ import annotations

import argparse
import csv
import signal
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint, odeint_adjoint

from data.synthetic import make_spheres
from models.networks import ConvODENet, ODENet
from training.engine import train_epoch
from training.utils import set_seed

FIELDS = ["spheres", "mnist"]
FIELDS_HELP = "spheres (toy, CPU) or mnist (conv ODE-Net; reuses a C1 checkpoint if present)"

FIELDS_OUT = ["field", "seed", "trained", "solver", "tol", "adjoint_tol", "adj_offset",
              "fwd_nfe", "bwd_nfe", "ratio", "recon_rel", "recon_ok", "grad_reldiff",
              "grad_ok", "wall_s", "capped", "ref_tol", "hardware"]


def spec_to_method(spec: str):
    """'dopri5' -> (method, options); 'scipy:LSODA' -> the SciPy wrapper with that solver."""
    if spec.startswith("scipy:"):
        return "scipy_solver", {"solver": spec.split(":", 1)[1]}
    return spec, {}


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as h:
            for line in h:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


def build_field(field: str, seed: int, epochs: int, tol: float, cfg, device):
    """Return (func, y0) -- the vector field under test and a fixed input batch."""
    set_seed(seed)
    if field == "spheres":
        X, Y = make_spheres(n_samples=1200, noise=0.05, seed=seed)
        model = ODENet(data_dim=2, hidden_dim=2, num_classes=2, augment_dim=0,
                       ode_hidden_dim=32, use_stem=False, head_hidden_dim=None,
                       solver_type="dopri5", atol=tol, rtol=tol,
                       solver_options={"max_num_steps": 10_000_000}).to(device)
        if epochs:
            loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=3e-3)
            crit = nn.CrossEntropyLoss()
            for _ in range(epochs):
                train_epoch(model, loader, opt, crit, device)
        return model.ode_func, X[: cfg.batch].to(device)

    model = ConvODENet(in_channels=1, num_filters=cfg.filters, num_classes=10,
                       solver_type="dopri5").to(device)
    model.ode_block.atol = model.ode_block.rtol = tol
    ckpt = Path(cfg.ckpt_dir) / f"c1_seed{seed}_f{cfg.filters}_e{cfg.mnist_epochs}.pt"
    if epochs and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"  [seed {seed}] reusing C1 checkpoint {ckpt.name}", flush=True)
    elif epochs:
        from data.dataloaders import get_mnist_dataloaders
        tr, _ = get_mnist_dataloaders(batch_size=128, flatten=False, seed=seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for _ in range(cfg.mnist_epochs):
            train_epoch(model, tr, opt, crit, device)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
    from data.dataloaders import get_mnist_dataloaders
    _, te = get_mnist_dataloaders(batch_size=cfg.batch, flatten=False, seed=seed)
    xb = next(iter(te))[0][: cfg.batch].to(device)
    with torch.no_grad():
        h = model.downsampling(xb)
    return model.ode_func, h


def reference_grads(func, y0, t, ref_tol: float):
    """Gradients from DIRECT backprop at high accuracy -- the correctness reference.

    Direct backprop stores every intermediate state, so its memory grows with NFE (that
    is precisely what claim 8 measures). On a conv field a very tight reference therefore
    exhausts GPU memory, so we loosen the reference until it fits and RECORD the tolerance
    actually used -- the reference must still be far tighter than any tolerance under test.
    """
    tol = ref_tol
    for _ in range(3):
        try:
            func.zero_grad(set_to_none=True)
            func.nfe = 0
            out = odeint(func, y0, t, method="dopri5", rtol=tol, atol=tol,
                         options={"max_num_steps": 10_000_000})
            out[1].pow(2).sum().backward()
            g = torch.cat([p.grad.flatten() for p in func.parameters() if p.grad is not None])
            func.zero_grad(set_to_none=True)
            return g.detach().clone(), tol
        except torch.cuda.OutOfMemoryError:
            func.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            tol *= 100
            print(f"    [reference] out of memory; loosening reference to {tol:.0e}", flush=True)
    raise RuntimeError("could not compute a reference gradient within memory")


class _CellTimeout(Exception):
    pass


def one_cell(func, y0, t, spec, tol, adj_tol, ref_g, cfg) -> dict:
    """A runaway cell is DATA (capped=1), not a hung job: one configuration reached
    ~5e5 function evaluations on the toy field, which on a conv field would run for
    hours. SIGALRM turns that into a recorded row."""
    method, options = spec_to_method(spec)
    if method != "scipy_solver":
        options = dict(options, max_num_steps=10_000_000)
    row = {"solver": spec, "tol": tol, "adjoint_tol": adj_tol}
    t0 = time.perf_counter()
    if cfg.cell_timeout:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_CellTimeout()))
        signal.alarm(int(cfg.cell_timeout))
    try:
        # cost + gradient, through the adjoint
        func.zero_grad(set_to_none=True)
        func.nfe = 0
        y = y0.detach().clone().requires_grad_(True)
        out = odeint_adjoint(func, y, t, method=method, rtol=tol, atol=tol, options=options,
                             adjoint_rtol=adj_tol, adjoint_atol=adj_tol,
                             adjoint_method=method, adjoint_options=options)
        fwd = func.nfe
        out[1].pow(2).sum().backward()
        bwd = func.nfe - fwd
        g = torch.cat([p.grad.flatten() for p in func.parameters() if p.grad is not None])
        rel = ((g - ref_g).norm() / (ref_g.norm() + 1e-12)).item()
        func.zero_grad(set_to_none=True)

        # is the flow being integrated at all, at this tolerance and solver?
        with torch.no_grad():
            f1 = odeint(func, y0, t, method=method, rtol=tol, atol=tol, options=options)[1]
            b1 = odeint(func, f1, torch.flip(t, [0]), method=method, rtol=tol, atol=tol,
                        options=options)[1]
            recon = ((b1 - y0).flatten(1).norm(dim=1).max()
                     / (y0.flatten(1).norm(dim=1).max() + 1e-12)).item()
    except Exception as exc:
        print(f"    [capped] {spec} tol={tol:.0e} adj={adj_tol:.0e}: "
              f"{type(exc).__name__}: {str(exc)[:60]}", flush=True)
        row.update({"fwd_nfe": -1, "bwd_nfe": -1, "ratio": float("nan"),
                    "recon_rel": float("nan"), "recon_ok": 0, "grad_reldiff": float("nan"),
                    "grad_ok": 0, "wall_s": round(time.perf_counter() - t0, 2), "capped": 1})
        return row
    finally:
        if cfg.cell_timeout:
            signal.alarm(0)
    row.update({"fwd_nfe": fwd, "bwd_nfe": bwd, "ratio": bwd / max(fwd, 1),
                "recon_rel": recon, "recon_ok": int(recon == recon and recon < cfg.recon_thresh),
                "grad_reldiff": rel, "grad_ok": int(rel == rel and rel < cfg.grad_thresh),
                "wall_s": round(time.perf_counter() - t0, 2), "capped": 0})
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--field", default="spheres", choices=FIELDS, help=FIELDS_HELP)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=100, help="toy training epochs")
    p.add_argument("--mnist_epochs", type=int, default=5, help="matches the C1 checkpoints")
    p.add_argument("--untrained", action="store_true",
                   help="also run an untrained control of the same field")
    p.add_argument("--solvers", default="dopri5,dopri8,bosh3,adaptive_heun,scipy:LSODA,scipy:BDF,scipy:RK45")
    p.add_argument("--tols", default="1e-3,1e-5,1e-7")
    p.add_argument("--adj_offsets", default="1,100",
                   help="adjoint tol = forward tol x offset (1 = same, 100 = looser)")
    p.add_argument("--ref_tol", type=float, default=1e-9)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--ckpt_dir", default="results/c1/ckpt")
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--grad_thresh", type=float, default=1e-2)
    p.add_argument("--cell_timeout", type=float, default=900,
                   help="seconds before a cell is recorded as capped (0 disables)")
    p.add_argument("--results_dir", default="results/c2_diagnosis")
    p.add_argument("--tag", default="")
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else cpu_name()
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    solvers = [s for s in cfg.solvers.split(",") if s.strip()]
    tols = [float(x) for x in cfg.tols.split(",") if x.strip()]
    offsets = [float(x) for x in cfg.adj_offsets.split(",") if x.strip()]
    states = [True] + ([False] if cfg.untrained else [])
    t = torch.tensor([0.0, 1.0], device=device)

    out = Path(cfg.results_dir) / f"c2_diagnosis_{cfg.field}{cfg.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t_start = [], time.perf_counter()
    print(f"C2 diagnosis | field {cfg.field} | {device} | seeds {seeds} | solvers {solvers}",
          flush=True)

    for trained in states:
        for seed in seeds:
            epochs = cfg.epochs if trained else 0
            func, y0 = build_field(cfg.field, seed, epochs, min(tols), cfg, device)
            ref_g, ref_tol_used = reference_grads(func, y0, t, cfg.ref_tol)
            print(f"  [{cfg.field} seed {seed} trained={trained}] reference gradients at "
                  f"{ref_tol_used:.0e}", flush=True)
            for spec in solvers:
                for tol in tols:
                    for off in offsets:
                        r = one_cell(func, y0, t, spec, tol, tol * off, ref_g, cfg)
                        r.update({"field": cfg.field, "seed": seed, "trained": int(trained),
                                  "adj_offset": off, "ref_tol": ref_tol_used,
                                  "hardware": hardware})
                        rows.append(r)
                        print(f"    {spec:12s} tol {tol:.0e} adj x{off:<5.0f} "
                              f"fwd {r['fwd_nfe']:6} bwd {r['bwd_nfe']:8} "
                              f"ratio {r['ratio']:8.2f} recon_ok {r['recon_ok']} "
                              f"grad_ok {r['grad_ok']}", flush=True)
            with out.open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=FIELDS_OUT)
                w.writeheader()
                w.writerows(rows)
    print(f"Wrote {out} | {len(rows)} rows | {(time.perf_counter()-t_start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
