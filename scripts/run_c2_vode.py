"""C2: the original's solver, SciPy VODE implicit Adams, on our fields.

Chen et al. integrated with VODE's implicit Adams method through `scipy.integrate.ode`, and
their adjoint solved the reverse system with the same call and the same tolerance
(`src/integrate.py`, shared by the authors). `torchdiffeq` cannot drive VODE, so this script
implements that adjoint directly: the augmented reverse system [y, adj_y, adj_params] is
integrated backwards with VODE at the forward tolerance, with the field evaluated in torch.

Counting follows the rest of this repository and the original: forward NFE is the number of
field evaluations in the forward solve, and backward NFE the number of evaluations of the
augmented system, each of which is one field evaluation plus one vector-Jacobian product.

Every row carries the forward-backward reconstruction check, run with the same solver and
tolerance, and a gradient check against direct backpropagation. A cell that is cheap because it
solves the adjoint badly is not evidence about cost, so both checks gate the conclusions.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.integrate import ode

from scripts.run_c2_diagnosis import FIELDS, build_field, cpu_name, reference_grads

FIELDS_OUT = ["field", "seed", "trained", "solver", "tol", "adjoint_tol", "fwd_nfe", "bwd_nfe",
              "ratio", "recon_rel", "recon_ok", "grad_reldiff", "grad_ok", "wall_s", "capped",
              "ref_tol", "hardware"]


def _solve(f, y0: np.ndarray, t0: float, t1: float, tol: float, nsteps: int) -> np.ndarray:
    """One VODE implicit-Adams solve from t0 to t1. Backwards when t1 < t0."""
    r = ode(f).set_integrator("vode", method="adams", atol=tol, rtol=tol, nsteps=nsteps)
    r.set_initial_value(y0, t0)
    y = r.integrate(t1)
    if not r.successful():
        raise RuntimeError(f"vode failed between t={t0} and t={t1}")
    return y


def _field_fn(func, shape, device):
    """The field itself, as SciPy wants it: float64 numpy in and out."""
    def f(t, y):
        with torch.no_grad():
            out = func(float(t), torch.as_tensor(y, dtype=torch.float32, device=device).view(shape))
        return out.reshape(-1).double().cpu().numpy()
    return f


def _augmented_fn(func, shape, device, params, D):
    """[y, adj_y, adj_params]' = [f, -adj_y^T df/dy, -adj_y^T df/dtheta], the reverse system."""
    def f(t, s):
        y = torch.as_tensor(s[:D], dtype=torch.float32, device=device).view(shape).requires_grad_(True)
        adj = torch.as_tensor(s[D:2 * D], dtype=torch.float32, device=device).view(shape)
        with torch.enable_grad():
            out = func(float(t), y)
            grads = torch.autograd.grad(out, (y, *params), -adj, allow_unused=True)
        dtheta = [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                  for g, p in zip(grads[1:], params)]
        return torch.cat([out.detach().reshape(-1), grads[0].reshape(-1), *dtheta]).double().cpu().numpy()
    return f


def vode_adjoint(func, y0: torch.Tensor, tol: float, nsteps: int):
    """Forward and reverse VODE solves for the loss sum(y(1)^2). Returns the parameter
    gradient, the two evaluation counts, and the reconstruction error."""
    device, shape, D = y0.device, y0.shape, y0.numel()
    params = [p for p in func.parameters() if p.requires_grad]
    field, augmented = _field_fn(func, shape, device), _augmented_fn(func, shape, device, params, D)
    y0_flat = y0.reshape(-1).double().cpu().numpy()

    func.nfe = 0
    y1 = _solve(field, y0_flat, 0.0, 1.0, tol, nsteps)
    fwd = func.nfe

    y1_t = torch.as_tensor(y1, dtype=torch.float32, device=device).view(shape)
    start = np.concatenate([y1, (2 * y1_t).reshape(-1).double().cpu().numpy(),
                            np.zeros(sum(p.numel() for p in params))])
    func.nfe = 0
    end = _solve(augmented, start, 1.0, 0.0, tol, nsteps)
    bwd = func.nfe
    grad = torch.as_tensor(end[2 * D:], dtype=torch.float32, device=device)

    back = torch.as_tensor(_solve(field, y1, 1.0, 0.0, tol, nsteps),
                           dtype=torch.float32, device=device).view(shape)
    recon = ((back - y0).flatten(1).norm(dim=1).max()
             / (y0.flatten(1).norm(dim=1).max() + 1e-12)).item()
    return grad, fwd, bwd, recon


class _CellTimeout(Exception):
    pass


def one_cell(func, y0, tol, ref_g, cfg) -> dict:
    """A runaway cell is DATA (capped=1), not a hung job: the original's solver may simply be
    unable to integrate our field within a reasonable time, which is itself a result."""
    row = {"solver": "vode:adams", "tol": tol, "adjoint_tol": tol}
    t0 = time.perf_counter()
    if cfg.cell_timeout:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_CellTimeout()))
        signal.alarm(int(cfg.cell_timeout))
    try:
        grad, fwd, bwd, recon = vode_adjoint(func, y0, tol, cfg.nsteps)
        assert grad.numel() == ref_g.numel(), (grad.numel(), ref_g.numel())
        rel = ((grad - ref_g).norm() / (ref_g.norm() + 1e-12)).item()
    except Exception as exc:
        print(f"    [capped] vode:adams tol={tol:.0e}: {type(exc).__name__}: {str(exc)[:70]}",
              flush=True)
        row.update({"fwd_nfe": -1, "bwd_nfe": -1, "ratio": float("nan"), "recon_rel": float("nan"),
                    "recon_ok": 0, "grad_reldiff": float("nan"), "grad_ok": 0,
                    "wall_s": round(time.perf_counter() - t0, 2), "capped": 1})
        return row
    finally:
        if cfg.cell_timeout:
            signal.alarm(0)
    row.update({"fwd_nfe": fwd, "bwd_nfe": bwd, "ratio": bwd / max(fwd, 1), "recon_rel": recon,
                "recon_ok": int(recon == recon and recon < cfg.recon_thresh), "grad_reldiff": rel,
                "grad_ok": int(rel == rel and rel < cfg.grad_thresh),
                "wall_s": round(time.perf_counter() - t0, 2), "capped": 0})
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="mnist", choices=FIELDS)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--tols", default="1e-5,1e-3")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ref_tol", type=float, default=1e-7)
    p.add_argument("--nsteps", type=int, default=10_000_000, help="VODE steps per call")
    p.add_argument("--cell_timeout", type=float, default=1800)
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--grad_thresh", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=100, help="toy training epochs")
    p.add_argument("--train_tol", type=float, default=1e-6, help="2-D training tolerance")
    p.add_argument("--mnist_epochs", type=int, default=5, help="matches the C1 checkpoints")
    p.add_argument("--mnist_train_tol", type=float, default=1e-3, help="matches the C1 checkpoints")
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--ckpt_dir", default="results/c1/ckpt")
    p.add_argument("--results_dir", default="results/c2_vode")
    p.add_argument("--tag", default="")
    cfg = p.parse_args()

    device = torch.device("cuda" if cfg.field == "mnist" and torch.cuda.is_available() else "cpu")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else cpu_name()
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    tols = [float(t) for t in cfg.tols.split(",") if t.strip()]
    t = torch.tensor([0.0, 1.0], device=device)
    out = Path(cfg.results_dir) / f"c2_vode_{cfg.field}{cfg.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"C2 VODE | field {cfg.field} | {device} | seeds {seeds} | tols {tols}", flush=True)

    for seed in seeds:
        func, y0 = build_field(cfg.field, seed, cfg.epochs, cfg.train_tol, cfg, device)
        ref_g, ref_tol_used = reference_grads(func, y0, t, cfg.ref_tol)
        print(f"  [{cfg.field} seed {seed}] reference gradients at {ref_tol_used:.0e}", flush=True)
        for tol in tols:
            r = one_cell(func, y0, tol, ref_g, cfg)
            r.update({"field": cfg.field, "seed": seed, "trained": 1, "ref_tol": ref_tol_used,
                      "hardware": hardware})
            rows.append(r)
            print(f"    vode:adams   tol {tol:.0e}   fwd {r['fwd_nfe']:6} bwd {r['bwd_nfe']:8} "
                  f"ratio {r['ratio']:8.2f} recon_ok {r['recon_ok']} grad_ok {r['grad_ok']} "
                  f"{r['wall_s']:.0f}s", flush=True)
        with out.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=FIELDS_OUT)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out} | {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
