"""C4 — O(1)-memory-in-NFE via the adjoint (Chen 2018, Table 1 Memory column).

The coursework's Table-7 varied the *Euler baseline depth*, which does NOT test Chen's claim.
This is the corrected experiment: hold ONE conv ODE field fixed, drive its forward NFE up by
tightening the solver tolerance, and measure PEAK GPU MEMORY of a forward+backward pass under
  (a) odeint_adjoint  -- O(1) memory (recompute the state in the reverse pass), vs
  (b) plain odeint     -- direct backprop through every solver step (stores all states).

Prediction: (a) peak memory ~flat as NFE grows; (b) rises with NFE. This is an isolated
measurement (fixed field, one batch), not a training run.

Faithfulness (the analog of the flow reconstruction check): at each tolerance we verify the
forward solve actually integrates (reconstruction error small) AND that the adjoint gradient
matches the direct-backprop gradient (same computation, two memory profiles). Rows without both
are flagged. Emits raw (NFE, mem) rows; any slope summary prints alongside the raws.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torchdiffeq import odeint, odeint_adjoint

from models.continuous import ConvODEFunc
from training.utils import set_seed


def peak_mem_mb(device):
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def one_pass(func, y0, tol, adjoint, device):
    """Forward+backward one pass; return (peak_mem_mb, fwd_nfe, grad_flat)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    solver = odeint_adjoint if adjoint else odeint
    for p in func.parameters():
        p.grad = None
    func.nfe = 0
    y0 = y0.detach().requires_grad_(True)
    t = torch.tensor([0.0, 1.0], device=device)
    out = solver(func, y0, t, method="dopri5", atol=tol, rtol=tol,
                 options={"max_num_steps": 10_000_000})[1]
    fwd_nfe = func.nfe
    loss = out.pow(2).sum()
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in func.parameters()])
    return peak_mem_mb(device), fwd_nfe, grad.detach()


@torch.no_grad()
def recon_err(func, y0, tol, device):
    func.nfe = 0
    t01 = torch.tensor([0.0, 1.0], device=device)
    fwd = odeint(func, y0, t01, method="dopri5", atol=tol, rtol=tol,
                 options={"max_num_steps": 10_000_000})[1]
    back = odeint(func, fwd, torch.tensor([1.0, 0.0], device=device), method="dopri5",
                  atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    return (back - y0).norm(dim=1).flatten().abs().max().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--hw", type=int, default=28)
    p.add_argument("--tolerances", default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--recon_thresh", type=float, default=1e-2)
    p.add_argument("--grad_rtol", type=float, default=1e-2)
    p.add_argument("--results_dir", default="results/c4")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("C4 needs CUDA (peak-memory instrumentation). Aborting."); return
    device = torch.device("cuda")
    tols = [float(t) for t in args.tolerances.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    print(f"C4 on {torch.cuda.get_device_name(device)} | batch {args.batch} filters {args.filters} "
          f"{args.hw}x{args.hw} | seeds {seeds}")
    print(f"{'seed':>4} {'tol':>7} {'fwd NFE':>8} {'mem adj MB':>11} {'mem dir MB':>11} "
          f"{'grad reldiff':>13} {'recon':>10} {'faithful':>9}")
    print("-" * 82)
    for seed in seeds:
        set_seed(seed)
        # Fixed conv ODE field on an MNIST-shaped feature map (num_filters x 28 x 28).
        func = ConvODEFunc(num_channels=args.filters).to(device)
        y0 = torch.randn(args.batch, args.filters, args.hw, args.hw, device=device)
        for tol in tols:
            try:
                m_adj, nfe_adj, g_adj = one_pass(func, y0, tol, adjoint=True, device=device)
                m_dir, nfe_dir, g_dir = one_pass(func, y0, tol, adjoint=False, device=device)
            except RuntimeError as e:  # direct backprop OOM at high NFE -> that IS the finding
                print(f"{seed:4d} {tol:7.0e}  direct backprop failed: {type(e).__name__} (OOM at high NFE)")
                rows.append({"seed": seed, "tol": tol, "fwd_nfe_adj": float("nan"),
                             "mem_adjoint_mb": float("nan"), "mem_direct_mb": float("nan"),
                             "grad_reldiff": float("nan"), "recon_max": float("nan"),
                             "faithful": 0, "note": "direct_OOM"})
                torch.cuda.empty_cache()
                continue
            reldiff = (g_adj - g_dir).norm().item() / (g_dir.norm().item() + 1e-12)
            rec = recon_err(func, y0, tol, device)
            faithful = int(rec < args.recon_thresh and reldiff < args.grad_rtol)
            print(f"{seed:4d} {tol:7.0e} {nfe_adj:8d} {m_adj:11.1f} {m_dir:11.1f} {reldiff:13.2e} "
                  f"{rec:10.2e} {'YES' if faithful else 'no':>9}")
            rows.append({"seed": seed, "tol": tol, "fwd_nfe_adj": nfe_adj, "mem_adjoint_mb": m_adj,
                         "mem_direct_mb": m_dir, "grad_reldiff": reldiff, "recon_max": rec,
                         "faithful": faithful, "note": ""})

    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "c4_memory_vs_nfe.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # slope summary (printed with raws, not a bare verdict)
    good = [r for r in rows if r["faithful"]]
    if len(good) >= 2:
        nfe = np.array([r["fwd_nfe_adj"] for r in good], float)
        madj = np.array([r["mem_adjoint_mb"] for r in good], float)
        mdir = np.array([r["mem_direct_mb"] for r in good], float)
        sa = np.polyfit(nfe, madj, 1)[0]; sd = np.polyfit(nfe, mdir, 1)[0]
        print(f"\nfaithful rows: {len(good)} | mem-vs-NFE slope  adjoint {sa:+.4f} MB/NFE  "
              f"direct {sd:+.4f} MB/NFE  (ratio adj/dir = {sa/sd if sd else float('nan'):.3f})")
        print("pre-declared refutation: FALSE if adjoint slope/direct slope > 0.5, "
              "or direct slope ~ 0 (raw slopes above decide).")
    print(f"Wrote {out}/c4_memory_vs_nfe.csv")


if __name__ == "__main__":
    main()
