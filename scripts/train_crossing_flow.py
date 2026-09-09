"""D8 — the 1-D crossing-flow demo (Dupont Fig 3 / Proposition 1).

A NODE flow in 1-D is order-preserving (trajectories cannot cross), so it CANNOT
represent g(x): x<0 -> +1, x>0 -> -1 -- it collapses outputs toward 0. Adding one
augmented dimension (ANODE) lets trajectories separate, and the task becomes easy.

Trains NODE (augment 0) vs ANODE (augment 1) on the 1-D crossing regression over
several seeds, writes a per-seed CSV, and (optionally) a bar figure. Cheap
(<0.5 GPU-h); also serves as the GPU acceptance-gate experiment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from torchdiffeq import odeint

from models.continuous import ODEFunc, ODEBlock
from training.logging_backend import get_logger
from training.utils import set_seed


class CrossingFlow(nn.Module):
    r"""Readout-free flow for the Dupont Fig 3 demonstration.

    The output IS the flow endpoint's data coordinate, ``h(x) = [\phi_T(x)]_0`` --
    there is NO learnable linear readout. This is essential: with a learnable head
    ``w·\phi(x)+b`` a 1-D NODE can cheat the crossing via ``w<0`` (flipping the
    monotonic flow's order), which is not what Dupont's flow argument is about. With
    the readout removed, a 1-D NODE flow is order-preserving and genuinely cannot
    represent ``x<0 -> +1, x>0 -> -1``; one augmented dimension (ANODE) can.
    """

    def __init__(self, augment_dim: int, vf_width: int, solver: str) -> None:
        super().__init__()
        self.augment_dim = augment_dim
        self.ode_dim = 1 + augment_dim
        self.ode_func = ODEFunc(in_features=self.ode_dim, hidden_dim=vf_width)
        self.ode_block = ODEBlock(self.ode_func, solver_type=solver)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.ode_func.nfe = 0
        if self.augment_dim > 0:
            zeros = torch.zeros(x.size(0), self.augment_dim, device=x.device, dtype=x.dtype)
            z = torch.cat([x, zeros], dim=1)
        else:
            z = x
        z_t = self.ode_block(z)  # (B, ode_dim)
        return z_t[:, :1]  # data coordinate = learned function h(x); no linear readout


def make_crossing(n: int, noise: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-D crossing data: x~-1 -> +1, x~+1 -> -1 (the map a 1-D flow cannot learn)."""
    rng = np.random.default_rng(seed)
    n_neg = n // 2
    x_neg = -1.0 + noise * rng.standard_normal(n_neg)
    x_pos = 1.0 + noise * rng.standard_normal(n - n_neg)
    x = np.concatenate([x_neg, x_pos])[:, None]
    y = np.concatenate([np.ones(n_neg), -np.ones(n - n_neg)])[:, None]
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


@torch.no_grad()
def eval_at_tol(model: CrossingFlow, x: torch.Tensor, y: torch.Tensor, tol: float,
                cap: int) -> Dict[str, Any]:
    """Re-evaluate a trained flow at one solver tolerance, with a reconstruction check.

    Integrate forward then backward and require the input to be recovered: at a loose
    tolerance the integrator reports a plausible NFE while not actually integrating the
    field, so an MSE/NFE measured there is an artifact. Every reported row carries
    recon_ok (see DEVIATIONS.md A4).
    """
    model.ode_block.atol = model.ode_block.rtol = tol
    z = x
    if model.augment_dim > 0:
        z = torch.cat([x, torch.zeros(x.size(0), model.augment_dim, device=x.device, dtype=x.dtype)], dim=1)
    t01 = torch.tensor([0.0, 1.0], device=x.device)
    model.ode_func.nfe = 0
    try:
        fwd = odeint(model.ode_func, z, t01, method=model.ode_block.solver_type,
                     atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        nfe = float(model.ode_func.nfe)
        back = odeint(model.ode_func, fwd, torch.flip(t01, [0]), method=model.ode_block.solver_type,
                      atol=tol, rtol=tol, options={"max_num_steps": cap})[1]
        rel = ((back - z).norm(dim=1).max() / (z.norm(dim=1).max() + 1e-9)).item()
        mse = float(nn.functional.mse_loss(fwd[:, :1], y))
    except Exception:
        return {"eval_tol": tol, "mse": float("nan"), "fwd_nfe": -1.0,
                "recon_rel": float("nan"), "recon_ok": 0, "capped": 1}
    return {"eval_tol": tol, "mse": mse, "fwd_nfe": nfe, "recon_rel": rel,
            "recon_ok": int(rel == rel and rel < 1e-2), "capped": 0}


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_one(augment_dim: int, seed: int, cfg: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    set_seed(seed)
    x, y = make_crossing(cfg.n_samples, cfg.noise, seed)
    x, y = x.to(device), y.to(device)

    model = CrossingFlow(
        augment_dim=augment_dim, vf_width=cfg.ode_hidden_dim, solver=cfg.solver
    ).to(device)
    model.ode_block.atol = model.ode_block.rtol = cfg.train_tol
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()

    model_name = "NODE" if augment_dim == 0 else f"ANODE-p{augment_dim}"
    logger = get_logger(run_name=f"crossing_{model_name}_seed{seed}", config=vars(cfg))
    final_mse, final_nfe = float("nan"), 0.0
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        pred = model(x)
        loss = lossf(pred, y)
        loss.backward()
        opt.step()
        final_mse = loss.item()
        final_nfe = float(model.ode_func.nfe)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            logger.log({"epoch": epoch, "mse": final_mse, "forward_nfe": final_nfe})
    logger.finish()

    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    rows = []
    for tol in [float(t) for t in cfg.eval_tols.split(",") if t.strip()]:
        m = eval_at_tol(model, x, y, tol, cfg.cap)
        print(f"  {model_name:9s} seed {seed} @tol {tol:.0e}: MSE {m['mse']:.4f} | "
              f"fwd NFE {m['fwd_nfe']:.0f} | recon {m['recon_rel']:.2e} ok={m['recon_ok']}")
        rows.append({"model_name": model_name, "augment_dim": augment_dim, "seed": seed,
                     "epochs": cfg.epochs, "train_tol": cfg.train_tol,
                     "train_mse_at_train_tol": final_mse, "train_nfe_at_train_tol": final_nfe,
                     **m, "hardware": hardware})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D8 1-D crossing-flow demo (NODE vs ANODE).")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ode_hidden_dim", type=int, default=16)
    p.add_argument("--solver", type=str, default="dopri5")
    p.add_argument("--train_tol", type=float, default=1e-5,
                   help="training solver tolerance (the old default 1e-3 does not integrate "
                        "these fields -- see DEVIATIONS.md A4)")
    p.add_argument("--eval_tols", type=str, default="1e-3,1e-5,1e-6,1e-7",
                   help="ladder re-evaluated after training; every row carries recon_ok")
    p.add_argument("--cap", type=int, default=500_000)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--results_dir", type=str, default="results/crossing")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    print(f"D8 crossing-flow on {device} | seeds {seeds}")

    summary_path = Path(cfg.results_dir) / "crossing_summary.csv"
    if summary_path.exists():
        summary_path.unlink()  # full re-run: every row must come from this invocation
    for augment_dim in (0, 1):
        for seed in seeds:
            for row in run_one(augment_dim, seed, cfg, device):
                _append_row(summary_path, row)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
