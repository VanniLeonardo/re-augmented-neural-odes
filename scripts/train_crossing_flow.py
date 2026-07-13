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
    print(f"  {model_name:9s} seed {seed}: final MSE {final_mse:.4f} | fwd NFE {final_nfe:.0f}")
    return {
        "model_name": model_name,
        "augment_dim": augment_dim,
        "seed": seed,
        "epochs": cfg.epochs,
        "final_mse": final_mse,
        "final_forward_nfe": final_nfe,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D8 1-D crossing-flow demo (NODE vs ANODE).")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ode_hidden_dim", type=int, default=16)
    p.add_argument("--solver", type=str, default="dopri5")
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
    for augment_dim in (0, 1):
        for seed in seeds:
            _append_row(summary_path, run_one(augment_dim, seed, cfg, device))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
