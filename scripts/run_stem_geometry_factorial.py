"""A1/A3 factorial: which deviation makes the coursework NODE not fail?

Runs {geometry: circles, spheres} x {stem, no-stem} x {head: linear, mlp} x seeds,
measuring the RIGHT quantity for a NODE-failure claim: forward/backward NFE
(final + peak), NFE variance across seeds, val loss AND accuracy -- not accuracy
alone. Writes one row per (cell, seed) to a committed CSV; aggregate/print with
scripts/print_factorial_table.py.

The adaptive solver is capped with max_num_steps (Dupont Fig 13: NODE NFE can exceed
1000 and become unstable). A NODE that must tear the space either (a) drives NFE up
until it hits the cap and DIVERGES, or (b) fails to separate. Both are recorded, so
the "NODE failed" signature is visible without an unbounded hang. Each run also has a
wall-clock budget so a slow-but-not-diverged cell cannot stall the sweep.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import eval_epoch, train_epoch
from training.utils import set_seed

# geometry -> (n_samples, noise)
_GEOM = {"circles": (1000, 0.05), "spheres": (1200, 0.05)}


def _append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_cell(geometry: str, use_stem: bool, head: str, seed: int, cfg, device) -> Dict[str, Any]:
    set_seed(seed)
    n_samples, noise = _GEOM[geometry]
    tr, va = get_dataloaders(
        geometry, n_samples=n_samples, batch_size=64, val_split=0.2, noise=noise, seed=seed
    )
    model = ODENet(
        data_dim=2, hidden_dim=2, num_classes=2, augment_dim=0,
        ode_hidden_dim=32, use_stem=use_stem,
        head_hidden_dim=(None if head == "linear" else 32),
        solver_options={"max_num_steps": cfg.max_num_steps},
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    status = "converged"
    peak_fwd_nfe = 0.0
    final: Dict[str, float] = {}
    val: Dict[str, float] = {}
    t0 = time.perf_counter()
    completed = 0
    for epoch in range(cfg.epochs):
        try:
            final = train_epoch(model, tr, opt, crit, device)
        except Exception as ex:  # solver hit max_num_steps -> NODE-failure signature
            status = f"diverged:{type(ex).__name__}"
            break
        peak_fwd_nfe = max(peak_fwd_nfe, final.get("forward_nfe_mean", 0.0))
        val = eval_epoch(model, va, crit, device)
        completed = epoch + 1
        if time.perf_counter() - t0 > cfg.time_budget_s:
            status = "time_capped"
            break

    tag = ("stem" if use_stem else "no-stem")
    print(
        f"  {geometry:8s} {tag:8s} head={head:6s} seed {seed}: "
        f"acc {val.get('accuracy', float('nan')):.3f} | loss {val.get('loss', float('nan')):.3f} | "
        f"fwd NFE final {final.get('forward_nfe_mean', 0):.0f} peak {peak_fwd_nfe:.0f} | "
        f"{status} @ ep {completed} | {time.perf_counter()-t0:.0f}s",
        flush=True,
    )
    return {
        "geometry": geometry,
        "stem": tag,
        "head": head,
        "seed": seed,
        "epochs_completed": completed,
        "status": status,
        "val_accuracy": val.get("accuracy", float("nan")),
        "val_loss": val.get("loss", float("nan")),
        "final_fwd_nfe": final.get("forward_nfe_mean", 0.0),
        "final_bwd_nfe": final.get("backward_nfe_mean", 0.0),
        "peak_fwd_nfe": peak_fwd_nfe,
        "max_num_steps": cfg.max_num_steps,
    }


def parse_args():
    p = argparse.ArgumentParser(description="A1/A3 stem x geometry x head factorial.")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--max_num_steps", type=int, default=2000)
    p.add_argument("--time_budget_s", type=float, default=90.0)
    p.add_argument("--results_dir", type=str, default="results/factorial")
    p.add_argument("--with_mlp_head", action="store_true", help="Add the MLP-head diagnostic contrast.")
    return p.parse_args()


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    out = Path(cfg.results_dir) / "factorial_raw.csv"
    if out.exists():
        out.unlink()
    print(f"Factorial on {device} | seeds {seeds} | max_num_steps {cfg.max_num_steps}", flush=True)

    # Main factorial: geometry x stem, linear head (the faithful setup).
    for geometry in ("circles", "spheres"):
        for use_stem in (True, False):
            for seed in seeds:
                _append(out, run_cell(geometry, use_stem, "linear", seed, cfg, device))
    # Diagnostic: does an MLP head trivialise the hardest cell (spheres, no-stem)?
    if cfg.with_mlp_head:
        for seed in seeds:
            _append(out, run_cell("spheres", False, "mlp", seed, cfg, device))

    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
