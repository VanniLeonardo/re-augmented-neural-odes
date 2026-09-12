"""Is the claim-5 reference converged?

Claim 5 measures endpoint error against a high-accuracy reference, dopri8 at 1e-8. On a trained
field the cost of that reference grows by 5 to 7 times per decade of tolerance, so the reference
could itself be the limit of the error axis rather than the solvers under test. This compares it
with a tighter reference, dopri8 at 1e-9, on the same field and in the norm the sweep uses, and
against the smallest error the sweep reports.
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

from scripts.run_c1_solver_dynamics import _solve
from scripts.run_c2_diagnosis import build_field, cpu_name

FIELDS_OUT = ["seed", "tol_loose", "tol_tight", "nfe_loose", "nfe_tight", "wall_loose_s",
              "wall_tight_s", "abs_diff", "rel_diff", "smallest_sweep_err", "hardware"]


def verdict(rel_diff: float, smallest: float) -> str:
    """The pre-declared condition: the reference is adequate if the two references agree far
    inside the smallest error the sweep reports."""
    if rel_diff <= 0.1 * smallest:
        return "ADEQUATE"
    if rel_diff >= smallest:
        return "REFERENCE-LIMITED"
    return "PARTIAL"


def report(path: Path, smallest: float) -> None:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        print(f"  seed {r['seed']}: dopri8 {float(r['tol_loose']):.0e} (NFE {r['nfe_loose']}, "
              f"{float(r['wall_loose_s']):.0f}s) against {float(r['tol_tight']):.0e} "
              f"(NFE {r['nfe_tight']}, {float(r['wall_tight_s']):.0f}s): relative difference "
              f"{float(r['rel_diff']):.2e}")
    worst = max(float(r["rel_diff"]) for r in rows)
    print(f"Worst {worst:.2e} against the smallest error the sweep reports, {smallest:.2e} "
          f"({worst / smallest:.1%} of it)  ->  {verdict(worst, smallest)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0")
    p.add_argument("--tol_loose", type=float, default=1e-8, help="the reference claim 5 uses")
    p.add_argument("--tol_tight", type=float, default=1e-9)
    p.add_argument("--batch", type=int, default=32, help="the eval batch of the claim-5 sweep")
    p.add_argument("--smallest_sweep_err", type=float, default=2.678e-5,
                   help="smallest rel_err in results/c1/, the yardstick")
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--mnist_epochs", type=int, default=5)
    p.add_argument("--mnist_train_tol", type=float, default=1e-3)
    p.add_argument("--ckpt_dir", default="results/c1/ckpt")
    p.add_argument("--results_dir", default="results/c1_reference")
    p.add_argument("--from_csv", action="store_true", help="print the committed result, run nothing")
    cfg = p.parse_args()
    out = Path(cfg.results_dir) / "c1_reference.csv"

    if cfg.from_csv:
        if out.exists():
            report(out, cfg.smallest_sweep_err)
        else:
            print("c1_reference: no results yet (make c1-reference)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([0.0, 1.0], device=device)
    rows = []
    print(f"C1 reference check | {device} | dopri8 {cfg.tol_loose:.0e} against "
          f"{cfg.tol_tight:.0e}", flush=True)
    for seed in [int(s) for s in cfg.seeds.split(",") if s.strip()]:
        func, h = build_field("mnist", seed, cfg.mnist_epochs, cfg.mnist_train_tol, cfg, device)
        ys, nfes, walls = {}, {}, {}
        for tol in (cfg.tol_loose, cfg.tol_tight):
            t0 = time.perf_counter()
            ys[tol], nfes[tol] = _solve(func, h, t, "dopri8", tol, None)
            walls[tol] = time.perf_counter() - t0
            print(f"  [seed {seed}] dopri8 @ {tol:.0e}: NFE {nfes[tol]}, {walls[tol]:.0f}s",
                  flush=True)
        diff = ys[cfg.tol_loose] - ys[cfg.tol_tight]
        abs_diff = diff.flatten(1).norm(dim=1).max().item()
        rel_diff = abs_diff / (ys[cfg.tol_tight].flatten(1).norm(dim=1).max().item() + 1e-12)
        rows.append({"seed": seed, "tol_loose": cfg.tol_loose, "tol_tight": cfg.tol_tight,
                     "nfe_loose": nfes[cfg.tol_loose], "nfe_tight": nfes[cfg.tol_tight],
                     "wall_loose_s": round(walls[cfg.tol_loose], 2),
                     "wall_tight_s": round(walls[cfg.tol_tight], 2), "abs_diff": abs_diff,
                     "rel_diff": rel_diff, "smallest_sweep_err": cfg.smallest_sweep_err,
                     "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else cpu_name()})
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS_OUT)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out}")
    report(out, cfg.smallest_sweep_err)


if __name__ == "__main__":
    main()
