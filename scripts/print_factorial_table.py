"""Print the A1/A3 factorial table from the committed raw CSV.

Reads results/factorial/factorial_raw.csv (one row per cell x seed) and reports,
per (geometry, stem, head) cell: n seeds, #diverged, val acc mean+/-std, val loss
mean+/-std, forward-NFE median [IQR] and max (NFE is skewed, so median/IQR not
mean+/-std). Also writes an aggregated CSV alongside.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/factorial")
    args = p.parse_args()
    raw = Path(args.results_dir) / "factorial_raw.csv"
    df = pd.read_csv(raw)

    rows = []
    print(f"\nA1/A3 factorial ({len(df)} runs). NFE = forward NFE per batch.\n")
    header = (
        f"{'geometry':9s} {'stem':8s} {'head':6s} | {'n':>2s} {'div':>3s} | "
        f"{'val acc':>13s} | {'val loss':>13s} | {'NFE median[IQR]':>18s} {'NFEmax':>7s}"
    )
    print(header)
    print("-" * len(header))
    for (geom, stem, head), g in df.groupby(["geometry", "stem", "head"], sort=True):
        n = len(g)
        ndiv = int((~g["status"].astype(str).str.startswith("converged")).sum())
        acc = g["val_accuracy"].to_numpy(dtype=float)
        loss = g["val_loss"].to_numpy(dtype=float)
        nfe = g["final_fwd_nfe"].to_numpy(dtype=float)
        q1, med, q3 = np.percentile(nfe, [25, 50, 75])
        nfe_cell = f"{med:.0f} [{q1:.0f},{q3:.0f}]"
        print(
            f"{geom:9s} {stem:8s} {head:6s} | {n:2d} {ndiv:3d} | "
            f"{np.nanmean(acc):6.3f}+/-{np.nanstd(acc):5.3f} | "
            f"{np.nanmean(loss):6.3f}+/-{np.nanstd(loss):5.3f} | "
            f"{nfe_cell:>18s} {g['peak_fwd_nfe'].max():7.0f}"
        )
        rows.append({
            "geometry": geom, "stem": stem, "head": head, "n_seeds": n, "n_diverged": ndiv,
            "val_acc_mean": float(np.nanmean(acc)), "val_acc_std": float(np.nanstd(acc)),
            "val_loss_mean": float(np.nanmean(loss)), "val_loss_std": float(np.nanstd(loss)),
            "fwd_nfe_median": float(med), "fwd_nfe_q1": float(q1), "fwd_nfe_q3": float(q3),
            "fwd_nfe_peak_max": float(g["peak_fwd_nfe"].max()),
        })

    agg = Path(args.results_dir) / "factorial_aggregated.csv"
    pd.DataFrame(rows).to_csv(agg, index=False)
    print(f"\nWrote {agg}")


if __name__ == "__main__":
    main()
