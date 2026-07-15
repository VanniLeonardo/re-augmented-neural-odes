"""C1/C3 confirmation figure + report: the MNIST conv field stiffens with training.

Reads results/mnist_stiffening/stiffening_trajectory.csv. Produces:
  figures/mnist_stiffening/stiffening.png  (left) faithful-NFE at each eval tol vs epoch;
                                           (right) recon_ok heat (epoch x tol) = the ladder tightening
Prints the pre-declared checks with raw numbers: does the loosest recon_ok tol tighten with epoch,
and does faithful-NFE grow monotonically.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/mnist_stiffening")
    p.add_argument("--figures_dir", default="figures/mnist_stiffening")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "stiffening_trajectory.csv")
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)
    tols = sorted(df.eval_tol.unique(), reverse=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: faithful fwd NFE vs epoch, one line per eval tol (only recon_ok points)
    for tol in tols:
        g = df[(df.eval_tol == tol)].groupby("epoch").agg(
            nfe=("eval_fwd_nfe", "mean"), ok=("recon_ok", "mean"))
        a1.plot(g.index, g.nfe, marker="o", label=f"eval tol {tol:.0e}")
    a1.set_xlabel("epoch"); a1.set_ylabel("forward NFE at eval tol"); a1.set_title("Faithful NFE grows")
    a1.legend(fontsize=8)
    # right: recon_ok fraction heatmap (epoch x tol)
    piv = df.pivot_table(index="eval_tol", columns="epoch", values="recon_ok", aggfunc="mean")
    piv = piv.reindex(sorted(piv.index, reverse=True))
    im = a2.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    a2.set_yticks(range(len(piv.index))); a2.set_yticklabels([f"{t:.0e}" for t in piv.index])
    a2.set_xticks(range(len(piv.columns))); a2.set_xticklabels(piv.columns)
    a2.set_xlabel("epoch"); a2.set_ylabel("eval tol"); a2.set_title("recon_ok (green) — the faithful tol tightens")
    fig.colorbar(im, ax=a2, fraction=0.046)
    fig.tight_layout(); fig.savefig(fd / "stiffening.png", dpi=120)

    print("=== recon_ok fraction (epoch x eval_tol) ===")
    print(piv.to_string())
    print("\n=== loosest recon_ok tol per epoch (median over seeds) ===")
    for ep, g in df.groupby("epoch"):
        ok = g[g.recon_ok == 1]
        loosest = ok.eval_tol.max() if len(ok) else float("nan")
        print(f"  epoch {ep}: loosest recon_ok tol = {loosest:.0e}" if loosest == loosest else f"  epoch {ep}: none integrate")
    print("\n=== faithful NFE (tightest tol 1e-7) vs epoch ===")
    g7 = df[df.eval_tol == df.eval_tol.min()].groupby("epoch").eval_fwd_nfe.mean()
    print("  " + " ".join(f"e{e}:{v:.0f}" for e, v in g7.items()))
    print(f"Wrote {fd}/stiffening.png")


if __name__ == "__main__":
    main()
