"""C2 figure: adaptive backward/forward NFE ratio vs tolerance (both fields, trained + control).

Reads results/c2/c2_recharacterise.csv, writes figures/c2/bwd_fwd_ratio_vs_tol.png. Integrating
(recon_ok) points are solid; non-integrating (recon fails) are hollow -- the small-ratio regime
is exactly the non-faithful one. Reference lines at 1.0 (old "~=1") and 0.5 (Chen). No hand numbers.
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
    p.add_argument("--results_dir", default="results/c2")
    p.add_argument("--figures_dir", default="figures/c2")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "c2_recharacterise.csv")
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"spheres": "tab:red", "circles": "tab:blue"}
    for geom, g in df[df.epochs > 0].groupby("geometry"):
        agg = g.groupby("tol").agg(ratio=("bwd_over_fwd", "median"),
                                   ok=("recon_ok", "mean")).reset_index()
        c = colors.get(geom, "gray")
        ax.plot(agg.tol, agg.ratio, "-", color=c, label=f"{geom} (trained, median 5 seeds)")
        for _, r in agg.iterrows():
            ax.scatter(r.tol, r.ratio, s=60, color=c,
                       facecolors=c if r.ok >= 0.5 else "none", zorder=5)
    # untrained control
    for geom, g in df[df.epochs == 0].groupby("geometry"):
        ax.plot(g.tol, g.bwd_over_fwd, "--", color=colors.get(geom, "gray"), alpha=0.5,
                label=f"{geom} (untrained control)")
    ax.axhline(1.0, ls=":", c="k", lw=0.8, label='old claim "bwd ~= fwd" (=1)')
    ax.axhline(0.5, ls=":", c="gray", lw=0.8, label="Chen (bwd ~= 0.5 fwd)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()  # tighter tol to the right
    ax.set_xlabel("solver tolerance (atol=rtol)"); ax.set_ylabel("backward / forward NFE ratio")
    ax.set_title("C2 recharacterised: adaptive bwd/fwd ratio is tolerance-driven\n"
                 "(solid = integrating / recon OK; hollow = non-integrating)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "bwd_fwd_ratio_vs_tol.png", dpi=120)
    print(f"Wrote {fd}/bwd_fwd_ratio_vs_tol.png")


if __name__ == "__main__":
    main()
