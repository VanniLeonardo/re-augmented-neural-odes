"""C4 figure: peak memory vs forward NFE, adjoint (O(1)) vs direct backprop.

Reads results/c4/c4_memory_vs_nfe.csv (faithful rows only) and writes
figures/c4/memory_vs_nfe.png. No number entered by hand.
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
    p.add_argument("--results_dir", default="results/c4")
    p.add_argument("--figures_dir", default="figures/c4")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "c4_memory_vs_nfe.csv")
    df = df[df.faithful == 1]
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df.fwd_nfe_adj, df.mem_adjoint_mb, c="tab:green", label="adjoint (odeint_adjoint)")
    ax.scatter(df.fwd_nfe_adj, df.mem_direct_mb, c="tab:red", label="direct backprop (odeint)")
    # linear fits
    x = np.linspace(df.fwd_nfe_adj.min(), df.fwd_nfe_adj.max(), 50)
    for col, c in [("mem_adjoint_mb", "tab:green"), ("mem_direct_mb", "tab:red")]:
        s, b = np.polyfit(df.fwd_nfe_adj, df[col], 1)
        ax.plot(x, s * x + b, c=c, lw=1.0, ls="--", alpha=0.7,
                label=f"  slope {s:+.2f} MB/NFE")
    ax.set_xlabel("forward NFE (driven by solver tolerance)")
    ax.set_ylabel("peak GPU memory (MB), forward+backward")
    ax.set_title("C4: adjoint is O(1) in NFE; direct backprop grows (Chen Table 1)")
    ax.legend()
    fig.tight_layout(); fig.savefig(fd / "memory_vs_nfe.png", dpi=120)
    print(f"Wrote {fd}/memory_vs_nfe.png")
    sa = np.polyfit(df.fwd_nfe_adj, df.mem_adjoint_mb, 1)[0]
    sd = np.polyfit(df.fwd_nfe_adj, df.mem_direct_mb, 1)[0]
    print(f"slopes: adjoint {sa:+.4f} MB/NFE | direct {sd:+.4f} MB/NFE | ratio {sa/sd:.4f}")


if __name__ == "__main__":
    main()
