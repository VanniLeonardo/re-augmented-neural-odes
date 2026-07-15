"""D2 figure: held-out-slice generalisation, NODE vs ANODE (Dupont Fig 9).

Reads results/slice_spheres/slice_raw.csv, writes figures/slice/slice_generalisation.png:
per-seed slice (held-out wedge) accuracy and loss, NODE vs ANODE. No hand numbers.
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
    p.add_argument("--results_dir", default="results/slice_spheres")
    p.add_argument("--figures_dir", default="figures/slice")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "slice_raw.csv")
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)
    order = ["NODE", "ANODE-p1"]
    colors = {"NODE": "tab:red", "ANODE-p1": "tab:blue"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, m in enumerate(order):
        g = df[df.model == m]
        a1.scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-0.05, 0.05, len(g)),
                   g.slice_val_acc, color=colors[m], s=50, zorder=3)
        a1.hlines(g.slice_val_acc.median(), i - 0.2, i + 0.2, color="k", lw=2)
        a2.scatter(np.full(len(g), i) + np.random.default_rng(1).uniform(-0.05, 0.05, len(g)),
                   g.slice_val_loss, color=colors[m], s=50, zorder=3)
        a2.hlines(g.slice_val_loss.median(), i - 0.2, i + 0.2, color="k", lw=2)
    a1.axhline(2/3, ls="--", c="gray", lw=0.8, label="majority (0.667)")
    a1.set_xticks(range(len(order))); a1.set_xticklabels(order)
    a1.set_ylabel("held-out slice accuracy"); a1.set_title("Generalisation to the removed wedge")
    a1.legend()
    a2.set_xticks(range(len(order))); a2.set_xticklabels(order)
    a2.set_ylabel("held-out slice loss"); a2.set_title("Held-out slice loss (log)")
    a2.set_yscale("symlog", linthresh=1e-2)
    fig.suptitle("D2 (Dupont Fig 9): NODE fails on the held-out slice; ANODE generalises")
    fig.tight_layout(); fig.savefig(fd / "slice_generalisation.png", dpi=120)
    print(f"Wrote {fd}/slice_generalisation.png")
    for m in order:
        g = df[df.model == m]
        print(f"  {m}: slice_acc median {g.slice_val_acc.median():.3f} | "
              f"slice_loss median {g.slice_val_loss.median():.3f} | recon_ok {int(g.recon_ok.sum())}/{len(g)}")


if __name__ == "__main__":
    main()
