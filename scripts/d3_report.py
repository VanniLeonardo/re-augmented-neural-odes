"""D3 report + figures: matched-param NODE vs ANODE on MNIST (Dupont Table 1), + D6/D7.

Reads results/d3/d3_trajectory.csv. Evaluates the pre-declared refutation (ANODE >= NODE test acc
at matched params with lower/flatter NFE), reports raw vs Dupont Table 1 (NODE 96.4+/-0.5, ANODE
98.2+/-0.1), and produces:
  figures/d3/test_acc.png     test acc vs epoch, NODE vs ANODE
  figures/d3/nfe_vs_loss.png  D6: forward NFE vs test loss (how complex a flow for a given loss)
  figures/d3/nfe_and_gap.png  NFE vs epoch (D6) + train/test gap vs epoch (D7)
Every reported row is recon-checked (recon_ok carried).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"NODE": "tab:red", "ANODE-p5": "tab:blue"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/d3")
    p.add_argument("--figures_dir", default="figures/d3")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "d3_trajectory.csv")
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)
    last = df.epoch.max()

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in df.groupby("model"):
        piv = g.pivot_table(index="epoch", columns="seed", values="test_acc")
        ax.plot(piv.index, piv.mean(axis=1), color=COL.get(m, "gray"), label=m)
        ax.fill_between(piv.index, piv.mean(axis=1) - piv.std(axis=1),
                        piv.mean(axis=1) + piv.std(axis=1), color=COL.get(m, "gray"), alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("MNIST test accuracy"); ax.set_title("D3: NODE vs ANODE (matched params)")
    ax.legend(); fig.tight_layout(); fig.savefig(fd / "test_acc.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4.5))  # D6: NFE vs loss
    for m, g in df.groupby("model"):
        ax.scatter(g.test_loss, g.train_fwd_nfe, color=COL.get(m, "gray"), s=25, alpha=0.6, label=m)
    ax.set_xlabel("test loss"); ax.set_ylabel("forward NFE"); ax.set_title("D6: NFE vs loss")
    ax.legend(); fig.tight_layout(); fig.savefig(fd / "nfe_vs_loss.png", dpi=120)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, g in df.groupby("model"):
        gg = g.groupby("epoch")
        a1.plot(gg.train_fwd_nfe.mean().index, gg.train_fwd_nfe.mean(), color=COL.get(m, "gray"), label=m)
        a2.plot(gg.gap_acc.mean().index, gg.gap_acc.mean(), color=COL.get(m, "gray"), label=m)
    a1.set_xlabel("epoch"); a1.set_ylabel("forward NFE"); a1.set_title("D6: NFE over training"); a1.legend()
    a2.set_xlabel("epoch"); a2.set_ylabel("train-test acc gap"); a2.set_title("D7: overfit gap"); a2.legend()
    fig.tight_layout(); fig.savefig(fd / "nfe_and_gap.png", dpi=120)

    # --- pre-declared refutation, raw numbers ---
    print("=" * 70)
    fin = df[df.epoch == last]
    print(f"Final epoch {last} (median[IQR] over seeds), recon_ok carried:")
    stats = {}
    for m, g in fin.groupby("model"):
        acc = g.test_acc; nfe = g.train_fwd_nfe
        stats[m] = (acc.median(), acc.std(), nfe.median(), int(g.recon_ok.sum()), len(g))
        print(f"  {m:9}: test_acc {acc.median():.4f} (±{acc.std():.4f}) | fwd_nfe {nfe.median():.1f} "
              f"| recon_ok {int(g.recon_ok.sum())}/{len(g)}")
    # NFE growth over training per model
    for m, g in df.groupby("model"):
        gg = g.groupby("epoch").train_fwd_nfe.median()
        print(f"  {m} NFE {gg.iloc[0]:.1f}->{gg.iloc[-1]:.1f} (x{gg.iloc[-1]/gg.iloc[0]:.2f})")
    print("\nPRE-DECLARED REFUTATION (ANODE >= NODE acc AND ANODE cheaper NFE):")
    if "NODE" in stats and "ANODE-p5" in stats:
        na, _, nn_, _, _ = stats["NODE"]; aa, _, an, _, _ = stats["ANODE-p5"]
        print(f"  ANODE acc {aa:.4f} vs NODE acc {na:.4f}  -> REFUTED if ANODE<NODE  [ANODE>=NODE={aa>=na}]")
        print(f"  ANODE NFE {an:.1f} vs NODE NFE {nn_:.1f}  -> REFUTED if ANODE>=NODE  [ANODE<NODE={an<nn_}]")
    print(f"\n  vs Dupont Table 1: NODE 96.4±0.5 / ANODE 98.2±0.1 (report raw, a miss is a finding)")
    print(f"Wrote figures to {fd}")


if __name__ == "__main__":
    main()
