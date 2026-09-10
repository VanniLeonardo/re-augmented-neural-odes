"""Report the matched-parameter MNIST experiment (Dupont Table 1).

Reads the per-seed files, applies the reporting rule in scripts.image_table, and writes the
accuracy, NFE and overfitting figures.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.image_table import faithful_table, load_shards, per_epoch

COL = {"NODE": "tab:red", "ANODE-p5": "tab:blue"}
DUPONT = {"NODE": (96.4, 0.5), "ANODE-p5": (98.2, 0.1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/d3")
    p.add_argument("--figures_dir", default="figures/d3")
    args = p.parse_args()
    rd, fd = Path(args.results_dir), Path(args.figures_dir)
    df = load_shards(rd, "d3_trajectory")
    ep = per_epoch(df)
    fd.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in ep.groupby("model"):
        piv = g.pivot_table(index="epoch", columns="seed", values="test_acc")
        ax.plot(piv.index, piv.mean(axis=1), color=COL.get(m, "gray"), label=m)
        ax.fill_between(piv.index, piv.mean(axis=1) - piv.std(axis=1),
                        piv.mean(axis=1) + piv.std(axis=1), color=COL.get(m, "gray"), alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("MNIST test accuracy"); ax.set_title("D3: NODE vs ANODE (matched params)")
    ax.legend(); fig.tight_layout(); fig.savefig(fd / "test_acc.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4.5))  # D6: NFE vs loss
    for m, g in ep.groupby("model"):
        ax.scatter(g.test_loss, g.train_fwd_nfe, color=COL.get(m, "gray"), s=25, alpha=0.6, label=m)
    ax.set_xlabel("test loss"); ax.set_ylabel("forward NFE (train tol)"); ax.set_title("D6: NFE vs loss")
    ax.legend(); fig.tight_layout(); fig.savefig(fd / "nfe_vs_loss.png", dpi=120)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, g in ep.groupby("model"):
        gg = g.groupby("epoch")
        a1.plot(gg.train_fwd_nfe.mean().index, gg.train_fwd_nfe.mean(), color=COL.get(m, "gray"), label=m)
        a2.plot(gg.gap_acc.mean().index, gg.gap_acc.mean(), color=COL.get(m, "gray"), label=m)
    a1.set_xlabel("epoch"); a1.set_ylabel("forward NFE (train tol)"); a1.set_title("D6: NFE over training"); a1.legend()
    a2.set_xlabel("epoch"); a2.set_ylabel("train-test acc gap"); a2.set_title("D7: overfit gap"); a2.legend()
    fig.tight_layout(); fig.savefig(fd / "nfe_and_gap.png", dpi=120)

    for m, g in ep.groupby("model"):  # D6 growth summary, per-epoch rows only
        gg = g.groupby("epoch").train_fwd_nfe.median()
        print(f"  {m} train-tol NFE {gg.iloc[0]:.1f}->{gg.iloc[-1]:.1f} (x{gg.iloc[-1]/gg.iloc[0]:.2f})")

    faithful_table(df, "faithful_fwd_nfe", DUPONT, fd,
                   replicate=(rd / "run1_rtx3090" / "d3_trajectory.csv", "run 1, RTX 3090"))
    print(f"Wrote figures to {fd}")


if __name__ == "__main__":
    main()
