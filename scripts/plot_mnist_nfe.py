"""C1/C3 figure: MNIST conv ODE-Net forward NFE and test accuracy over training.

Reads results/mnist_nfe/mnist_nfe_trajectory.csv and writes figures/mnist_nfe/nfe_over_training.png.
Shows train-tol NFE and the recon-checked faithful-tol NFE (both should grow if Chen holds), plus
test accuracy, mean +/- std over seeds. No number entered by hand.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/mnist_nfe")
    p.add_argument("--figures_dir", default="figures/mnist_nfe")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "mnist_nfe_trajectory.csv")
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, c, lab in [("train_fwd_nfe", "tab:red", f"train-tol NFE"),
                        ("faithful_fwd_nfe", "tab:purple", "faithful-tol NFE (recon-checked)")]:
        piv = df.pivot_table(index="epoch", columns="seed", values=col)
        m, s = piv.mean(axis=1), piv.std(axis=1)
        a1.plot(piv.index, m, color=c, label=lab)
        a1.fill_between(piv.index, m - s, m + s, color=c, alpha=0.2)
    a1.set_xlabel("epoch"); a1.set_ylabel("forward NFE"); a1.set_title("NFE over training (Chen)")
    a1.legend()

    piv = df.pivot_table(index="epoch", columns="seed", values="test_acc")
    m, s = piv.mean(axis=1), piv.std(axis=1)
    a2.plot(piv.index, m, color="tab:blue"); a2.fill_between(piv.index, m - s, m + s, alpha=0.2)
    a2.set_xlabel("epoch"); a2.set_ylabel("test accuracy"); a2.set_title("MNIST test accuracy")

    fig.tight_layout(); fig.savefig(fd / "nfe_over_training.png", dpi=120)
    # console: NFE growth check
    g = df.groupby("epoch")[["train_fwd_nfe", "faithful_fwd_nfe", "recon_ok"]].mean()
    print(g.to_string())
    e1, eN = g.index.min(), g.index.max()
    print(f"\ntrain-tol NFE {g.loc[e1,'train_fwd_nfe']:.1f}->{g.loc[eN,'train_fwd_nfe']:.1f} | "
          f"faithful-tol NFE {g.loc[e1,'faithful_fwd_nfe']:.1f}->{g.loc[eN,'faithful_fwd_nfe']:.1f} "
          f"(REFUTED if flat/decreasing)")
    print(f"Wrote {fd}/nfe_over_training.png")


if __name__ == "__main__":
    main()
