"""Plots for the budget-dependence sweep (reproduces Dupont Fig 6 / Fig 7 signature).

Reads results/budget/{budget_raw.csv, budget_trajectory.csv} and writes figures/budget/:
  nfe_vs_epoch.png   NFE trajectory during training, NODE vs ANODE (Dupont Fig 6)
  acc_nfe_vs_budget.png  dense accuracy AND forward NFE vs training budget, both models
Every number is read from the committed CSVs (no manual entry).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"NODE": "tab:red", "ANODE-p1": "tab:blue"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/budget")
    p.add_argument("--figures_dir", default="figures/budget")
    args = p.parse_args()
    rd, fd = Path(args.results_dir), Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(rd / "budget_raw.csv")
    traj = pd.read_csv(rd / "budget_trajectory.csv")

    # --- Fig 6 analog: NFE-vs-epoch, mean +/- std across seeds -----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in traj.groupby("model"):
        piv = g.pivot_table(index="epoch", columns="seed", values="train_fwd_nfe")
        m, s = piv.mean(axis=1), piv.std(axis=1)
        ax.plot(piv.index, m, color=COLORS.get(model, "gray"), label=model)
        ax.fill_between(piv.index, m - s, m + s, color=COLORS.get(model, "gray"), alpha=0.2)
    ax.set_xlabel("training epoch"); ax.set_ylabel("forward NFE (train, accurate tol)")
    ax.set_title("NFE grows as the NODE breaks apart the annulus (Dupont Fig 6)")
    ax.legend(); fig.tight_layout(); fig.savefig(fd / "nfe_vs_epoch.png", dpi=120)

    # --- acc AND NFE vs budget (dual axis), NODE vs ANODE ----------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, g in raw.groupby("model"):
        agg = g.groupby("budget").agg(
            acc_m=("dense_acc", "mean"), acc_s=("dense_acc", "std"),
            nfe_m=("fwd_nfe_median", "mean"), nfe_s=("fwd_nfe_median", "std"),
        ).reset_index()
        c = COLORS.get(model, "gray")
        a1.errorbar(agg.budget, agg.acc_m, yerr=agg.acc_s, marker="o", color=c, label=model, capsize=3)
        a2.errorbar(agg.budget, agg.nfe_m, yerr=agg.nfe_s, marker="o", color=c, label=model, capsize=3)
    a1.axhline(2/3, ls="--", c="gray", lw=0.8, label="majority (0.667)")
    a1.set_xscale("log"); a1.set_xlabel("training budget (epochs)")
    a1.set_ylabel("dense continuum accuracy"); a1.set_title("Accuracy vs budget")
    a1.legend()
    a2.set_xscale("log"); a2.set_xlabel("training budget (epochs)")
    a2.set_ylabel("forward NFE (median, accurate tol)")
    a2.set_title("Cost (NFE) vs budget"); a2.legend()
    fig.tight_layout(); fig.savefig(fd / "acc_nfe_vs_budget.png", dpi=120)

    # console summary at Dupont's 50-epoch budget
    d50 = raw[raw.budget == 50]
    print("At Dupont's 50-epoch budget (accurate tol), mean over seeds:")
    for model, g in d50.groupby("model"):
        print(f"  {model:9s}: dense acc {g.dense_acc.mean():.4f}+/-{g.dense_acc.std():.4f} | "
              f"val acc {g.val_acc.mean():.4f} | fwd NFE median {g.fwd_nfe_median.mean():.0f} | "
              f"recon_ok {int(g.recon_ok.sum())}/{len(g)}")
    print(f"\nWrote {fd}/nfe_vs_epoch.png and acc_nfe_vs_budget.png")


if __name__ == "__main__":
    main()
