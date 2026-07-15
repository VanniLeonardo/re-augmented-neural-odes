"""D4 report + figures: matched-param NODE vs ANODE on CIFAR-10 (Dupont Table 1), + D6/D7.

Reads results/d4/d4_trajectory.csv. Evaluates the pre-declared refutation (ANODE >= NODE test acc
at matched params with lower/flatter NFE, at a recon-faithful tol), reports raw vs Dupont Table 1
(NODE 53.7+/-0.2, ANODE 60.6+/-0.4). Every reported row carries recon_ok; rows failing recon are
flagged, never reported as faithful.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"NODE": "tab:red", "ANODE-p10": "tab:blue"}
DUPONT = {"NODE": (53.7, 0.2), "ANODE-p10": (60.6, 0.4)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/d4")
    p.add_argument("--figures_dir", default="figures/d4")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "d4_trajectory.csv")
    # if a ladder was used, pick the loosest eval_tol that is recon_ok for BOTH at the last epoch
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)
    last = df.epoch.max()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in df[df.eval_tol == df.eval_tol.min()].groupby("model"):
        piv = g.pivot_table(index="epoch", columns="seed", values="test_acc")
        ax.plot(piv.index, piv.mean(axis=1), color=COL.get(m, "gray"), label=m)
        ax.fill_between(piv.index, piv.mean(axis=1) - piv.std(axis=1),
                        piv.mean(axis=1) + piv.std(axis=1), color=COL.get(m, "gray"), alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("CIFAR-10 test accuracy"); ax.set_title("D4: NODE vs ANODE (matched params)")
    ax.legend(); fig.tight_layout(); fig.savefig(fd / "test_acc.png", dpi=120)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, g in df.groupby("model"):
        gg = g.groupby("epoch")
        a1.plot(gg.train_fwd_nfe.mean().index, gg.train_fwd_nfe.mean(), color=COL.get(m, "gray"), label=m)
        a2.plot(gg.gap_acc.mean().index, gg.gap_acc.mean(), color=COL.get(m, "gray"), label=m)
    a1.set_xlabel("epoch"); a1.set_ylabel("forward NFE"); a1.set_title("D6: NFE over training"); a1.legend()
    a2.set_xlabel("epoch"); a2.set_ylabel("train-test acc gap"); a2.set_title("D7: overfit gap"); a2.legend()
    fig.tight_layout(); fig.savefig(fd / "nfe_and_gap.png", dpi=120)

    print("=" * 70)
    fin = df[df.epoch == last]
    # find loosest common recon-faithful eval tol
    tols = sorted(fin.eval_tol.unique(), reverse=True)
    faithful_tol = None
    for tol in tols:
        oks = {m: fin[(fin.model == m) & (fin.eval_tol == tol)].recon_ok.mean() for m in COL}
        if all(v >= 1.0 for v in oks.values()):
            faithful_tol = tol; break
    print(f"Final epoch {last}. Loosest common recon-faithful eval tol: "
          f"{faithful_tol if faithful_tol else 'NONE (report tightest + recon_ok counts)'}")
    use_tol = faithful_tol if faithful_tol else min(tols)
    stats = {}
    for m, g in fin[fin.eval_tol == use_tol].groupby("model"):
        acc = g.test_acc
        nfe = g[g.eval_fwd_nfe > 0].eval_fwd_nfe
        stats[m] = (acc.median(), acc.std(), nfe.median(), int(g.recon_ok.sum()), len(g))
        dm, ds = DUPONT.get(m, (float("nan"), float("nan")))
        print(f"  {m:10}: test_acc {acc.median()*100:.2f}±{acc.std()*100:.2f}%  (Dupont {dm}±{ds}) "
              f"| faithful NFE {nfe.median():.0f} | recon_ok {int(g.recon_ok.sum())}/{len(g)} @tol {use_tol:.0e}")
    print("\nPRE-DECLARED REFUTATION (ANODE >= NODE acc AND ANODE cheaper NFE, recon-faithful):")
    if "NODE" in stats and "ANODE-p10" in stats:
        na, _, nn_, _, _ = stats["NODE"]; aa, _, an, _, _ = stats["ANODE-p10"]
        print(f"  ANODE acc {aa*100:.2f} vs NODE {na*100:.2f} -> REFUTED if ANODE<NODE [{aa>=na}]")
        print(f"  ANODE NFE {an:.0f} vs NODE {nn_:.0f} -> REFUTED if ANODE>=NODE [{an<nn_}]")
    print("  (a MISS vs Dupont is a partial-replication finding; not tuned)")
    print(f"Wrote figures to {fd}")


if __name__ == "__main__":
    main()
