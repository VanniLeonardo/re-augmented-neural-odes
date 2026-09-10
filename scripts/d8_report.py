"""Report the one-dimensional crossing flow (Dupont Proposition 1, Figure 3).

Lists every tolerance, then compares the two models at the loosest tolerance where both
pass the reconstruction check. An order-preserving map can do no better than a mean squared
error of 1 on this task, so that value is the floor the Neural ODE is expected to reach.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COL = {"NODE": "tab:red", "ANODE-p1": "tab:blue"}
THEORY_FLOOR = 1.0  # best order-preserving map: predict 0 for targets +/-1
SOLVED = 0.1        # pre-declared: MSE below this counts as "solves the task"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/crossing")
    p.add_argument("--figures_dir", default="figures/crossing")
    args = p.parse_args()

    df = pd.read_csv(Path(args.results_dir) / "crossing_summary.csv")
    arms = sorted(df.model_name.unique(), key=lambda m: m != "NODE")
    n_seeds = df.seed.nunique()

    print(f"=== D8 crossing flow: {n_seeds} seeds, trained at tol "
          f"{df.train_tol.iloc[0]:.0e} ===")
    print(f"{'eval_tol':>9} | " + " | ".join(
        f"{m:>9} MSE  NFE  recon" for m in arms))
    for tol, g in df.groupby("eval_tol"):
        cells = []
        for m in arms:
            s = g[g.model_name == m]
            cells.append(f"{s.mse.median():>13.4f} {s.fwd_nfe.median():4.0f} "
                         f"{int(s.recon_ok.sum())}/{len(s)}")
        print(f"{tol:>9.0e} | " + " | ".join(cells))

    faithful = [t for t, g in df.groupby("eval_tol")
                if all(g[g.model_name == m].recon_ok.eq(1).all() for m in arms)]
    print("\n=== faithful comparison ===")
    if not faithful:
        print("  NO tolerance has BOTH arms recon_ok on every seed -- nothing is reportable "
              "as faithful. Record as data, do not headline.")
        return
    tol = max(faithful)  # loosest that is still faithful for both arms
    at = df[df.eval_tol == tol]
    med = {m: at[at.model_name == m] for m in arms}
    print(f"  Loosest tol where BOTH arms recon_ok {n_seeds}/{n_seeds}: {tol:.0e}")
    for m in arms:
        s = med[m]
        print(f"    {m:9s} MSE {s.mse.median():.4f} "
              f"[{s.mse.min():.4f}, {s.mse.max():.4f}]  fwd NFE {s.fwd_nfe.median():.0f}")

    node, anode = med["NODE"], med[[m for m in arms if m != "NODE"][0]]
    print("\nPRE-DECLARED REFUTATION (REFUTED if NODE MSE < 0.1, or ANODE MSE > 0.1):")
    print(f"  NODE MSE  {node.mse.median():.4f} -> solves? "
          f"{node.mse.median() < SOLVED}  (must be False)")
    print(f"  ANODE MSE {anode.mse.median():.4f} -> solves? "
          f"{anode.mse.median() < SOLVED}  (must be True)")
    refuted = (node.mse.median() < SOLVED) or (anode.mse.median() > SOLVED)
    print(f"  => {'REFUTED' if refuted else 'NOT REFUTED'}")
    print(f"  NODE MSE vs Prop.1 theory floor {THEORY_FLOOR:.1f}: "
          f"{node.mse.median():.4f}")
    if node.mse.median() < 0.5:
        print("  *** STOP-AND-FLAG: NODE materially beats the order-preserving floor at a "
              "recon-faithful tolerance -- this contradicts Proposition 1. Do not resolve "
              "alone. ***")

    fd = Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for m in arms:
        s = df[df.model_name == m].groupby("eval_tol")
        a1.plot(s.mse.median().index, s.mse.median().values, "o-", color=COL.get(m, "gray"), label=m)
    a1.axhline(THEORY_FLOOR, ls="--", c="k", lw=1, label="Prop. 1 floor (MSE=1)")
    a1.axvline(tol, ls=":", c="green", lw=1.5, label=f"faithful tol {tol:.0e}")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.invert_xaxis()
    a1.set_xlabel("eval tolerance"); a1.set_ylabel("MSE (median)")
    a1.set_title("NODE cannot cross; ANODE can"); a1.legend(fontsize=8)

    for m in arms:
        s = df[df.model_name == m].groupby("eval_tol")
        a2.plot(s.recon_rel.median().index, s.recon_rel.median().values, "o-",
                color=COL.get(m, "gray"), label=m)
    a2.axhline(1e-2, ls="--", c="k", lw=1, label="recon threshold")
    a2.axvline(tol, ls=":", c="green", lw=1.5)
    a2.set_xscale("log"); a2.set_yscale("log"); a2.invert_xaxis()
    a2.set_xlabel("eval tolerance"); a2.set_ylabel("fwd->bwd reconstruction error")
    a2.set_title("Where the flow is actually integrated"); a2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fd / "crossing_flow.png", dpi=120)
    print(f"\nWrote {fd / 'crossing_flow.png'}")


if __name__ == "__main__":
    main()
