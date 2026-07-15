"""Report the D3 faithful NFE re-measurement: find the loosest COMMON recon-faithful tolerance
and report the ANODE/NODE forward-NFE ratio there (the submission-clean comparison).

Reads results/d3_faithful/d3_faithful_trajectory.csv. Writes figures/d3/faithful_nfe.png and
prints, per eval tol at the final epoch: NODE/ANODE median NFE + recon_ok fraction, then the
loosest tol where BOTH are recon_ok 5/5, and the NFE ratio there.
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
    p.add_argument("--results_dir", default="results/d3_faithful")
    p.add_argument("--figures_dir", default="figures/d3")
    args = p.parse_args()
    df = pd.read_csv(Path(args.results_dir) / "d3_faithful_trajectory.csv")
    fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)
    last = df.epoch.max()
    tols = sorted(df.eval_tol.unique(), reverse=True)

    fin = df[df.epoch == last]
    print(f"=== final epoch {last}: median fwd NFE + recon_ok fraction (5 seeds) ===")
    print(f"{'eval_tol':>9} | {'NODE nfe':>9} {'NODE ok':>8} | {'ANODE nfe':>10} {'ANODE ok':>9}")
    common = []
    for tol in tols:
        n = fin[(fin.model == "NODE") & (fin.eval_tol == tol)]
        a = fin[(fin.model == "ANODE-p5") & (fin.eval_tol == tol)]
        # NFE ignoring capped (-1) for the median, but count them as not-ok
        n_nfe = n[n.fwd_nfe > 0].fwd_nfe.median(); a_nfe = a[a.fwd_nfe > 0].fwd_nfe.median()
        n_ok = n.recon_ok.mean(); a_ok = a.recon_ok.mean()
        print(f"{tol:9.0e} | {n_nfe:9.0f} {n_ok:8.2f} | {a_nfe:10.0f} {a_ok:9.2f}")
        if n_ok >= 1.0 and a_ok >= 1.0:
            common.append((tol, n_nfe, a_nfe))

    print("\n=== faithful comparison ===")
    if common:
        tol, n_nfe, a_nfe = max(common, key=lambda r: r[0])  # loosest common-faithful tol
        print(f"  Loosest tol where BOTH recon_ok 5/5 at ep{last}: {tol:.0e}")
        print(f"  NODE NFE {n_nfe:.0f} vs ANODE NFE {a_nfe:.0f}  ->  ratio NODE/ANODE = {n_nfe/a_nfe:.2f}x")
        print(f"  Claim 'ANODE cheaper at a common recon-faithful tol': "
              f"{'HOLDS' if a_nfe < n_nfe else 'REFUTED'} (ANODE<NODE = {a_nfe < n_nfe})")
    else:
        # no common 5/5 tol -> report the tightest measured + how many NODE seeds still fail
        tightest = min(tols)
        n = fin[(fin.model == "NODE") & (fin.eval_tol == tightest)]
        print(f"  NO tol has BOTH 5/5 at ep{last}. Even at {tightest:.0e}, NODE recon_ok "
              f"{int(n.recon_ok.sum())}/{len(n)} (stiffest NODE seeds not integrable within cap).")
        print(f"  This IS the Dupont mechanism (NODE flow too stiff) -- report as data, NFE lower-bounded.")
        ok = n[n.recon_ok == 1]
        a = fin[(fin.model == "ANODE-p5") & (fin.eval_tol == tightest)]
        if len(ok):
            print(f"  Among recon_ok NODE seeds at {tightest:.0e}: NODE NFE {ok.fwd_nfe.median():.0f} "
                  f"vs ANODE {a[a.fwd_nfe>0].fwd_nfe.median():.0f}")

    # figure: faithful NFE vs epoch at each tol, per model
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m, c in [("NODE", "tab:red"), ("ANODE-p5", "tab:blue")]:
        for tol, ls in zip(tols, ["-", "--", ":"]):
            g = df[(df.model == m) & (df.eval_tol == tol) & (df.fwd_nfe > 0)].groupby("epoch").fwd_nfe.median()
            ax.plot(g.index, g.values, ls, color=c, alpha=0.9,
                    label=f"{m} @{tol:.0e}")
    ax.set_xlabel("epoch"); ax.set_ylabel("forward NFE (faithful, median)")
    ax.set_title("D3 faithful NFE: NODE vs ANODE at recon-checked tolerances")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(fd / "faithful_nfe.png", dpi=120)
    print(f"\nWrote {fd}/faithful_nfe.png")


if __name__ == "__main__":
    main()
