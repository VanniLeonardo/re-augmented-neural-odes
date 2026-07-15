"""C2 consolidated artifact: the bwd/fwd NFE ratio as a tolerance x field surface.

Reads the toy data (results/c2/c2_recharacterise.csv: spheres+circles, trained+untrained) and the
MNIST data (results/c2/c2_mnist.csv), writes:
  results/c2/c2_surface.csv     one row per (field, state, tol): median ratio + recon_ok fraction
  figures/c2/c2_surface.png     ratio vs tolerance, per field x state (solid = integrating)
Prints the matched-tolerance field comparison used by the pre-declared refutation.

Through-line (framing stub for the co-author -- not paper narrative): the ratio is governed by the
tolerance a field needs to be faithfully integrated. The prose belongs to the human author.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_surface(results_dir: Path) -> pd.DataFrame:
    rows = []
    toy = results_dir / "c2_recharacterise.csv"
    if toy.exists():
        df = pd.read_csv(toy)
        df["field"] = df["geometry"]
        df["state"] = np.where(df["epochs"] == 0, "untrained", "trained")
        for (field, state, tol), g in df.groupby(["field", "state", "tol"]):
            rows.append({"field": field, "state": state, "tol": tol,
                         "ratio_median": g.bwd_over_fwd.median(),
                         "recon_ok_frac": g.recon_ok.mean(), "n": len(g)})
    mn = results_dir / "c2_mnist.csv"
    if mn.exists():
        df = pd.read_csv(mn)
        for (state, tol), g in df.groupby(["state", "tol"]):
            rows.append({"field": "mnist_conv", "state": state, "tol": tol,
                         "ratio_median": g.bwd_over_fwd.median(),
                         "recon_ok_frac": g.recon_ok.mean(), "n": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/c2")
    p.add_argument("--figures_dir", default="figures/c2")
    args = p.parse_args()
    rd = Path(args.results_dir); fd = Path(args.figures_dir); fd.mkdir(parents=True, exist_ok=True)
    surf = _load_surface(rd).sort_values(["field", "state", "tol"])
    surf.to_csv(rd / "c2_surface.csv", index=False)

    colors = {"spheres": "tab:red", "circles": "tab:blue", "mnist_conv": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (field, state), g in surf.groupby(["field", "state"]):
        g = g.sort_values("tol")
        ls = "-" if state == "trained" else "--"
        ax.plot(g.tol, g.ratio_median, ls, color=colors.get(field, "gray"), alpha=0.9,
                label=f"{field} ({state})")
        for _, r in g.iterrows():
            ax.scatter(r.tol, r.ratio_median, s=55, color=colors.get(field, "gray"),
                       facecolors=colors.get(field, "gray") if r.recon_ok_frac >= 0.5 else "none",
                       zorder=5)
    ax.axhline(1.0, ls=":", c="k", lw=0.8, label='old claim "bwd~=fwd" (=1)')
    ax.axhline(0.5, ls=":", c="gray", lw=0.8, label="Chen (bwd~=0.5 fwd)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("solver tolerance (atol=rtol; tighter →)")
    ax.set_ylabel("backward / forward NFE ratio (median)")
    ax.set_title("C2 surface: adaptive bwd/fwd ratio is tolerance-driven across fields\n"
                 "(solid = integrating / recon_ok; hollow = non-integrating)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(fd / "c2_surface.png", dpi=120)

    print("=== C2 surface (median ratio; recon_ok fraction) ===")
    print(surf.to_string(index=False))
    print("\n=== matched-tolerance field comparison (pre-declared: same order => tolerance-driven) ===")
    tr = surf[surf.state == "trained"]
    for tol in sorted(tr.tol.unique()):
        row = tr[tr.tol == tol]
        vals = {r.field: (r.ratio_median, r.recon_ok_frac) for _, r in row.iterrows()}
        s = " | ".join(f"{f} {v[0]:.1f}(ok{v[1]:.1f})" for f, v in vals.items())
        print(f"  tol {tol:.0e}: {s}")
    print(f"\nWrote {rd}/c2_surface.csv and {fd}/c2_surface.png")


if __name__ == "__main__":
    main()
