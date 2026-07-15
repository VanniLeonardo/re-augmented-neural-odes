"""D1 report: NODE vs ANODE across budgets on spheres + circles, at accurate tol.

Reads the two committed budget sweeps (spheres: results/budget/, circles:
results/budget_circles/) and evaluates the PRE-DECLARED refutation checks R1-R4 from
OVERNIGHT_LOG.md. Per the governing rule this prints the RAW NUMBERS and, separately, the
mechanical evaluation of each refutation condition -- the observation decides, not a verdict
string. recon_ok is carried so non-integrating rows are visible.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: Path, geometry: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "geometry" not in df.columns:
        df["geometry"] = geometry
    if "status" not in df.columns:
        df["status"] = "ok"
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spheres_dir", default="results/budget")
    p.add_argument("--circles_dir", default="results/budget_circles")
    args = p.parse_args()

    frames = []
    for geom, d in [("spheres", args.spheres_dir), ("circles", args.circles_dir)]:
        f = Path(d) / "budget_raw.csv"
        if f.exists():
            frames.append(_load(f, geom))
        else:
            print(f"[warn] missing {f}")
    df = pd.concat(frames, ignore_index=True)

    print("=" * 78)
    print("RAW: dense_acc (median[IQR] over seeds) and fwd_nfe_median (median over seeds), "
          "recon_ok count")
    print("=" * 78)
    for (geom, model), g in df.groupby(["geometry", "model"]):
        print(f"\n{geom} / {model}:")
        print(f"  {'budget':>7} {'dense_acc median[IQR]':>26} {'NFE med (median seeds)':>24} {'recon_ok':>9} {'capped':>7}")
        for b, gb in g.groupby("budget"):
            da = gb.dense_acc.to_numpy()
            q1, med, q3 = np.percentile(da, [25, 50, 75])
            ncap = int((gb.get("status", pd.Series(["ok"]*len(gb))) == "time_capped").sum())
            print(f"  {b:7d}   {med:.4f}[{q1:.4f},{q3:.4f}]   {np.median(gb.fwd_nfe_median):18.0f}   "
                  f"{int(gb.recon_ok.sum())}/{len(gb)}   {ncap}")

    print("\n" + "=" * 78)
    print("PRE-DECLARED REFUTATION CHECKS (raw numbers + condition; observation decides)")
    print("=" * 78)
    for geom, g in df.groupby("geometry"):
        b50 = g[g.budget == 50]
        node50 = b50[b50.model == "NODE"]
        an50 = b50[b50.model == "ANODE-p1"]
        if node50.empty or an50.empty:
            print(f"\n[{geom}] budget-50 rows incomplete; skipping"); continue
        node_nfe = np.median(node50.fwd_nfe_median); an_nfe = np.median(an50.fwd_nfe_median)
        node_acc = np.median(node50.dense_acc); an_acc = np.median(an50.dense_acc)
        # NFE growth 25->500 (median over seeds at each budget)
        def nfe_at(model, b):
            r = g[(g.model == model) & (g.budget == b)]
            return np.median(r.fwd_nfe_median) if not r.empty else float("nan")
        node_grow = nfe_at("NODE", 500) / nfe_at("NODE", 25)
        an_grow = nfe_at("ANODE-p1", 500) / nfe_at("ANODE-p1", 25)
        print(f"\n[{geom}]")
        print(f"  R1 (ANODE cheaper @50): NODE NFE {node_nfe:.0f} vs ANODE NFE {an_nfe:.0f} "
              f"-> REFUTED if ANODE>=NODE  [ANODE<NODE = {an_nfe < node_nfe}]")
        print(f"  R2 (NODE grows>=+30%, ANODE<+30%, 25->500): NODE x{node_grow:.2f}, ANODE x{an_grow:.2f} "
              f"-> REFUTED if NODE grow<1.30 or ANODE grow>=1.30")
        print(f"  R4 (ANODE>=NODE acc @50): NODE {node_acc:.4f} vs ANODE {an_acc:.4f} "
              f"-> REFUTED if ANODE<NODE beyond noise")
        print(f"  S1 STOP-check (NODE acc<=0.70 @>=50 => contradicts Dupont 'eventually approximates'): "
              f"min NODE dense_acc @>=50ep = {g[(g.model=='NODE')&(g.budget>=50)].dense_acc.min():.4f}")
    print("\n(R3 = do R1/R2 hold on BOTH geometries — read the two blocks above.)")


if __name__ == "__main__":
    main()
