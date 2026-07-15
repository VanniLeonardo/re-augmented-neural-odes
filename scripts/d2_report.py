"""D2 report: NODE vs ANODE missing-slice generalisation, evaluate pre-declared R-D2a/b.

Reads results/slice_circles/ and results/slice_spheres/ (slice_raw.csv) and prints, per
(geometry, model), median[IQR] over seeds of: train_loss, slice_val_loss, slice_val_acc, and the
generalisation gap. Then the mechanical evaluation of R-D2a (NODE has a gap) and R-D2b (ANODE
generalises better) with RAW numbers -- the observation decides, not a verdict string. recon_ok
carried so non-integrating rows are visible.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def med_iqr(a):
    q1, m, q3 = np.percentile(a, [25, 50, 75])
    return m, q1, q3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", default="results/slice_circles,results/slice_spheres")
    args = p.parse_args()
    frames = []
    for d in args.dirs.split(","):
        f = Path(d.strip()) / "slice_raw.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            print(f"[warn] missing {f}")
    if not frames:
        print("no data yet"); return
    df = pd.concat(frames, ignore_index=True)

    print("=" * 84)
    print("RAW (median[IQR] over seeds). SLICE = held-out wedge = the generalisation metric.")
    print("=" * 84)
    for (geom, model), g in df.groupby(["geometry", "model"]):
        tl = med_iqr(g.train_loss.to_numpy())
        sl = med_iqr(g.slice_val_loss.to_numpy())
        sa = med_iqr(g.slice_val_acc.to_numpy())
        fa = med_iqr(g.full_val_acc.to_numpy())
        print(f"\n{geom:8} / {model:9} (n={len(g)}, recon_ok {int(g.recon_ok.sum())}/{len(g)})")
        print(f"   train_loss   {tl[0]:.3f}[{tl[1]:.3f},{tl[2]:.3f}]")
        print(f"   SLICE loss   {sl[0]:.3f}[{sl[1]:.3f},{sl[2]:.3f}]   SLICE acc {sa[0]:.3f}[{sa[1]:.3f},{sa[2]:.3f}]")
        print(f"   full_val_acc {fa[0]:.3f}[{fa[1]:.3f},{fa[2]:.3f}]")

    print("\n" + "=" * 84)
    print("PRE-DECLARED REFUTATION CHECKS (raw numbers + condition)")
    print("=" * 84)
    for geom, g in df.groupby("geometry"):
        node = g[g.model == "NODE"]; an = g[g.model == "ANODE-p1"]
        if node.empty or an.empty:
            print(f"\n[{geom}] incomplete"); continue
        node_sl = np.median(node.slice_val_loss); node_tl = np.median(node.train_loss)
        an_sl = np.median(an.slice_val_loss)
        node_sa = np.median(node.slice_val_acc); an_sa = np.median(an.slice_val_acc)
        gap_ratio = node_sl / node_tl if node_tl > 0 else float("inf")
        print(f"\n[{geom}]")
        print(f"  R-D2a (NODE gap): NODE slice_loss {node_sl:.3f} vs train_loss {node_tl:.3f} "
              f"(ratio {gap_ratio:.2f}) -> REFUTED if ratio < ~2")
        print(f"  R-D2b (ANODE better): ANODE slice_loss {an_sl:.3f} vs NODE slice_loss {node_sl:.3f} "
              f"| ANODE slice_acc {an_sa:.3f} vs NODE {node_sa:.3f} "
              f"-> REFUTED if ANODE_loss >= NODE_loss")
        print(f"  S3 STOP-check (ANODE worse on BOTH loss AND acc = contradicts Dupont): "
              f"ANODE worse? {an_sl > node_sl and an_sa < node_sa}")


if __name__ == "__main__":
    main()
