"""Report the missing-slice grid.

Evaluates the conditions recorded in OVERNIGHT_LOG.md before the run, including the stop
condition, which covers both geometries. The held-out metrics are skewed across seeds, so
results are medians with interquartile ranges. Cells failing the reconstruction check are
counted and excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GAP = 0.05  # G3: a "material" advantage


def _q(s: pd.Series) -> str:
    return f"{s.median():.3f} [{s.quantile(.25):.3f},{s.quantile(.75):.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/slice_grid")
    ap.add_argument("--d2_dir", default="results/slice_spheres")
    ap.add_argument("--figures_dir", default="figures/slice_grid")
    a = ap.parse_args()

    raw = pd.read_csv(Path(a.results_dir) / "slice_grid.csv")
    df = raw[raw.recon_ok == 1]
    print(f"loaded {len(raw)} cells ({len(raw) - len(df)} fail recon and are excluded), "
          f"seeds {raw.seed.nunique()}, hardware {sorted(raw.hardware.unique())}")
    bad = raw[raw.recon_ok == 0].groupby(["geometry", "width_over_pi", "model"]).size()
    if len(bad):
        print("  recon failures by group:\n" + bad.to_string())

    geoms = [g for g in ("spheres", "circles") if g in set(df.geometry)]
    widths = sorted(df.width_over_pi.unique())
    order = sorted(df.augment_dim.unique())
    name = {p: ("NODE" if p == 0 else f"ANODE-p{p}") for p in order}
    med = df.groupby(["geometry", "width_over_pi", "augment_dim"]).slice_val_acc.median()

    for g in geoms:
        for metric in ("slice_val_acc", "slice_val_loss", "obs_val_acc"):
            print(f"\n=== {g}: {metric}  median [IQR]  (rows: model, cols: wedge width / pi) ===")
            print(f"{'':10}" + "".join(f"{w:>26.4f}" for w in widths))
            for p in order:
                cells = [df[(df.geometry == g) & (df.width_over_pi == w) & (df.augment_dim == p)][metric]
                         for w in widths]
                print(f"{name[p]:10}" + "".join(f"{_q(c) if len(c) else '-':>26}" for c in cells))

    print("\n" + "=" * 78)
    print("PRE-DECLARED CHECKS (raw numbers + condition; observation decides)")
    print("=" * 78)
    anodes = [p for p in order if p > 0]

    def adv(g, p, w):
        return med[(g, w, p)] - med[(g, w, 0)]

    if "spheres" in geoms:
        print("\n[G1] spheres: every ANODE median slice acc >= NODE's at every width")
        worse = [(p, w) for p in anodes for w in widths if adv("spheres", p, w) < 0]
        for w in widths:
            print(f"  w={w:.4f}pi  NODE {med[('spheres', w, 0)]:.3f} | " +
                  " ".join(f"{name[p]} {med[('spheres', w, p)]:.3f}" for p in anodes))
        print(f"  => G1 {'HOLDS' if not worse else 'REFUTED at ' + str(worse)}")

        print("\n[G2] spheres: advantage A_p(w) non-decreasing in width, for the majority of p")
        mono = {}
        for p in anodes:
            seq = [adv("spheres", p, w) for w in widths]
            mono[p] = all(b >= a - 1e-9 for a, b in zip(seq, seq[1:]))
            print(f"  {name[p]:9} A(w) = {' -> '.join(f'{v:+.3f}' for v in seq)}  monotone: {mono[p]}")
        g2 = sum(mono.values()) > len(mono) / 2
        print(f"  => G2 {'HOLDS' if g2 else 'REFUTED'} ({sum(mono.values())}/{len(mono)} monotone)")


    # The stop condition covers both geometries, as recorded before the run. Narrowing
    # it after seeing a one-seed probe would change the condition after the fact.
    print("\n[STOP] any ANODE worse than NODE on BOTH median slice loss AND acc (any geometry)")
    lmed = df.groupby(["geometry", "width_over_pi", "augment_dim"]).slice_val_loss.median()
    stop = [(g, name[p], w) for g in geoms for p in anodes for w in widths
            if adv(g, p, w) < 0 and lmed[(g, w, p)] > lmed[(g, w, 0)]]
    print("  none" if not stop else f"  *** STOP-AND-FLAG: {stop} -- ANODE generalises worse "
          "than NODE; do not resolve alone ***")

    if "circles" in geoms:
        wmax = max(widths)
        print(f"\n[G3] circles: A_p > {GAP} at the widest wedge ({wmax:.4f}pi), majority of p")
        ok = {p: adv("circles", p, wmax) > GAP for p in anodes}
        for p in anodes:
            print(f"  {name[p]:9} A = {adv('circles', p, wmax):+.3f}  ({'material' if ok[p] else 'NOT material'})")
        print(f"  => G3 {'HOLDS' if sum(ok.values()) > len(ok) / 2 else 'REFUTED'}")

    d2 = Path(a.d2_dir) / "slice_raw.csv"
    if d2.exists():
        ref = pd.read_csv(d2)
        cell = raw[(raw.geometry == "spheres") & (raw.width_over_pi.round(4) == 0.2)
                   & (raw.augment_dim.isin([0, 1]))]
        m = cell.merge(ref, on=["model", "seed"], suffixes=("_grid", "_d2"))
        if len(m):
            d = (m.slice_val_acc_grid - m.slice_val_acc_d2).abs()
            print(f"\n[consistency] grid (spheres, 0.2pi, NODE/ANODE-p1) vs committed D2, "
                  f"{len(m)} matched seeds: max |slice acc diff| = {d.max():.4f} "
                  "(same code + config; differences come only from CPU thread-count float order)")

    fd = Path(a.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(geoms), 2, figsize=(11, 4 * len(geoms)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for r, g in enumerate(geoms):
        for c, metric in enumerate(("slice_val_acc", "slice_val_loss")):
            ax = axes[r][c]
            for i, p in enumerate(order):
                q = df[(df.geometry == g) & (df.augment_dim == p)].groupby("width_over_pi")[metric]
                col = "tab:red" if p == 0 else cmap(0.25 + 0.6 * i / max(1, len(order) - 1))
                ax.plot(q.median().index, q.median().values, "o-", color=col, label=name[p])
                ax.fill_between(q.median().index, q.quantile(.25).values, q.quantile(.75).values,
                                color=col, alpha=0.15)
            ax.set_xlabel("removed wedge width / π"); ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"{g}: held-out {'accuracy' if c == 0 else 'loss'} (median, IQR)")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / "slice_grid.png", dpi=120)
    print(f"\nWrote {fd / 'slice_grid.png'}")


if __name__ == "__main__":
    main()
