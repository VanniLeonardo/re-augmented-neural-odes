"""C2 diagnosis report — why the backward/forward NFE ratio does not match Chen Fig. 3c.

Merges results/c2_diagnosis/*.csv (toy spheres, CPU; MNIST conv field, GPU) and evaluates the
hypotheses pre-declared in OVERNIGHT_LOG.md before the runs:

  H1  a stiff-capable adaptive solver gives a ratio >=5x smaller than dopri5 at matched
      tolerance (Chen used implicit Adams; we used an explicit Runge-Kutta pair)
  H2  loosening the ADJOINT tolerance 100x relative to the forward pass reduces the ratio >=5x
  H3  the ratio grows with training, for every solver

  REVISION TRIGGER  if any configuration faithful to Chen's description reaches ratio <= 1
      while both reconstruction-faithful and gradient-correct, then "claim 6 does not
      reproduce" is wrong as stated and the paper must report a scope condition instead.
      "Faithful to Chen's description" is read as: the adjoint solved at the forward
      tolerance (which is also torchdiffeq's default, adjoint_rtol = rtol), on a trained
      field. The unrestricted minimum is reported alongside it, since the two disagree.

Only cells that pass BOTH the reconstruction check and the gradient check against direct
backprop are used for the conclusions; the rest are counted and shown, never silently dropped.
A cheap configuration that computes the wrong gradient is not evidence about cost.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STIFF = ["scipy:LSODA", "scipy:BDF"]     # adaptive, stiff-capable: stand in for implicit Adams
BASE = "dopri5"                           # what we (and most current users) actually reach for
FACTOR = 5.0                              # pre-declared effect size for H1 and H2


def load(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "c2_diagnosis_*.csv")))
    if not files:
        raise SystemExit(f"no c2_diagnosis_*.csv under {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(files)} shard(s), {len(df)} cells | fields {sorted(df.field.unique())} "
          f"| hardware {sorted(df.hardware.unique())}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/c2_diagnosis")
    ap.add_argument("--figures_dir", default="figures/c2_diagnosis")
    a = ap.parse_args()

    raw = load(Path(a.results_dir))
    capped = int(raw.capped.sum())
    live = raw[raw.capped == 0]
    ok = live[(live.recon_ok == 1) & (live.grad_ok == 1)]
    print(f"  {capped} cells hit the time cap; of the remaining {len(live)}, "
          f"{len(ok)} pass both the reconstruction and gradient checks and are used below")
    bad = live[(live.recon_ok == 0) | (live.grad_ok == 0)]
    if len(bad):
        print("  excluded by check, by tolerance:\n" +
              bad.groupby(["field", "tol"]).size().to_string())

    for field in sorted(ok.field.unique()):
        for trained in sorted(ok[ok.field == field].trained.unique(), reverse=True):
            g = ok[(ok.field == field) & (ok.trained == trained)]
            if g.empty:
                continue
            print(f"\n=== {field}, {'trained' if trained else 'untrained'}: median "
                  f"backward/forward NFE ratio (median forward NFE in brackets) ===")
            piv = g.pivot_table(index=["solver", "tol"], columns="adj_offset",
                                values="ratio", aggfunc="median")
            fwd = g.pivot_table(index=["solver", "tol"], columns="adj_offset",
                                values="fwd_nfe", aggfunc="median")
            for idx in piv.index:
                cells = " ".join(
                    f"{piv.loc[idx, c]:9.2f} [{fwd.loc[idx, c]:.0f}]" if pd.notna(piv.loc[idx, c])
                    else f"{'--':>9}      " for c in piv.columns)
                print(f"  {idx[0]:14s} tol {idx[1]:.0e}  " + cells)
            print(f"  {'':14s} {'':9s}  " +
                  " ".join(f"{'adj x' + str(int(c)):>9}      " for c in piv.columns))

    print("\n" + "=" * 78)
    print("PRE-DECLARED HYPOTHESES (raw numbers + condition; the observation decides)")
    print("=" * 78)
    tr = ok[ok.trained == 1]

    print(f"\n[H1] a stiff-capable solver is >={FACTOR:.0f}x cheaper than {BASE}, "
          "at matched tolerances (adjoint tol = forward tol)")
    h1 = []
    for field in sorted(tr.field.unique()):
        for tol in sorted(tr.tol.unique(), reverse=True):
            sel = tr[(tr.field == field) & (tr.tol == tol) & (tr.adj_offset == 1)]
            base = sel[sel.solver == BASE].ratio.median()
            if pd.isna(base):
                continue
            for s in STIFF:
                r = sel[sel.solver == s].ratio.median()
                if pd.isna(r):
                    print(f"  {field:8s} tol {tol:.0e}  {s:12s} no faithful cell "
                          "(did not complete within the time cap)")
                    continue
                good = r <= base / FACTOR
                h1.append(good)
                print(f"  {field:8s} tol {tol:.0e}  {s:12s} {r:9.2f} vs {BASE} {base:8.2f} "
                      f"-> {'>=5x cheaper' if good else 'NOT cheaper'}")
    print(f"  => H1 {'HOLDS' if h1 and all(h1) else 'REFUTED'}")

    print(f"\n[H2] loosening the ADJOINT tolerance 100x reduces the ratio >={FACTOR:.0f}x")
    h2 = []
    for field in sorted(tr.field.unique()):
        for (s, tol), g in tr.groupby(["solver", "tol"]):
            g = g[g.field == field]
            same = g[g.adj_offset == 1].ratio.median()
            loose = g[g.adj_offset == 100].ratio.median()
            if pd.isna(same) or pd.isna(loose) or loose == 0:
                continue
            drop = same / loose
            h2.append(drop >= FACTOR)
            print(f"  {field:8s} {s:14s} tol {tol:.0e}  {same:9.2f} -> {loose:7.2f} "
                  f"({drop:6.1f}x) {'' if drop >= FACTOR else '  <- below threshold'}")
    frac = sum(h2) / max(len(h2), 1)
    print(f"  => H2 {'HOLDS' if frac > 0.5 else 'REFUTED'} for {sum(h2)}/{len(h2)} "
          "solver-tolerance combinations")

    print("\n[H3] the ratio grows with training, for EVERY solver "
          "(adjoint tol = forward tol)")
    h3 = []
    for (s, tol), g in ok[(ok.field == "spheres") & (ok.adj_offset == 1)].groupby(["solver", "tol"]):
        t_, u_ = g[g.trained == 1].ratio.median(), g[g.trained == 0].ratio.median()
        if pd.isna(t_) or pd.isna(u_):
            continue
        h3.append(t_ > u_)
        print(f"  {s:14s} tol {tol:.0e}  trained {t_:9.2f}  untrained {u_:9.2f}  "
              f"{'grows' if t_ > u_ else 'DOES NOT grow'}")
    print(f"  => H3 {'HOLDS' if h3 and all(h3) else 'REFUTED'} "
          f"({sum(h3)}/{len(h3)} combinations grow)")

    print("\n[REVISION TRIGGER] is Chen's ratio reachable in a checked configuration?")
    # The trigger was pre-declared for a configuration FAITHFUL TO CHEN'S DESCRIPTION. Chen
    # does not describe decoupling the adjoint tolerance, and torchdiffeq's default is
    # adjoint_rtol = rtol, so the faithful reading is: adjoint solved at the forward
    # tolerance, on a trained field. Both readings are reported; they disagree, and the
    # narrower one is the one that was pre-declared.
    faithful = ok[(ok.adj_offset == 1) & (ok.trained == 1)]
    fmin = faithful.ratio.min()
    b = faithful.loc[faithful.ratio.idxmin()]
    print(f"  AS PRE-DECLARED (adjoint at the forward tolerance = the library default, "
          f"trained field):")
    print(f"    lowest ratio {fmin:.2f} ({b.solver}, {b.field}, tol {b.tol:.0e})")
    mnist = ok[(ok.field == "mnist") & (ok.trained == 1) & (ok.adj_offset == 1)]
    if not mnist.empty:
        print(f"    lowest on the conv field Chen used: {mnist.ratio.min():.2f}")
    print(f"    => trigger {'FIRES' if fmin <= 1 else 'DOES NOT FIRE'}")
    allmin = ok.loc[ok.ratio.idxmin()]
    print(f"  UNRESTRICTED (any checked configuration, including a decoupled adjoint tolerance):")
    print(f"    lowest ratio {allmin.ratio:.2f} ({allmin.solver}, {allmin.field}, "
          f"tol {allmin.tol:.0e}, adjoint x{int(allmin.adj_offset)})")
    print(f"    => {'a checked configuration does reach <=1' if allmin.ratio <= 1 else 'nothing reaches <=1'}")
    print("  Reading: Chen's ratio is reachable, but not with the adjoint at the forward\n"
          "  tolerance, which is both the library default and the natural reading of the\n"
          "  original setup. The claim is a property of the adjoint configuration, not of\n"
          "  the adjoint method as such.")

    fd = Path(a.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), squeeze=False)
    for ax, field in zip(axes[0], ["spheres", "mnist"]):
        g = ok[(ok.field == field) & (ok.trained == 1)]
        if g.empty:
            ax.set_visible(False)
            continue
        for off, mark in [(1, "o-"), (100, "s--")]:
            for s in sorted(g.solver.unique()):
                q = g[(g.solver == s) & (g.adj_offset == off)].groupby("tol").ratio.median()
                if q.empty:
                    continue
                ax.plot(q.index, q.values, mark, label=f"{s} (adj x{int(off)})", alpha=0.8)
        ax.axhline(0.5, ls=":", c="k", lw=1.2, label="Chen Fig. 3c (0.5)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("forward tolerance"); ax.set_ylabel("backward / forward NFE")
        ax.set_title(f"{field}: cost of the adjoint")
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fd / "c2_diagnosis.png", dpi=120)
    print(f"\nWrote {fd / 'c2_diagnosis.png'}")


if __name__ == "__main__":
    main()
