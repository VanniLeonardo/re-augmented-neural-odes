"""C1 report — solver dynamics on the trained MNIST conv ODE-Net (Chen Fig 3a-b).

Merges the per-seed shards written by `scripts/run_c1_solver_dynamics.py`
(results/c1/c1_solver_dynamics*.csv -- one per seed when run as a SLURM array) and
evaluates the pre-declared refutation checks:

  R-C1a  numerical error falls monotonically as tolerance tightens
  R-C1b  cost (forward NFE, wall-clock) rises monotonically as tolerance tightens
  R-C1c  forward time is rank-correlated with NFE (Chen Fig 3b), Spearman rho >= 0.9

plus a STOP-and-flag check: a higher-order solver showing systematically LARGER error
than a lower-order one at matched NFE would contradict numerical analysis, not just Chen.

Rows failing the reconstruction check are kept and shown (the loose-tolerance rows are
expected to fail -- that is the point of the axis) but are excluded from the faithful
cost/error claims and marked in the table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ADAPTIVE = ["bosh3", "dopri5", "dopri8"]
FIXED = ["euler", "midpoint", "rk4"]
ORDER = {"euler": 1, "midpoint": 2, "rk4": 4, "bosh3": 3, "dopri5": 5, "dopri8": 8}


def _monotone(vals: list[float], increasing: bool) -> bool:
    d = 1 if increasing else -1
    return all(d * (b - a) >= 0 for a, b in zip(vals, vals[1:]))


def load(results_dir: Path) -> pd.DataFrame:
    shards = sorted(results_dir.glob("c1_solver_dynamics*.csv"))
    if not shards:
        raise SystemExit(f"no C1 shards under {results_dir}")
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    print(f"loaded {len(shards)} shard(s), {len(df)} rows, "
          f"seeds {sorted(df.seed.unique())}, hardware {sorted(df.hardware.unique())}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/c1")
    p.add_argument("--figures_dir", default="figures/c1")
    args = p.parse_args()

    df = load(Path(args.results_dir))
    ok = df[df.capped == 0]
    ad = ok[ok.adaptive == 1]
    fx = ok[ok.adaptive == 0]

    print("\n=== ADAPTIVE: median over seeds (tolerance tightens downward) ===")
    print(f"{'solver':9} {'tol':>8} {'rel_err':>11} {'fwd_nfe':>8} {'bwd_nfe':>9} "
          f"{'time_ms':>9} {'recon':>7}")
    for s in ADAPTIVE:
        g = ad[ad.solver == s]
        if g.empty:
            continue
        for tol, gt in sorted(g.groupby("tol"), key=lambda kv: -kv[0]):
            print(f"{s:9} {tol:>8.0e} {gt.rel_err.median():>11.3e} "
                  f"{gt.fwd_nfe.median():>8.0f} {gt.bwd_nfe.median():>9.0f} "
                  f"{gt.fwd_time_s.median()*1e3:>9.2f} "
                  f"{int(gt.recon_ok.sum()):>3}/{len(gt):<3}")

    print("\n=== FIXED-STEP: median over seeds (cost axis = step count) ===")
    print(f"{'solver':9} {'steps':>8} {'rel_err':>11} {'fwd_nfe':>8} {'time_ms':>9} {'recon':>7}")
    for s in FIXED:
        g = fx[fx.solver == s]
        if g.empty:
            continue
        for n, gn in sorted(g.groupby("n_steps")):
            print(f"{s:9} {int(n):>8} {gn.rel_err.median():>11.3e} "
                  f"{gn.fwd_nfe.median():>8.0f} {gn.fwd_time_s.median()*1e3:>9.2f} "
                  f"{int(gn.recon_ok.sum()):>3}/{len(gn):<3}")

    print("\n" + "=" * 78)
    print("PRE-DECLARED REFUTATION CHECKS (raw numbers + condition; observation decides)")
    print("=" * 78)

    print("\n[R-C1a] error falls monotonically as tol tightens  (REFUTED if flat/increasing)")
    a_pass = True
    for s in ADAPTIVE:
        g = ad[ad.solver == s].groupby("tol").rel_err.median().sort_index(ascending=False)
        if g.empty:
            continue
        mono = _monotone(list(g.values), increasing=False)
        a_pass &= mono
        print(f"  {s:8} {' -> '.join(f'{v:.2e}' for v in g.values)}   monotone-down: {mono}")

    print("\n[R-C1b] cost rises monotonically as tol tightens  (REFUTED if flat/decreasing)")
    b_pass = True
    for s in ADAPTIVE:
        g = ad[ad.solver == s].groupby("tol").fwd_nfe.median().sort_index(ascending=False)
        if g.empty:
            continue
        mono = _monotone(list(g.values), increasing=True)
        b_pass &= mono
        print(f"  {s:8} NFE {' -> '.join(f'{v:.0f}' for v in g.values)}   monotone-up: {mono}")

    print("\n[R-C1c] forward time ~ NFE (Chen Fig 3b)  (REFUTED if Spearman rho < 0.9)")
    c_pass = True
    for s in ADAPTIVE + FIXED:
        g = ok[ok.solver == s]
        if len(g) < 3:
            continue
        rho = spearmanr(g.fwd_nfe, g.fwd_time_s).statistic
        c_pass &= bool(rho >= 0.9)
        print(f"  {s:8} rho = {rho:.3f}  (n={len(g)})")

    print(f"\n  => R-C1a {'HOLDS' if a_pass else 'REFUTED'} | "
          f"R-C1b {'HOLDS' if b_pass else 'REFUTED'} | "
          f"R-C1c {'HOLDS' if c_pass else 'REFUTED'}")

    print("\n[STOP-check] higher order should not be systematically WORSE at matched NFE")
    faithful = ok[ok.recon_ok == 1]
    for lo, hi in [("euler", "midpoint"), ("midpoint", "rk4"), ("bosh3", "dopri5"),
                   ("dopri5", "dopri8")]:
        a, b = faithful[faithful.solver == lo], faithful[faithful.solver == hi]
        if a.empty or b.empty:
            continue
        # compare at the closest matched NFE budget available to both
        lo_err = a.groupby("fwd_nfe").rel_err.median()
        hi_err = b.groupby("fwd_nfe").rel_err.median()
        worse = []
        for nfe, e_hi in hi_err.items():
            near = lo_err.index[(lo_err.index - nfe).to_series().abs().argsort()[:1]]
            if len(near) and e_hi > lo_err.loc[near[0]] * 10:
                worse.append((nfe, e_hi, near[0], lo_err.loc[near[0]]))
        verdict = "OK" if not worse else f"*** {len(worse)} cell(s) 10x worse ***"
        print(f"  order {ORDER[lo]} ({lo}) vs order {ORDER[hi]} ({hi}): {verdict}")
        if worse:
            print("      STOP-AND-FLAG: this contradicts standard numerical analysis, not "
                  "just Chen. Do not resolve alone.")

    fd = Path(args.figures_dir)
    fd.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for s in ADAPTIVE:  # Fig 3a analog: accuracy vs tolerance
        g = ad[ad.solver == s].groupby("tol").rel_err.median()
        if not g.empty:
            a1.plot(g.index, g.values, "o-", label=f"{s} (order {ORDER[s]})")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.invert_xaxis()
    a1.set_xlabel("solver tolerance"); a1.set_ylabel("relative error vs reference")
    a1.set_title("C1a: error falls as tolerance tightens"); a1.legend(fontsize=8)

    for s in ADAPTIVE + FIXED:  # Fig 3b analog: time vs NFE
        g = ok[ok.solver == s].groupby("fwd_nfe").fwd_time_s.median()
        if not g.empty:
            a2.plot(g.index, g.values * 1e3, "o-" if s in ADAPTIVE else "s--", label=s)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("forward NFE"); a2.set_ylabel("forward time (ms)")
    a2.set_title("C1b/c: cost is linear in NFE"); a2.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(fd / "solver_dynamics.png", dpi=120)
    print(f"\nWrote {fd / 'solver_dynamics.png'}")


if __name__ == "__main__":
    main()
