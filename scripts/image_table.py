"""Shared reporting rule for the image experiments, MNIST and CIFAR-10.

Accuracy and cost are quoted at the loosest tolerance where both models pass the
reconstruction check on every seed at the final epoch. Accuracy is re-measured at that
tolerance rather than taken from the training tolerance. Keeping the rule in one place
stops the two experiments from drifting apart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

KEY = ["model", "seed", "epoch", "eval_tol"]


def load_shards(results_dir: Path, stem: str) -> pd.DataFrame:
    """Merge per-seed shards (`<stem>_s<seed>.csv`); refuse duplicated cells."""
    shards = sorted(results_dir.glob(f"{stem}_s*.csv"))
    if not shards:
        raise SystemExit(f"no {stem}_s*.csv shards under {results_dir}")
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    if df.duplicated(KEY).any():
        raise SystemExit(f"duplicate {KEY} rows across shards in {results_dir} -- refusing "
                         "to report a double-counted seed")
    hw = sorted(df.hardware.unique()) if "hardware" in df else ["unrecorded"]
    print(f"loaded {len(shards)} shard(s), {len(df)} rows, seeds {sorted(df.seed.unique())}, "
          f"hardware {hw}")
    return df


def per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, seed, epoch): per-epoch metrics repeat across ladder rungs."""
    return df.drop_duplicates(["model", "seed", "epoch"])


def _ms(x: pd.Series) -> str:
    return f"{x.mean() * 100:6.2f} ± {x.std(ddof=1) * 100:4.2f}"


def faithful_table(df: pd.DataFrame, nfe_col: str, dupont: dict, fig_dir: Path,
                   replicate: tuple[Path, str] | None = None) -> None:
    last = df.epoch.max()
    fin = df[df.epoch == last]
    arms = sorted(fin.model.unique(), key=lambda m: m != "NODE")
    node, anode = arms[0], arms[1]
    train_tol = float(fin["train_tol" if "train_tol" in fin else "tol"].iloc[0])

    print(f"\n=== final epoch {last}: accuracy RE-MEASURED at each tolerance ===")
    print(f"{'model':10} {'eval_tol':>8} | {'recon':>5} | {'acc % (mean ± sd)':>17} | {'NFE med':>7}")
    for m in arms:
        for t, g in sorted(fin[fin.model == m].groupby("eval_tol"), key=lambda kv: -kv[0]):
            nfe = g[g[nfe_col] > 0][nfe_col].median()
            tag = "  <- train tol" if t == train_tol else ""
            print(f"{m:10} {t:>8.0e} | {int(g.recon_ok.sum())}/{len(g):<3} | {_ms(g.test_acc_at_tol):>17} "
                  f"| {nfe:7.0f}{tag}")

    both = sorted(t for t, g in fin.groupby("eval_tol")
                  if all(g[g.model == m].recon_ok.eq(1).all() for m in arms))
    if not both:
        print("\nNO tolerance has BOTH arms recon_ok on every seed -- nothing reportable as "
              "faithful. Recorded as data; no headline.")
        return
    tf = max(both)
    at = {m: fin[(fin.model == m) & (fin.eval_tol == tf)] for m in arms}
    print(f"\nLoosest tolerance with BOTH arms recon_ok on every seed: {tf:.0e}  <- headline")

    print(f"\n[R-ACC1] accuracy at train tol {train_tol:.0e} vs at faithful tol {tf:.0e} "
          "(REFUTED if |delta| >= 0.5 pp)")
    base = fin[fin.eval_tol == train_tol]
    if base.empty:
        print("  train tolerance is not a ladder rung -- R-ACC1 cannot be evaluated")
    for m in arms:
        b = base[base.model == m].set_index("seed").test_acc_at_tol
        f = at[m].set_index("seed").test_acc_at_tol
        if b.empty:
            continue
        rec = int(base[base.model == m].recon_ok.sum())
        d = (f.mean() - b.mean()) * 100
        print(f"  {m:10} {b.mean()*100:.2f} -> {f.mean()*100:.2f}  delta {d:+.3f} pp "
              f"(worst seed {(f - b).abs().max()*100:.3f} pp; train-tol recon {rec}/{len(b)})  "
              f"{'HOLDS' if abs(d) < 0.5 else 'REFUTED'}")

    gap = (at[anode].test_acc_at_tol.mean() - at[node].test_acc_at_tol.mean()) * 100
    print(f"\n[R-ACC2] {anode} - {node} accuracy gap at {tf:.0e}: {gap:+.2f} pp  "
          f"{'HOLDS' if gap > 1 else 'REFUTED'} (REFUTED if <= 1 pp)")

    n_nfe = at[node][nfe_col].median()
    a_nfe = at[anode][nfe_col].median()
    print(f"\nPRE-DECLARED REFUTATION (ANODE >= NODE acc AND ANODE cheaper, at {tf:.0e}):")
    print(f"  acc  {anode} {at[anode].test_acc_at_tol.mean()*100:.2f} vs {node} "
          f"{at[node].test_acc_at_tol.mean()*100:.2f}  -> ANODE>=NODE {gap >= 0}")
    print(f"  NFE  {anode} {a_nfe:.0f} vs {node} {n_nfe:.0f}  -> ANODE<NODE {a_nfe < n_nfe}")
    ratios = [fin[(fin.model == node) & (fin.eval_tol == t)][nfe_col].median()
              / fin[(fin.model == anode) & (fin.eval_tol == t)][nfe_col].median() for t in both]
    print(f"  NODE/ANODE NFE ratio across every faithful rung {[f'{t:.0e}' for t in both]}: "
          f"{min(ratios):.2f}x - {max(ratios):.2f}x (report the range, not one number)")

    print(f"\nvs Dupont Table 1 (raw, untuned; a miss is a partial-replication finding):")
    for m in arms:
        d, s = dupont[m]
        print(f"  {m:10} ours {_ms(at[m].test_acc_at_tol)} %  | Dupont {d} ± {s}  | "
              f"diff {at[m].test_acc_at_tol.mean()*100 - d:+.2f} pp")

    if replicate and replicate[0].exists():
        r = pd.read_csv(replicate[0])
        r = per_epoch(r[r.epoch == r.epoch.max()])
        print(f"\nREPLICATE run ({replicate[1]}; accuracy at train tol, NOT recon-checked) -- "
              "shows run-to-run spread:")
        for m in arms:
            print(f"  {m:10} {_ms(r[r.model == m].test_acc)} %")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, c in zip(arms, ["tab:red", "tab:blue"]):
        g = fin[fin.model == m].groupby("eval_tol")
        acc = g.test_acc_at_tol.mean() * 100
        ok = g.recon_ok.mean()
        ax.plot(acc.index, acc.values, "-", color=c, label=m)
        ax.scatter(acc.index[ok == 1], acc.values[ok == 1], color=c, marker="o", zorder=3)
        ax.scatter(acc.index[ok < 1], acc.values[ok < 1], facecolor="white", edgecolor=c,
                   marker="o", zorder=3)
    ax.axvline(tf, ls=":", c="green", label=f"faithful tol {tf:.0e}")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("evaluation tolerance"); ax.set_ylabel("final-epoch test accuracy (%)")
    ax.set_title("Accuracy vs tolerance (hollow = fails recon)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "acc_vs_tol.png", dpi=120)
    print(f"\nWrote {fig_dir / 'acc_vs_tol.png'}")
