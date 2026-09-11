"""D3 long run: does a 20-epoch budget close the MNIST Neural ODE undershoot?

Evaluates the condition recorded in OVERNIGHT_LOG.md [9] before the run. Reads
results/d3_long/, written by `make d3-long`.
"""
from __future__ import annotations
from pathlib import Path

from scripts.image_table import load_shards

DUPONT_NODE, OURS_8EP = 96.4, 94.16  # Dupont Table 1; our 8-epoch, 5-seed value


def verdict(test_acc: float, train_acc: float) -> str:
    """The pre-declared condition, on mean accuracies in percent."""
    if test_acc >= 95.9 and train_acc > 94.3:
        return "SUPPORTED"
    if test_acc <= 94.7:
        return "REFUTED"
    return "PARTIAL"


def secondary(ep, fin) -> None:
    """Not part of the pre-declared condition: reported, labelled, never used for the verdict."""
    print("\n=== secondary, not pre-declared ===")
    m = ep.groupby("epoch").test_acc.mean() * 100
    first = m[m >= 95.9]
    print(f"Mean test accuracy at the training tolerance first reaches 95.9 at epoch "
          f"{first.index.min() if len(first) else 'never'}; peak {m.max():.2f} at epoch {m.idxmax()}.")
    t = (fin.pivot_table(index="seed", columns="eval_tol", values="test_acc_at_tol") * 100).round(2)
    ok = fin.pivot_table(index="seed", columns="eval_tol", values="recon_ok")
    print("Final-epoch test accuracy per seed at each tolerance (* = fails the check, unchecked):")
    print(t.astype(str).where(ok.eq(1), t.astype(str) + "*").to_string())


def main() -> None:
    df = load_shards(Path("results/d3_long"), "d3_trajectory")
    ep = df.drop_duplicates(["seed", "epoch"])
    print("=== NODE, mean over seeds, accuracy at the training tolerance ===")
    print((ep.groupby("epoch")[["test_acc", "train_acc"]].mean() * 100).round(2).assign(
        seeds=ep.groupby("epoch").seed.nunique()).to_string())

    last = df.epoch.max()
    fin = df[df.epoch == last]
    print(f"\n=== epoch {last}: reconstruction check per tolerance ===")
    print(fin.groupby("eval_tol").recon_ok.agg(passed="sum", seeds="count").to_string())
    passing = [t for t, g in fin.groupby("eval_tol") if g.recon_ok.eq(1).all()]
    if not passing:
        print("No tolerance passes on every seed. Recorded as data, no verdict.")
        secondary(ep, fin)
        return
    tf = max(passing)
    acc = fin[fin.eval_tol == tf].test_acc_at_tol * 100
    if acc.isna().any():
        print(f"\nRun incomplete at epoch {last}: accuracy is re-measured at the final epoch only.")
        return
    train = ep[ep.epoch == last].train_acc.mean() * 100
    e8 = ep[ep.epoch == 8].test_acc.mean() * 100
    print(f"\nSanity check, epoch-8 test accuracy at the training tolerance: {e8:.2f} "
          f"(must lie in 94.2 ± 1.0: {abs(e8 - 94.2) <= 1.0})")
    print(f"Epoch {last}, loosest tolerance passing on every seed {tf:.0e}: test "
          f"{acc.mean():.2f} ± {acc.std():.2f} % (n={len(acc)}), train {train:.2f} %")
    print(f"Gap to Dupont closed: {(acc.mean() - OURS_8EP) / (DUPONT_NODE - OURS_8EP):.0%} "
          f"of {DUPONT_NODE - OURS_8EP:.2f} pp  ->  {verdict(acc.mean(), train)}")


if __name__ == "__main__":
    main()
