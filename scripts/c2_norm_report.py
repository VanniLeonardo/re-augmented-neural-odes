"""C2 adjoint error norm: evaluates the condition recorded in OVERNIGHT_LOG.md [11].

Reads results/c2_norm/, written by `make c2-norm`. Only cells passing the reconstruction and
gradient checks count.
"""
from __future__ import annotations
import argparse
import glob

import pandas as pd

COMMITTED_DEFAULT = 123.25  # committed MNIST dopri5 1e-5 median ratio, adjoint at the forward tol


def verdict(reduction: float) -> str:
    """The pre-declared condition on default/flat, the reduction factor of the median ratio."""
    if reduction >= 5:
        return "SUPPORTED"
    if reduction < 2:
        return "REFUTED"
    return "PARTIAL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/c2_norm")
    files = sorted(glob.glob(f"{p.parse_args().results_dir}/c2_diagnosis_*.csv"))
    if not files:
        print("c2_norm: no results yet (make c2-norm)")
        return
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    ok = df[(df.recon_ok == 1) & (df.grad_ok == 1) & (df.capped == 0)]
    print(f"{len(df)} cells, {len(ok)} pass both checks | fields {sorted(df.field.unique())}")
    table = ok.groupby("adj_norm").agg(seeds=("seed", "nunique"), fwd=("fwd_nfe", "median"),
                                       bwd=("bwd_nfe", "median"), ratio=("ratio", "median"))
    print(table.round(2).to_string())
    bad = df[~df.index.isin(ok.index)]
    if len(bad):
        print("Excluded (fail a check), not used for the verdict:")
        print(bad.groupby(["seed", "adj_norm"]).ratio.median().round(2).unstack().to_string())
    if not {"default", "flat"} <= set(table.index):
        print("default or flat missing: no verdict")
        return
    med = table.ratio
    if set(df.field) == {"mnist"}:
        print(f"Sanity check: default {med['default']:.2f} against committed {COMMITTED_DEFAULT} "
              f"(within 10%: {abs(med['default'] / COMMITTED_DEFAULT - 1) <= 0.10})")
    red = med["default"] / med["flat"]
    print(f"default/flat reduction x{red:.1f} -> {verdict(red)} | flat reaches <= 2: "
          f"{med['flat'] <= 2}" + (f" | seminorm {med['seminorm']:.2f}" if "seminorm" in med else ""))


if __name__ == "__main__":
    main()
