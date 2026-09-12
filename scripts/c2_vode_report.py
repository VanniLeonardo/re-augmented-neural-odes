"""C2 with the original's solver: evaluates the condition recorded in OVERNIGHT_LOG.md [12].

Reads results/c2_vode/, written by `make c2-vode`. Only cells passing the reconstruction and
gradient checks count. The comparison point is the committed dopri5 cell at the same tolerance.
"""
from __future__ import annotations
import argparse
import glob

import pandas as pd

COMMITTED_DOPRI5 = 123.25  # committed MNIST dopri5 1e-5 median ratio, adjoint at the forward tol
PRIMARY_TOL = 1e-5


def verdict(ratio: float) -> str:
    """The pre-declared condition on the median ratio at the primary tolerance."""
    if ratio <= 2:
        return "SUPPORTED"
    if ratio >= 20:
        return "REFUTED"
    return "PARTIAL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/c2_vode")
    files = sorted(glob.glob(f"{p.parse_args().results_dir}/c2_vode_*.csv"))
    if not files:
        print("c2_vode: no results yet (make c2-vode)")
        return
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    ok = df[(df.recon_ok == 1) & (df.grad_ok == 1) & (df.capped == 0)]
    print(f"{len(df)} cells, {len(ok)} pass both checks | fields {sorted(df.field.unique())}")
    if len(df) > len(ok):
        bad = df[~df.index.isin(ok.index)]
        print("Excluded (capped or failing a check), not used for the verdict:")
        print(bad[["field", "seed", "tol", "fwd_nfe", "bwd_nfe", "ratio", "recon_ok",
                   "grad_ok", "capped", "wall_s"]].to_string(index=False))
    if ok.empty:
        print("No cell passes both checks: recorded as data, no verdict.")
        return
    table = ok.groupby(["field", "tol"]).agg(seeds=("seed", "nunique"), fwd=("fwd_nfe", "median"),
                                             bwd=("bwd_nfe", "median"), ratio=("ratio", "median"),
                                             wall_s=("wall_s", "median"))
    print(table.round(2).to_string())
    primary = ok[(ok.field == "mnist") & (ok.tol == PRIMARY_TOL)]
    if primary.empty:
        print(f"No checked MNIST cell at {PRIMARY_TOL:.0e}: no verdict.")
        return
    med = primary.ratio.median()
    print(f"\nMNIST at {PRIMARY_TOL:.0e}: VODE implicit Adams {med:.2f} against dopri5 "
          f"{COMMITTED_DOPRI5} and the original's corrected 0.8  ->  {verdict(med)}")


if __name__ == "__main__":
    main()
