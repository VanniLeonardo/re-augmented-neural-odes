"""Aggregate raw per-seed ANODE summary CSVs into flat mean/std tables.

Replaces the manual hand-aggregation step the coursework relied on (the plot
script used to read files named after one-off SLURM job ids, e.g.
``circles_corrected_flat_table_job492199.csv``, that were produced by hand and
never committed). This script groups the committed per-seed summary CSVs by
``model_name``, computes mean/std of every numeric metric, and writes
deterministic filenames so the Table 2/3 figures regenerate end-to-end.

Inputs  (written by the training scripts):
    results/anode/circles_summary.csv
    results/anode/slice_circles_summary.csv
Outputs (deterministic, consumed by plot_anode_corrected_results.py):
    results/anode/final/circles_aggregated.csv
    results/anode/final/slice_aggregated.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# Columns that identify a run rather than being metrics to average over seeds.
_ID_COLS = {"seed", "augment_dim", "num_parameters", "n_samples", "n_val_samples"}


def aggregate(summary_csv: Path, out_csv: Path, group_col: str = "model_name") -> Optional[pd.DataFrame]:
    """Group ``summary_csv`` by ``group_col`` and write mean/std of numeric metrics."""
    if not summary_csv.exists():
        print(f"[aggregate] skip: {summary_csv} not found")
        return None

    df = pd.read_csv(summary_csv)
    if group_col not in df.columns:
        raise ValueError(f"{summary_csv} has no '{group_col}' column; columns={list(df.columns)}")

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c not in _ID_COLS
    ]
    grouped = df.groupby(group_col, sort=True)
    agg = grouped[numeric_cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = grouped.size().to_numpy()

    # Preserve augment_dim (constant within a model_name) for readable ordering.
    if "augment_dim" in df.columns:
        first_aug = grouped["augment_dim"].first().reset_index(drop=True)
        agg.insert(1, "augment_dim", first_aug)
        agg = agg.sort_values("augment_dim").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    print(f"[aggregate] {summary_csv.name} ({len(df)} rows) -> {out_csv} ({len(agg)} groups)")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ANODE per-seed CSVs.")
    parser.add_argument("--results_dir", type=str, default="results/anode")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    final_dir = results_dir / "final"

    aggregate(results_dir / "circles_summary.csv", final_dir / "circles_aggregated.csv")
    aggregate(
        results_dir / "slice_circles_summary.csv",
        final_dir / "slice_aggregated.csv",
    )


if __name__ == "__main__":
    main()
