"""§6 missing-slice GRID — augmentation p × wedge width × geometry × seed (extends Fig 9).

Dupont's Fig 9 shows one wedge ([0, π/5]) on one geometry. REPLICATION_PLAN.md §6 makes it
systematic: does ANODE's advantage on an unobserved region grow with the size of that region,
depend on how much augmentation is given, and survive a second geometry? Every cell is one
from-scratch training run of `scripts.run_missing_slice.run` (accurate tolerance, recon
check, slice / observed-region / full validation), so the grid inherits D2's faithfulness.

Cells run in parallel on CPU (the 2-D toy is faster on CPU than GPU, one thread per worker).
Only the parent process writes the CSV, so there are no races. RESUMABLE: a cell has no
intermediate state -- it is either finished (a row exists) or absent -- so re-running skips
finished cells and retries the rest.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # inherited by spawned workers before torch imports

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cell(job: tuple) -> dict:
    geometry, frac, p, seed, base = job
    import torch
    torch.set_num_threads(1)
    from scripts.run_missing_slice import run
    cfg = argparse.Namespace(**base, geometry=geometry, width=frac * math.pi, start=0.0)
    name = "NODE" if p == 0 else f"ANODE-p{p}"
    t0 = time.perf_counter()
    row = run(name, p, seed, cfg)
    row["wall_s"] = round(time.perf_counter() - t0, 1)
    return row


def _key(geometry, frac, model, seed) -> tuple:
    return (geometry, round(float(frac), 4), model, int(seed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometries", default="spheres,circles")
    ap.add_argument("--widths", default="0.125,0.2,0.333333", help="wedge width / pi")
    ap.add_argument("--augs", default="0,1,2,3,5", help="augmented dims (0 = NODE)")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_val", type=int, default=3000)
    ap.add_argument("--train_tol", type=float, default=1e-6)
    ap.add_argument("--eval_tol", type=float, default=1e-6)
    ap.add_argument("--recon_thresh", type=float, default=1e-2)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--results_dir", default="results/slice_grid")
    a = ap.parse_args()

    out = Path(a.results_dir) / "slice_grid.csv"
    done = set()
    if out.exists():
        with out.open(newline="") as h:
            done = {_key(r["geometry"], r["width_over_pi"], r["model"], r["seed"])
                    for r in csv.DictReader(h)}
    base = {k: getattr(a, k) for k in ("epochs", "n_val", "train_tol", "eval_tol", "recon_thresh")}
    jobs = [(g, float(w), int(p), int(s), base)
            for g in a.geometries.split(",") for w in a.widths.split(",")
            for p in a.augs.split(",") for s in a.seeds.split(",")]
    todo = [j for j in jobs if _key(j[0], j[1], "NODE" if j[2] == 0 else f"ANODE-p{j[2]}", j[3])
            not in done]
    print(f"slice grid: {len(jobs)} cells, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run on {a.jobs} workers -> {out}", flush=True)

    t0, failed = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_cell, j): j for j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                row = f.result()
            except Exception as exc:  # one bad cell must not sink the grid; it is retried on resume
                failed += 1
                print(f"  [FAILED] {futs[f][:4]}: {type(exc).__name__}: {exc}", flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            new = not out.exists()
            with out.open("a", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"  [{i}/{len(todo)}] {(time.perf_counter() - t0) / 60:.1f} min elapsed",
                  flush=True)
    print(f"done: {len(todo) - failed} cells written, {failed} failed "
          f"({(time.perf_counter() - t0) / 3600:.2f} h)", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
