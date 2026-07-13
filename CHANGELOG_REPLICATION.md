# CHANGELOG — coursework code → ReScience C replication

Running log of every change made to the original Bocconi 30562 course project (`Group 5`)
while converting it into a ReScience C replication of **Dupont et al. 2019 (ANODE)** (primary)
and **Chen et al. 2018 (Neural ODEs)** claims C1–C4 (secondary). The paper's
"deviations from the original coursework" and "reproduction notes" sections will be written
from this file.

Format: newest first. Each entry: *what changed*, *why*, *scope tag* (`infra` / `experiment` /
`report` / `scope`), and the relevant plan item.

---

## Phase 1 — Stage A (P0 reproducibility infrastructure) — in progress (2026-07-13)

### scope
- **Rubanova / Latent-ODE / ODE-RNN / sine / spiral CUT from the submission.** Per approved
  Phase-1 scope. Code is *not deleted*; it is excluded from `make reproduce-all` and from the
  gated test suite, and documented in `extra/README.md`. Reason: the paper's headline
  benchmarks (PhysioNet / MuJoCo / Human Activity) are unreached, and the 2-D spiral is a
  Chen-2018 experiment, not a Rubanova one — so no replication claim is possible at our scope.
- **Chen faithful conv Table-1 row dropped**; our MNIST rows are relabelled as our own seeded
  baselines. Chen's contribution to the submission = C1–C4 only.
- **ANODE MNIST (D3) kept core; CIFAR-10 (D4) gated to Stage D; SVHN/ImageNet out.**
- **C2 (backward/forward NFE ratio) promoted to the headline result** — pending a
  proven-correct NFE-split unit test and a dedicated characterisation sweep.

### infra — Stage A (P0) landed and verified
- `REPLICATION_PLAN.md` revised: shrunken scope banner, resolved decisions D1–D3, C2 promotion,
  A/B/C/D execution order, budget re-costed for the actual RTX 3090 (~30–55 GPU-h core).
  Working branch: `rescience-c-replication` (off `main`).
- **Pinned environment + container.** `environment.yml` now pins exact versions
  (torch 2.5.1 / torchvision 0.20.1 / pytorch-cuda 12.1 / numpy 2.4.3 / scipy 1.17.1 /
  scikit-learn 1.8.0 / matplotlib 3.10.9 / pandas 3.0.3 / rich 15.0.0 / pytest 9.0.3;
  torchdiffeq 0.2.5 via pip). Added `requirements.txt` (pip pins), `requirements-lock-cpu.txt`
  (fully-resolved lock from the container), a CPU `Dockerfile`, and `.dockerignore`.
  *Why:* the old file floated everything except python + pytorch-cuda → non-reproducible clone.
  **pandas** was declared but never installed in the coursework env; now pinned/installed.
- **Pluggable logging backend** `training/logging_backend.py` (CSV default / `none` / `wandb`).
  Removed the load-bearing top-level `import wandb` from the core path (notably
  `training/utils.py`, reached via `NFEStats`) and swapped `wandb.init/log/finish` for the
  logger in all core scripts (`train_continuous_mnist`, `train_discrete_mnist`,
  `train_anode_circles`, `train_anode_slice_circles`, `solver_ablation`, `plot_fig3`,
  `utils.py`). `wandb` is now imported lazily only inside the `wandb` backend.
  *Why:* a fresh clone with no W&B login previously prompted/hung before any work.
- **Seeding.** Added `set_seed()` (python/numpy/torch/cuda, optional deterministic) and a
  `--seed` arg to the MNIST scripts (previously **unseeded**); `get_mnist_dataloaders` now
  takes a `seed` and uses a seeded shuffle generator. ANODE/solver scripts routed through
  `set_seed`. *Why:* Tables 1/7 were single unseeded runs.
- **Fixed `train_discrete_mnist.py` CPU crash** (`KeyError('memory_mb')`): both MNIST scripts
  were rewritten to use `.get("memory_mb", 0.0)` and to write per-run summary CSVs
  (`results/mnist/*.csv`). Verified `mem: 0.0 MB` (no crash) on CPU in the container.
- **Automated ANODE aggregation** `scripts/aggregate_anode_results.py` (groupby `model_name`,
  mean/std → deterministic filenames `results/anode/final/{circles,slice}_aggregated.csv`).
  Rewired `plot_anode_corrected_results.py` to read those (auto-aggregating if missing) instead
  of the hand-made, cluster-job-id-named CSVs. *Why:* the figures were previously not
  regenerable without a manual step; `results/anode/final/` was never committed.
- **SLURM genericised.** All 4 `.slurm` files: `CHANGE_ME_ACCOUNT`/`CHANGE_ME_PARTITION`
  placeholders, portable `cd "${SLURM_SUBMIT_DIR:-…}"`, `conda activate "${CONDA_ENV:-neural_odes}"`,
  removed personal W&B agent `gaiagrossi-bocconi-university/NeuralODEs/cedblosn`; headed
  "ILLUSTRATIVE — not required to reproduce".
- **pytest hygiene.** Removed the two `scripts/test_*.py` smoke scripts (pytest collected them
  and they downloaded MNIST at collection time) → `scripts/smoke_*.py` with `__main__` guards.
  Added `pytest.ini` (`testpaths=tests`, `norecursedirs`, `extra` marker); marked the Rubanova
  time-series tests `extra` so the gated suite is `pytest -m "not extra"`.
- **`.gitignore`** un-ignores committed artifacts (`results/**/*.csv|json`) while ignoring bulk
  logs (`results/logs/`) and smoke scratch (`.smoke/`, `results/smoke/`).
- **Code hygiene.** Deleted empty `models/discrete.py`; removed dead imports
  `from logging import config` (`models/continuous.py`) and `from email import generator`
  (`data/synthetic.py`).
- **`Makefile`** with `smoke` (<5 min, no GPU), `test`, `reproduce-all`, per-artifact targets
  (`table2`/`table3`/`solver-ablation`/`mnist-baselines`/`fig3`/`anode-figures`), `docker-smoke`,
  `clean`. A `NODE_MAX_BATCHES` env cap (read in `training/engine.py`) bounds MNIST for smoke.
- **`PROVENANCE.md`** written (per-module written-from-paper / library-dependency verdicts).
- **`extra/README.md`** documents the excluded Rubanova material.

### Stage A acceptance gate — VERIFIED
Built the CPU Docker image and ran `make smoke` inside it (no GPU, **wandb not installed**,
network only for the MNIST download):
```
docker build -t neural-odes-repro .
docker run --rm neural-odes-repro           # CMD = make smoke
```
Result: all 6 pipeline stages ran to completion in ~30 s; MNIST downloaded at runtime;
CPU memory reported `0.0 MB` (no `KeyError`); `pytest` = 5 passed, 1 skipped (CUDA), 16
deselected (extra); "SMOKE OK". Also passes locally (`make smoke`, GPU) in ~38 s.

---

## Baseline — original coursework state (commit `fb96460`, "Final Commit")

Recorded for reference; not a change. Key facts established during the Phase-0 audit:
- Env: conda `neural_odes` — python 3.11.15, torch 2.5.1, torchvision 0.20.1, torchdiffeq 0.2.5,
  numpy 2.4.3, scikit-learn 1.8.0, matplotlib 3.10.9, wandb 0.26.1, rich 15.0.0, pytest 9.0.3.
  `environment.yml` pinned only python + pytorch-cuda; everything else floated. **pandas was
  declared but not installed.** Hardware: RTX 3090 (24 GB).
- Provenance: all modules written-from-paper; torchdiffeq used only as the solver library
  (see `PROVENANCE.md`).
- Known reproducibility defects targeted by Stage A: unpinned deps; MNIST scripts unseeded;
  `wandb` load-bearing (top-level import incl. `training/utils.py` on the core path;
  online-default `wandb.init` in several scripts); SLURM files hardcoding account `3263572`,
  `/home/3263572/...`, and a personal W&B agent; ANODE figures requiring a manual hand-aggregation
  step with cluster job-IDs in filenames (`results/anode/final/` absent, uncommitted);
  `scripts/test_{continuous,discrete}_mnist.py` collected by pytest and downloading MNIST at
  collection time; `train_discrete_mnist.py` `KeyError('memory_mb')` on CPU; empty
  `models/discrete.py`; dead imports (`from logging import config`, `from email import generator`).
