# CHANGELOG — coursework code → ReScience C replication

Running log of every change made to the original Bocconi 30562 course project (`Group 5`)
while converting it into a ReScience C replication of **Dupont et al. 2019 (ANODE)** (primary)
and **Chen et al. 2018 (Neural ODEs)** claims C1–C4 (secondary). The paper's
"deviations from the original coursework" and "reproduction notes" sections will be written
from this file.

Format: newest first. Each entry: *what changed*, *why*, *scope tag* (`infra` / `experiment` /
`report` / `scope`), and the relevant plan item.

---

## Phase 1 — A1/A3 re-measurement: the stem/geometry/head factorial (2026-07-13)

### experiment (`report` + `experiment`)
- Re-ran the A1/A3 question **properly** (the user rejected the earlier 2-seed/accuracy-only
  claim): `scripts/run_stem_geometry_factorial.py` sweeps {geometry: circles, spheres} ×
  {stem, no-stem} × {head: linear, mlp-diagnostic}, **≥5 seeds**, logging **fwd/bwd NFE
  (final + peak), val loss, and divergence** (solver capped at `max_num_steps=1500` so a
  NODE-failure would show as divergence, not a hang). Committed CSV `results/factorial/`,
  table via `scripts/print_factorial_table.py`, `make factorial`.
- **Head audit (done first):** `ODENet.fc` is a **single `nn.Linear`** — already faithful to
  Dupont; the D8 readout-cheat does not recur. Recorded as `DEVIATIONS.md` A10; ruled out
  "head does the work" a priori. Added a diagnostic `head_hidden_dim` MLP knob.
- **`make_spheres`** (Dupont App. F.2.1 filled disk + annulus, 1000:2000) added + unit-tested.
- **RESULT — no NODE-failure at faithful settings.** All 25 runs converged (0 divergences);
  the data-space NODE + linear head separates **both** circles and spheres at ~99% at flat NFE
  (median ~42, peak ≤77 ≪ 1500 cap). Removing the stem *helps*; spheres is not harder; the MLP
  head needs less NFE (32 vs 42), confirming the head-does-the-work direction. **None of the
  three deviations reproduces a NODE-failure — there is none at these settings.**
- **What this does NOT do:** it does not refute Dupont's *comparative* ANODE-vs-NODE claim
  (lower/flatter NFE, held-out-slice generalization) — that is D1 (with ANODE) and D2
  (Fig 9, held-out slices), 500-epoch budget, in Stage C. iid-val accuracy is the wrong lens
  (aligns with plan §9). Full scoping in `DEVIATIONS.md` (finding note under section A).

### correction (`report`)
- **Retracted** the previous entry's claim that the no-stem NODE showed "NFE explosion" and
  that **geometry was "the load-bearing factor."** That was an artifact: the earlier 150-epoch
  run was on the **GPU while another user's 8 GB job was running** (contention → slowness that
  I misread as solver divergence). On CPU without contention, every cell converges at flat NFE.
  The "FINDING" bullet in the section below is superseded by the factorial result above.

---

## Phase 1 — Stage A corrections + Stage B (2026-07-13)

### model faithfulness (must precede experiments) — see `DEVIATIONS.md`
- **A1 — ODE now integrates in data space for the toy experiments.** Added `use_stem` to
  `ODENet` (Identity "stem" when False; `ode_dim = data_dim`). `train_anode_circles`,
  `train_anode_slice_circles`, `solver_ablation` now pass `use_stem=False`. Toy vf width
  `ode_hidden_dim` 64 → **32** (Dupont App. F.1.1). *Why:* the `Linear+Tanh` stem is a
  learned warp the paper's toy setup does not have.
- **FINDING (stem validation, 30 ep, 2 seeds, thin circles) — [SUPERSEDED / PARTLY RETRACTED;
  see the factorial re-measurement section above].** The stem-removal-doesn't-hurt observation
  held up, but the "geometry is the dominant/load-bearing factor" claim was **wrong** (it rested
  on a GPU-contended run misread as NFE explosion). The proper ≥5-seed factorial shows no
  NODE-failure on any cell. Data-space integration is kept regardless (it is faithful).
- **B1 — `ConvODEFunc` now injects time before every conv** (App. F.1.2), independently
  written (`_with_time` helper, state-first concat; not Dupont's `Conv2dTime` subclass). Was:
  a single time concat at the input (a different vector field). Also fixed a latent dtype bug
  in the time channel. ConvODENet param count changes accordingly (conv2/conv3 gain +1 input ch).
- **Item 3 — renamed `DiscreteResNet` → `EulerDiscretizedODENet`** ("weight-tied Euler ResNet")
  across code/docs; kept a deprecated alias. *Why:* it is a fixed-step Euler discretisation of
  the *same* field (weight-tied), not an independent ResNet — so "comparable accuracy" is close
  to tautological. Documented in `DEVIATIONS.md` D1.
- Added a `solver_options` passthrough (`ODEBlock`/`ODENet`) for fixed-step solving — enables
  the analytic NFE tests and the Stage-C C4 memory-vs-NFE experiment.

### docs (first-class ReScience deliverables)
- **`DEVIATIONS.md`** — full whole-codebase audit of where our implementation differs from the
  papers *as described* (geometry, widths, time-dependence, augmentation placement, solver/tol,
  optimiser, lr, batch, epochs, #runs), each with expected effect on the claim and status
  (RESOLVED / OPEN / INTENDED).
- **`PROVENANCE.md` trimmed to copying-provenance only** and cross-linked to `DEVIATIONS.md`
  (the two questions — "copied?" vs "matches the paper?" — are now cleanly separated).
- **Item 5 — replaced misplaced `extra/README.md` with root `OUT_OF_SCOPE.md`** (the excluded
  files are not physically in `extra/`); updated all references.
- **Item 2 recorded in the plan:** the C2 characterisation sweep now includes a **torchdiffeq
  version axis** (0.2.5 + earlier/later) — if the bwd≈fwd ratio holds only in 0.2.5 it is a bug
  report, not a finding.

### tests — Stage B (NFE split first, adversarial) — `pytest -m "not extra"` = 45 passed
- **`tests/test_nfe_split.py` (22 tests): the C2 counter is PROVEN correct.** Forward NFE is
  asserted *exactly* analytic for fixed-step solvers (euler=N, midpoint=2N, rk4=4N, verified
  against torchdiffeq 0.2.5); the adjoint backward for fixed-step rk4 is **exactly forward**
  (ratio 1.0, the C2 mechanism at unit level); `odeint`≡`odeint_adjoint` forward; the engine's
  snapshot recovers 4N/4N end-to-end; reset-per-forward makes cycles independent; a shared
  `ODEFunc` across two blocks sums (adversarial double-count check).
- **`tests/test_stage_b.py` (17 tests):** adjoint-vs-direct-backprop gradient agreement to
  **rtol 1e-5/atol 1e-6** for fixed-step rk4 (float64), *plus* a documented caveat test that
  **adaptive** dopri5 adjoint gradients legitimately differ from direct backprop
  (optimise-then-discretise); augmentation zero-init + shape; ODE-Net↔Euler param parity across
  widths {16,64,160} × depths {2,5,50} + depth-independence; CPU fixed-step seed determinism
  (with GPU/adaptive non-determinism documented).

### experiment — D8 crossing-flow (Dupont Fig 3 / Prop 1), pulled forward for the GPU gate
- `scripts/train_crossing_flow.py`. **Caught and fixed a faithfulness bug in our own script:**
  with a learnable linear readout `w·φ(x)+b` a 1-D NODE can cheat the crossing via `w<0`
  (NODE "succeeded" on 2/3 seeds). Dupont's Fig 3 is about the **flow itself**, so the demo now
  outputs the flow endpoint's data coordinate (readout-free, `CrossingFlow`).
- **RESULT (GPU, 5 seeds):** NODE MSE **1.0000 ± 0.0000** (the order-preserving 1-D flow cannot
  cross → collapses to 0), ANODE-p1 **0.0005 ± 0.0003** — a clean, faithful reproduction.

---

## Phase 1 — Stage A (P0 reproducibility infrastructure) — COMPLETE (2026-07-13)

### scope
- **Rubanova / Latent-ODE / ODE-RNN / sine / spiral CUT from the submission.** Per approved
  Phase-1 scope. Code is *not deleted*; it is excluded from `make reproduce-all` and from the
  gated test suite, and documented in `OUT_OF_SCOPE.md`. Reason: the paper's headline
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
- **`OUT_OF_SCOPE.md`** documents the excluded Rubanova material.

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
