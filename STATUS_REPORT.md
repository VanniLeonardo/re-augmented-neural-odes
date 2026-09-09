# STATUS_REPORT.md — orientation audit, 2026-09-09

> **⚠️ SUPERSEDED — point-in-time audit, 2026-09-09.** This was the orientation report
> written before the follow-up work. Several findings below have since been fixed and no
> longer describe the repository: D4/CIFAR-10 and C1 are now complete, D8 is recon-checked,
> the README documents the replication, and `make figures` / `make reproduce-all` are wired
> up. Kept as the record of the state the conversion was picked up in. For current state
> see `README.md`, `OVERNIGHT_LOG.md` and `DEVIATIONS.md`.

Read-only session. No experiments run, no code changed, no commits. Every number below was
read off the repository or produced by re-running a committed *report* script over committed
CSVs; nothing was re-trained.

**Headline:** the science is in better shape than the packaging. Six of the eight approved
experiments are committed and recon-faithful. The two that are not are **D4/CIFAR-10 (dead,
~40% of one arm, uncommitted)** and **Chen C1 (never run at seeds)**. The largest single
risk to a ReScience submission is not an experiment — it is that `README.md` and
`make reproduce-all` still describe the coursework, so a reviewer who clones this repo
cannot reach any of the replication results.

---

## 1. Where the work actually is

| | |
|---|---|
| Branch | `rescience-c-replication` (checked out), 17 commits ahead of `main` |
| Diverged from `main`? | No divergence — strictly ahead. `main` is at `fb96460 Final Commit` (coursework), fully contained in HEAD. |
| **Unpushed** | **All 17 replication commits.** `origin/rescience-c-replication` is at `cad93f1` (Stage A/B infra only). Everything from `7d77b67` (A1/A3 factorial) through `085f076` (D3 faithfulness fix) exists **only on this disk.** |
| Stashes | None |
| Working tree | Not clean — see below |

### Uncommitted working-tree contents

| Path | Status | Assessment |
|---|---|---|
| `OVERNIGHT_LOG.md` | modified, +39 lines | The D4 pre-declaration, the D4 probe/cap projection, and the "D4 interrupted then restarted" note. **This is the only record that D4 was ever attempted.** |
| `results/d4/d4_trajectory.csv` | untracked | The partial D4 run (see §3). |
| `data/cifar-10-batches-py/` | untracked, **178 MB** | Extracted CIFAR-10. **Not covered by `.gitignore`** — `.gitignore:238` ignores `*.tar.gz` (so the 163 MB tarball is safe) and `:211` ignores `data/MNIST/`, but nothing ignores the extracted CIFAR directory. A `git add -A` would commit 178 MB of dataset. |
| `Final_Report_Group5.pdf` | untracked, 3.6 MB | Coursework report. Deliberate or not, it is unignored and would be swept up the same way. |

> **Single most urgent item, and it is not scientific:** 17 commits of work exist on one
> machine, unpushed, and the only copy of the D4 attempt is an untracked file next to an
> untracked 178 MB dataset. This is a backup problem before it is a submission problem.

---

## 2. Docs vs disk reconciliation

### Documents are stale in the *safe* direction (they under-claim)

`DEVIATIONS.md` is the best-maintained document (last touched at the D3 fix, `085f076`) and its
Dupont sections are accurate. Its Chen section is not:

| Doc location | Asserts | Reality |
|---|---|---|
| `DEVIATIONS.md` row C2 | "**OPEN** — recharacterise across tolerance" | **Done and committed** (`9cbdfaa`, `e6c107a`): `results/c2/c2_recharacterise.csv`, `c2_surface.csv`. |
| `DEVIATIONS.md` row C3 | "**OPEN** — Stage C C4" | **Done and committed** (`2b3d053`): `results/c4/c4_memory_vs_nfe.csv`. |
| `DEVIATIONS.md` row D2 (§D) | "**OPEN** — Stage C for D3" (batch 256) | **Done** — D3 ran at batch 256. |
| `Makefile` help text | "(Stage C **will add**: C2 sweep, C4 memory-vs-NFE, D3 ANODE-MNIST, D8 crossing-flow, missing-slice grid.)" | All but the grid are done; the help text was never updated. |

### Documents that assert something the repository does not support

| Doc location | Asserts | Reality | Severity |
|---|---|---|---|
| `README.md:35` | "`make reproduce-all` — **Full reproduction at paper settings (GPU)**" | `reproduce-all` = `mnist-baselines table2 table3 solver-ablation`. It reproduces **none** of D1, D2, D3, D4, C1, C2, C3, C4. It reproduces the *coursework* artifacts. | **High** |
| `README.md` body (§§134–387) | Documents MNIST baselines, circles, solver ablation, **"Reproducing the standalone irregular time-series diagnostic"**, **"Reproducing the Latent ODE encoder comparison"** | README was last committed `cad93f1` (2026-07-13), *before all experimental work*. It presents the **cut Rubanova material** as a headline reproduction path and documents none of the replication experiments. | **High** |
| `REPLICATION_PLAN.md:80,263`; `Makefile:9` | Rubanova code "excluded via `extra/`", "see `extra/README.md`" | **`extra/` does not exist.** `CHANGELOG_REPLICATION.md:131` records the decision to replace it with root `OUT_OF_SCOPE.md` and says "updated all references" — three references were missed. (`pytest.ini` also lists `extra` in `norecursedirs`, harmless.) | Low |
| `REPLICATION_PLAN.md:28` vs `:289` | "~40–55 GPU-hours" vs "~30–55 GPU-hours" | Internal inconsistency. | Trivial |

### Claim-ID drift (a real hazard for the write-up)

`REPLICATION_PLAN.md` §3.B defines the canonical IDs: **C1** = solver dynamics (Fig 3a–b),
**C2** = bwd/fwd NFE ratio (Fig 3c), **C3** = NFE grows during training (Fig 3d), **C4** =
O(1) memory. `OVERNIGHT_LOG.md` consistently labels the MNIST NFE-over-training work
"**C1/C3**". Under the canonical IDs that work is **C3 only**. Nothing in the repo covers
C1 at seeds. The "C1/C3" label makes a missing experiment look covered — see §3.

### Every committed CSV has a working script; no figures are committed (by design)

- `git ls-files figures` → **empty**. `.gitignore:236` ignores `figures/`, and `.gitignore:222`
  states the intent: CSV/JSON are the canonical artifacts, figures are regenerated. This is
  defensible for ReScience *provided* the regeneration is one command — which it currently is not (§5).
- I re-ran every report/figure script against committed data. **All succeeded**, reproducing the
  documented numbers: `d1_report`, `d2_report`, `plot_d2`, `d3_report`, `d3_faithful_report`,
  `plot_c4`, `plot_c2_surface`, `plot_c2_recharacterise`, `plot_mnist_nfe`, `plot_mnist_stiffening`.
- **Orphan on disk, in no doc:** `results/topology/topology_diagnostics.csv` (1 row) — only
  consumer is its own generator `scripts/topology_diagnostics.py`. Either cite it or drop it.
- **Superseded but committed:** `results/factorial/` — no recon column, no tolerance column. Its
  verdict is explicitly **WITHDRAWN** in `DEVIATIONS.md` A1/A3 as a loose-tolerance artifact.
  Correct to keep as provenance; **must never be cited as a result.**

---

## 3. Experiment status table

Recon column reads: fraction of rows with `recon_ok=1`, **at the decision point** (final epoch /
reported budget), which is what determines whether a headline is faithful.

### Dupont 2019 (primary)

| ID | Experiment | Committed? | Seeds | Tolerance | `recon_ok` at the reported point | Verdict |
|---|---|---|---|---|---|---|
| **D1** | Toy separation / NFE growth (`results/budget/`, `budget_circles/`) | ✅ | 5 | train+eval **1e-6** | **circles 50/50** all budgets. **spheres 57/60** — NODE **4/5** at budgets 200/500/1000 (one stiff seed) | **Clean**, with a one-seed caveat already flagged in the log. ANODE cheaper (170 vs 218 @50) and flat (×1.04 vs ×1.74 over 25→500). |
| D1′ | Stem×geometry×head factorial (`results/factorial/`) | ✅ | 5 | **none recorded** | **no recon column** | **Withdrawn** (loose-tol artifact). Provenance only. |
| **D8** | 1-D crossing flow (`results/crossing/`) | ✅ | 5 NODE + 5 ANODE-p1 | **none recorded** | **no recon column** | ⚠️ **NOT submission-clean.** Result is clean-looking (NODE MSE ≈1.0 = fails; ANODE ≈1e-3 = solves) but carries **no tolerance and no reconstruction check**, and has **no report or figure script** — the only consumer of the CSV is the training script that wrote it. Violates the project's own standing rule. |
| **D2** | Missing-slice / Fig 9 (`results/slice_spheres/`) | ✅ | 5 | train+eval recorded | **10/10** | **Clean.** NODE held-out slice acc 0.619, loss 6.089; ANODE 1.000 / 0.000. Wedge `[0, π/5]` verified against paper p.6 (`f8d1949`), correcting an earlier π/13 misread. |
| §6 | Missing-slice **grid** (p × width × 2 geometries) | ❌ | — | — | — | **Not started.** P2 "strengthens", not a blocker. |
| **D3** | Matched-param MNIST (`results/d3/`, `results/d3_faithful/`) | ✅ | 5 | acc @1e-5; NFE @ ladder 1e-5/1e-6/1e-7 | see below | **Clean — question (a) is ANSWERED.** |
| **D4** | Matched-param CIFAR-10 (`results/d4/`) | ❌ **untracked** | **3.9 of 5, NODE only** | ladder 1e-5/1e-6/1e-7 | 117/117 ✅ | ⚠️ **NOT DONE.** See below. |

### Chen 2018 (secondary), canonical IDs from `REPLICATION_PLAN.md` §3.B

| ID | Experiment | Committed? | Seeds | Tolerance | `recon_ok` | Verdict |
|---|---|---|---|---|---|---|
| **C1** | Solver dynamics: error ↓, time ∝ NFE as tol tightens (Fig 3a–b) | ❌ | — | — | — | **NOT DONE.** No committed CSV anywhere; no `figures/` output. `solver_ablation.py` and `plot_fig3.py` exist and run (the smoke pipeline exercises 3 configs), but the plan's requirement — **≥5 seeds, persisted CSV, restored atol/rtol** — was never executed. The `results/logs/solver_*.csv` files on disk are untracked smoke leftovers (2 rows each). Masked by the "C1/C3" labelling. |
| **C2** | bwd/fwd NFE ratio (Fig 3c) | ✅ `results/c2/`, `c2_circles/` | 5 (recharacterise) | swept **1e-3…1e-8** | 42/60 across the *whole* sweep — **by design**: the loose-tol rows are the finding. At 1e-5/1e-6/1e-7 recon_ok = 1.0; at 1e-4 it is 0.2; at 1e-3, 0.0. | **Clean, and correctly reframed.** Chen's "½" is **not** reproduced; the honest result is that the ratio is a tolerance×field surface (≈13–121×). Headline **withdrawn** and demoted out of the abstract by co-author decision. This is a legitimate negative replication finding, not a failure. |
| **C3** | NFE grows during training (Fig 3d) | ✅ `results/mnist_nfe/` (10 ep) + `results/mnist_stiffening/` (6 ep) | 5 | 1e-5 only / ladder 1e-5,1e-6,1e-7 | see below | **Clean via the stiffening run; the 10-epoch curve is not.** |
| **C4** | O(1) memory vs NFE (`results/c4/`) | ✅ | **3** | swept 1e-3…1e-7 | **15/15** faithful | **Clean result, below the project's own seed floor.** Adjoint **+0.0007 MB/NFE** (flat) vs direct backprop **+31.8 MB/NFE** (442 MB constant vs 689→3185 MB). The standing rule is ≥5 seeds; this has 3. |

#### D3 detail — question (a): **yes, the re-measure happened, and it is clean**

`results/d3_faithful/` (commit `085f076`), final epoch, 5 seeds:

| eval_tol | NODE median NFE | NODE recon_ok | ANODE-p5 median NFE | ANODE recon_ok |
|---|---|---|---|---|
| 1e-5 | 86 | **2/5** | 50 | 5/5 |
| 1e-6 | 146 | **3/5** | 92 | 5/5 |
| **1e-7** | **482** | **5/5** | **230** | **5/5** |

- **Loosest tolerance where BOTH are recon-faithful: 1e-7.**
- **Faithful NODE NFE = 482; ANODE = 230; ratio = 2.10×** (was 1.7× at the unfaithful 1e-5).
- The original `recon_ok` 2/5 problem is resolved *by measurement, not by assumption* — and the
  correction moved the number **against** the convenient direction at first glance (NODE got more
  expensive), which is the right sign for a fix made in good faith.
- Accuracy (unchanged, @1e-5): **ANODE 98.18 ± 0.29** vs Dupont's 98.2 ± 0.1 — **matches**.
  **NODE 94.53 ± 0.44** vs Dupont's 96.4 ± 0.5 — **undershoots by ~1.9 pp**. Documented in
  `DEVIATIONS.md` B2 and attributed to the 8-epoch budget. This is a **partial replication**, and
  it is reported as one. It should stay reported as one.

#### C3 detail — the 10-epoch curve is not recon-clean, and the docs say so

`results/mnist_nfe/` was measured at a **single** eval tolerance (1e-5). Recon over the run is
22/50, and **at the final epoch it is 0/5** — by epoch 10 the conv field has stiffened past 1e-5
entirely. `OVERNIGHT_LOG.md:233` states this plainly ("**This is NOT a clean full-budget C1/C3
claim**", fully-faithful window = epochs 1–3). The follow-up `results/mnist_stiffening/`
(commit `f3930fb`) fixes it properly with a ladder: **83/90 recon_ok; at 1e-6 and 1e-7 it is 5/5
at every epoch**, and faithful NFE grows **384 → 738** over 6 epochs. So:

- the **claim** (NFE grows during training) is reproduced **faithfully** at 1e-7, 5 seeds, 6 epochs;
- the **10-epoch figure** is not faithful past epoch 3 and should not be the headline figure.

No dishonesty here — the docs are candid. But a reviewer reading `figures/mnist_nfe/nfe_over_training.png`
without the log will see a curve whose right-hand half is not integrating.

#### D4 detail — question (b): **started, died twice, never finished, never committed**

- **Gate:** the plan (§8) budgets D4 at **13–25 GPU-h**; the log declared an **18 GPU-h hard cap**.
  (Your brief said ~6 GPU-h — the repository does not record that figure anywhere.)
- **Probe result:** CIFAR downloaded (~47 min, network-throttled), timings measured, and the run
  **projected at ~3.5 GPU-h for the full 5 seeds × 10 epochs** — comfortably inside the cap.
- **First launch:** killed by a process teardown after NODE seeds 0–1 (57 rows). Correctly refused
  as a result.
- **Restart:** relaunched detached under `setsid` explicitly so a session boundary could not kill
  it. **It died anyway**, and the log has no entry for the second death — the machine move is the
  obvious suspect, but the repo does not record it.
- **What exists:** `results/d4/d4_trajectory.csv`, 117 rows. Loop order is model-outer/seed-inner
  (`run_d4_anode_cifar.py:116`), so this is NODE seeds 0, 1, 2 complete (10 epochs each) plus
  **seed 3 stopped at epoch 9**. **Zero ANODE rows.** There is no comparison here at all — only
  ~40% of the control arm.
- **The good news:** all **117/117 rows are recon_ok=1 at all three tolerances** (1e-5, 1e-6,
  1e-7) — CIFAR's NODE had not yet stiffened past 1e-5 by epoch 10, unlike MNIST's. The
  machinery worked; only the run died. NODE test acc reaches ~0.53 by epoch 10, which is
  *coincidentally* right at Dupont's 53.7 — **do not read anything into this**, it is one
  unfinished arm with no ANODE to compare against.
- **Cost to finish: ~3.5 GPU-h** (or ~2 h if the three complete NODE seeds are reused). CIFAR is
  already cached locally, so the 47-minute download will not repeat.
- **Infra hazard for the re-run:** `run_d4_anode_cifar.py` **appends** (`_append`, line 46) and has
  **no resume or idempotency logic**. Re-running the same command over the existing CSV will
  silently duplicate NODE seeds 0–2. Move or delete the partial file first.

---

## 4. Does it still reproduce on this machine?

**Verified by execution, not by reading.** All four checks pass.

| Check | Result |
|---|---|
| Pinned CPU torch resolves (`pip --dry-run`, pytorch CPU index) | ✅ `torch-2.5.1+cpu`, `torchvision-0.20.1+cpu` still fetchable |
| Remaining pinned PyPI stack resolves | ✅ all 8 pins available at the exact versions |
| Conda env `neural_odes` | ✅ **intact and working** — Python 3.11.15, torch 2.5.1 (cu121 build), torchvision 0.20.1, torchdiffeq 0.2.5, numpy 2.4.3, scipy 1.17.1, pandas 3.0.3, sklearn 1.8.0, matplotlib 3.10.9. CUDA available, GPU matmul verified. |
| `pytest -m "not extra"` (host, GPU) | ✅ **49 passed, 16 deselected** in 47.8 s |
| `make smoke` (host, GPU) | ✅ all 6 stages, `SMOKE OK`, exit 0 |
| `make docker-smoke` (container acceptance gate) | ✅ builds and runs, **48 passed, 1 skipped**, `SMOKE OK`, exit 0 |

### Environment: what changed, and an important caveat about this "experiment"

| | Docs record | Now | Δ |
|---|---|---|---|
| GPU | RTX 3090 (24 GB) | **RTX 3090 (24 GB)** | same |
| Driver / CUDA | driver 595, CUDA 12.1 | driver **595.84**, CUDA **13.2** runtime capability | driver same series; the *runtime* the driver advertises moved 12.1 → 13.2 |
| CPU | not recorded | AMD Ryzen 7 7700X, 16 threads, 62 GB RAM | **`REPLICATION_PLAN.md` §8 says "must be re-timed on this machine and reported in the paper" — the CPU/RAM were never recorded, and wall-clock timings are not yet collected for the paper.** |
| Python (system) | 3.11 | **3.12.3** | pinned env is unaffected (conda supplies 3.11.15) |

**Caveat — this was a weaker test than it looks.** The `neural_odes` conda environment *and* a
prebuilt `neural-odes-repro:latest` Docker image both survived the move. So this exercised
"does the existing environment still work", not "can a reviewer build it from scratch". The
`docker-smoke` run used **cached layers**, so it did not re-resolve the pins; I verified pin
resolution separately via `pip --dry-run`, which is the meaningful half. A genuine clean-room
test (`docker build --no-cache`) has **not** been run and is cheap — worth doing once before
submission.

**One real fragility found:** the container's MNIST download hits
`http://yann.lecun.com/exdb/mnist/…` → **HTTP 404**, then silently falls back to the
`ossci-datasets.s3.amazonaws.com` mirror and succeeds. That fallback is torchvision's, not
ours, so it works today — but reproduction depends on a third-party mirror and on network
access. Worth one sentence in the paper's reproduction notes.

Nothing broke. Nothing was fixed this session.

---

## 5. What is genuinely left

### A. Blocks submission

1. **Push the branch.** 17 commits, one machine, no remote copy. Also decide the fate of the
   untracked `results/d4/d4_trajectory.csv` and the +39 lines of `OVERNIGHT_LOG.md` — the only
   evidence D4 was attempted.
2. **Add `data/cifar-10-batches-py/` to `.gitignore`** before anyone runs `git add -A`. 178 MB.
3. **Rewrite `README.md`.** It is the coursework README. A reviewer's first action is to read it,
   and it currently routes them to cut Rubanova material and to `make reproduce-all`, which
   reproduces none of the submission. This is the single highest-leverage fix in the repo.
4. **Make the replication reproducible by `make`.** The standing rule ("every table/figure
   regenerable from committed CSVs via a script via a make target, zero manual steps") is
   currently violated. **No Makefile target exists for:** `run_missing_slice` (D2),
   `run_d3_anode_mnist`, `run_d3_faithful_nfe`, `run_mnist_nfe`, `run_mnist_stiffening`,
   `c2_recharacterise`, `c2_mnist_ratio`, `c4_memory_vs_nfe`, `run_d4_anode_cifar`, **nor for any
   of the ten report/figure scripts** (`d1_report`, `d2_report`, `plot_d2`, `d3_report`,
   `d3_faithful_report`, `plot_c4`, `plot_c2_surface`, `plot_c2_recharacterise`,
   `plot_mnist_nfe`, `plot_mnist_stiffening`). Since **no figures are committed**, a reviewer
   currently has no supported command that turns the committed CSVs into the paper's figures.
   All ten scripts work — they just are not wired up.
5. **Chen C1 is missing.** Solver dynamics (Fig 3a–b) at ≥5 seeds with a persisted CSV was never
   run. ~1–2 GPU-h per the plan. Either run it or drop C1 from the claimed scope explicitly —
   but the current state (claimed in scope, absent from disk, obscured by "C1/C3" labelling) is
   the worst of the three options.
6. **Decide D4.** It is ~3.5 GPU-h from done, CIFAR is cached, and the machinery is proven
   recon-faithful on the rows that exist. Either finish it (move the partial CSV aside first —
   the script appends and will duplicate) or formally drop it and delete the pre-declaration's
   implication that it is coming. Do not report the NODE-only partial.
7. **D8 crossing flow has no tolerance, no recon check, and no figure script.** By the project's
   own standing rule this is not a reportable result. It is cheap to re-run with the ladder.

### B. Would strengthen it

8. **Retire the C3 10-epoch caveat** by re-running `results/mnist_nfe/` at the 1e-6/1e-7 ladder,
   so the headline NFE-growth figure is faithful across its whole x-axis rather than epochs 1–3.
9. **C4 at 5 seeds** (currently 3, below the project's own floor). Cheap.
10. **Fix the stale `DEVIATIONS.md` statuses** (C2, C3, §D D2 all say OPEN for completed work) and
    the three dangling `extra/` references. Cheap, and these are documents a ReScience reviewer
    reads closely.
11. **Record hardware + wall-clock per experiment** — `REPLICATION_PLAN.md` §8 requires it for the
    paper and it has never been collected.
12. **One `docker build --no-cache`** to make the clean-room claim real.
13. `.github` CI running `make smoke` (P2-21); the §6 missing-slice grid (P2-18, ~10–20 GPU-h);
    D1 spheres' one non-integrating NODE seed at budgets ≥200.

### C. Write-up (separate deliverable, co-author, NOT started — untouched here)

The paper does not exist. Note for whoever writes it: **three results must be reported as
partial or negative, not smoothed** — D3's NODE accuracy undershoots Dupont by ~1.9 pp; C2
contradicts Chen's ½ and is a tolerance surface, not a number; and the C3 10-epoch curve is
faithful only over epochs 1–3 unless item 8 is done.

---

## 6. Flagged: documents asserting what the repository does not support

Ordered by how much damage each would do in open review.

1. **`README.md:35` — "`make reproduce-all` … Full reproduction at paper settings."** False for
   this submission. It runs the coursework pipeline. A reviewer following the README verbatim
   reproduces nothing that appears in the paper.
2. **`README.md` §§284–387 present the cut Rubanova/Latent-ODE material as live reproduction
   paths**, contradicting `OUT_OF_SCOPE.md`.
3. **"C1/C3" labelling throughout `OVERNIGHT_LOG.md`** makes canonical C1 (solver dynamics) look
   covered. It has zero committed artifacts.
4. **`REPLICATION_PLAN.md:80,263` and `Makefile:9` reference `extra/`**, which does not exist —
   despite `CHANGELOG_REPLICATION.md:131` claiming all references were updated.
5. **`DEVIATIONS.md` rows C2, C3, §D-D2 assert OPEN** for work that is done and committed. Harmless
   to the science, but it makes the deviations document — a first-class ReScience deliverable —
   look unmaintained.

**No case was found where a document overstates an experimental result.** Every recon-faithfulness
limitation I checked (D3's 2/5 at 1e-5, C3's epoch-3 window, D1's 4/5 spheres seed, C2's withdrawn
headline, D4's refusal to report a killed run) is disclosed in `OVERNIGHT_LOG.md` or
`DEVIATIONS.md` before I found it in the data. The experimental discipline held; the packaging
did not keep up.
