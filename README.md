# A replication of Augmented Neural ODEs (Dupont et al., 2019)

[![reproducibility](https://github.com/VanniLeonardo/NeuralODEs/actions/workflows/reproducibility.yml/badge.svg?branch=rescience-c-replication)](https://github.com/VanniLeonardo/NeuralODEs/actions/workflows/reproducibility.yml)

A [ReScience C](https://rescience.github.io/) replication. **Primary target:** Dupont,
Doucet & Teh, *Augmented Neural ODEs* (NeurIPS 2019). **Secondary:** four solver/memory
claims from Chen, Rubanova, Bettencourt & Duvenaud, *Neural Ordinary Differential
Equations* (NeurIPS 2018).

Everything here is reimplemented from the papers **as described**. No code was copied
from either author's release — see [`PROVENANCE.md`](PROVENANCE.md). Where our
implementation departs from the papers, it is recorded in
[`DEVIATIONS.md`](DEVIATIONS.md), which is a first-class deliverable, not an appendix.

---

## Start here (no GPU required)

```bash
conda env create -f environment.yml && conda activate neural_odes   # or: pip install -r requirements.txt
make test        # gated unit suite            (~1 min)
make smoke       # end-to-end pipeline check   (~4 min, CPU, no accounts)
make figures     # EVERY figure, from committed CSVs (~2 min, CPU)
```

`make figures` is the command that matters. The **per-seed CSVs under `results/` are the
canonical artifacts** and are committed; figures are not. Every figure in the paper is
rebuilt from those CSVs, with no GPU, no training and no network — and each target also
prints the **pre-declared refutation check** for its claim, so you can audit the numbers
rather than just look at the picture.

The same path runs in CI on every push ([`.github/workflows/reproducibility.yml`](.github/workflows/reproducibility.yml)):
a no-cache build of the pinned container, `make smoke`, `make figures`, and a check that
regenerating the figures modifies no committed result.

To re-run the experiments themselves (GPU, ~15–20 h total), see
[Reproducing from scratch](#reproducing-from-scratch). `make help` lists every target
with its cost.

---

## How to read a result: `recon_ok`

This is the methodological core of the replication, and it is why several numbers here
differ from what a naive re-run produces.

A Neural ODE's vector field **stiffens as it trains**. At a loose solver tolerance the
adaptive integrator still returns an answer and still reports a plausible NFE — while not
actually integrating the field. Numbers measured there are artifacts.

So every result carries a **reconstruction check**: integrate forward to *T*, then
integrate back to 0, and require the input to be recovered to a relative tolerance
(`recon_rel < 1e-2`). The per-row flag is **`recon_ok`**.

- **Tolerance is a first-class experimental axis**, never a fixed default. Results are
  reported at the **loosest tolerance at which *both* compared arms are `recon_ok` on
  every seed**.
- A row that fails the check is **kept and reported as data** — it is evidence about the
  tolerance axis — but it is never used for a headline cost or accuracy number.
- If no tolerance satisfies the check within the step cap, that is recorded (`recon_ok=0`,
  NFE lower-bounded) rather than papered over.

Several earlier conclusions in this project were **retracted** for exactly this reason.
The clearest case: the 1-D crossing-flow result was originally measured at `atol=rtol=1e-3`,
where the check fails for all 10 model-seeds and the worst reconstructs with **1200%**
relative error. The conclusion survived re-measurement, but it had not been *established*.

A second example, in the other direction: at the `1e-3` tolerance used to *train* the image
models, the check fails on **every** trained model — 0/5 seeds, both arms, MNIST and CIFAR-10.
So the image accuracies were first measured in a regime that does not integrate the field.
Re-measured at a tolerance that passes, they move by at most **0.11 pp** on average
(**0.37 pp** for the worst seed): classification accuracy is robust to integration error,
NFE is not. Both are now reported at a checked tolerance (`make fig-d3`, `make fig-d4` print
the comparison, and `figures/d{3,4}/acc_vs_tol.png` show it).

---

## What is replicated

Claim IDs match [`REPLICATION_PLAN.md`](REPLICATION_PLAN.md) §3. Full raw numbers, each
with its pre-declared refutation condition, are in
[`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md).

### Dupont et al. 2019 — primary

| ID | Claim (paper location) | Result | Figure target |
|---|---|---|---|
| **D8** | A 1-D NODE flow is order-preserving and cannot represent the crossing map; ANODE can (Prop. 1, Fig. 3) | Reproduced. At 1e-7 (both arms 5/5 `recon_ok`): NODE MSE **1.0002**, ANODE-p1 **0.0007**. NODE sits *at* the theoretical floor of 1.0, not below it. | `make fig-d8` |
| **D1** | NODE NFE grows with training budget; ANODE stays flat and is cheaper (§4.2, Fig. 6) | Reproduced on both geometries. Spheres @50 ep: NODE NFE 218 vs ANODE 170; growth ×1.74 vs ×1.04 over 25→500 epochs. | `make fig-d1` |
| **D2** | NODE has a large generalization gap across an unobserved angular slice; ANODE does not (§5.1, Fig. 9) | Reproduced. Held-out slice accuracy NODE **0.619** vs ANODE **1.000**; slice loss 6.089 vs 0.000. Wedge `[0, π/5]` verified against the paper text. | `make fig-d2` |
| **D3** | Matched-param ANODE beats NODE on MNIST and is cheaper (Table 1) | Reproduced; **partial on NODE**. At the loosest tolerance where both arms pass the check (1e-6): ANODE **98.05 ± 0.19%** vs Dupont 98.2 ± 0.1 (match); NODE **94.16 ± 0.44%** vs 96.4 ± 0.5 (**~2.2 pp low** — and ~2.1 pp low in an independent first run, so a consistent undershoot). ANODE is **1.65–2.10×** cheaper in NFE across faithful tolerances and runs. | `make fig-d3` |
| **D4** | Same on CIFAR-10 (Table 1) | Reproduced; **partial on ANODE**. At 1e-5 (both arms pass): NODE **53.69 ± 0.81%** vs Dupont 53.7 ± 0.2 (match); ANODE **60.04 ± 0.94%** vs 60.6 ± 0.4 (**0.6 pp low**; an independent first run gave 59.34, 1.3 pp low — the miss is within our own run-to-run spread, so it is reported as a small undershoot, not a match). ANODE is **1.24–1.93×** cheaper in NFE across faithful tolerances and runs. | `make fig-d4` |

### Chen et al. 2018 — secondary (C1–C4 only)

Chen's Table 1 is **deliberately not reproduced**; our MNIST rows are our own seeded
baselines, not a Table-1 claim (`DEVIATIONS.md` C4).

| ID | Claim (paper location) | Result | Figure target |
|---|---|---|---|
| **C1** | Error falls and cost rises as tolerance tightens; forward time ∝ NFE (Fig. 3a–b) | Mostly reproduced. Error falls monotonically for all adaptive solvers; Spearman ρ(time, NFE) = **0.98–0.99**. **The monotone-cost check is refuted as pre-declared**: dopri5 NFE goes 26→20 between tol 1e-1 and 1e-2 — both rows `recon_ok` 0/5, i.e. the violation lies where the solver is not integrating. | `make fig-c1` |
| **C2** | Backward NFE ≈ ½ forward NFE (Fig. 3c) | **Not reproduced — headline withdrawn.** There is no single ratio: in the integrating regime it runs **≈13–121×** and is driven by tolerance and by how trained the field is. Reported as a tolerance × field surface. | `make fig-c2` |
| **C3** | NFE increases during training (Fig. 3d) | Reproduced. At a recon-faithful 1e-7 the MNIST conv field's NFE grows **384 → 738** over 6 epochs, and the faithful tolerance itself tightens as training proceeds. | `make fig-c3` |
| **C4** | ODE-Net memory is O(1) in effective depth via the adjoint (Table 1, memory) | Reproduced (**corrected experiment** — the coursework swept the wrong axis). Adjoint **+0.0007 MB/NFE** (flat) vs direct backprop **+31.8 MB/NFE**. | `make fig-c4` |

### Extension: Fig 9 made systematic

Dupont's Fig 9 removes one wedge from one geometry. `make fig-slice-grid` sweeps augmentation
p ∈ {0, 1, 2, 3, 5} × wedge width {π/8, π/5, π/3} × {spheres, circles} × 10 seeds — 300 runs,
each Fig 9's exact recon-checked harness (6 fail the check and are excluded). Held-out-wedge
accuracy, median over seeds:

| | π/8 | π/5 | π/3 |
|---|---|---|---|
| spheres — NODE | 0.926 | 0.744 | 0.701 |
| spheres — best ANODE | 1.000 | 1.000 | 1.000 (p=2) |
| circles — NODE | 0.923 | 0.812 | 0.854 |
| circles — best ANODE | 1.000 | 1.000 | 0.998 (p=2) |

- ANODE generalises better than NODE **at every width, on both geometries, for every p ≥ 1**. No
  pre-declared check failed.
- On spheres the advantage **grows from π/8 to π/5, then plateaus**: the π/5 → π/3 step is inside
  the seed spread, and on *loss* the advantage shrinks there, because ANODE starts to degrade on
  the widest hole too.
- The circles gap is **smaller than on spheres and not monotone in width**. A one-seed probe had
  suggested circles would show no gap at all, and we predicted that check would fail; ten seeds
  showed the probe seed was an outlier.
- The failure is **specific to the unobserved region**: every model, NODE included, scores ≥ 0.998
  on the observed region.
- **More augmentation is not monotonically better**: p = 2 is best at the widest wedge on both
  geometries (Dupont chose p = 5 for fit, a different criterion — `DEVIATIONS.md` A12).
- The grid's π/5 spheres cells **reproduce the committed Fig 9 run exactly** (10/10 matched cells,
  identical held-out accuracy).

### Out of scope

Rubanova et al. 2019 (Latent ODE / ODE-RNN, sine, spiral) is **cut from the submission**;
the code remains in the repo but nothing in the paper depends on it. SVHN and ImageNet are
permanently out. Rationale: [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md).

The pre-replication coursework experiments are still runnable via `make coursework`, but
they are **not part of the submission** and are not cited by it. One of them — the
stem × geometry × head factorial — has an explicitly **withdrawn** verdict
(`DEVIATIONS.md` A1/A3): it was measured at a non-integrating tolerance.

---

## Reproducing from scratch

```bash
make reproduce-all       # everything (~15-20 GPU-h)
make reproduce-dupont    # d8 d1 d2 d3 d4
make reproduce-chen      # c4 c2 c3 c1
make d4                  # or any single claim
```

Per-claim costs (`make help`). D3 and D4 are measured (A100 MIG slice, serial over 5
seeds); the others are estimates on the hardware below and will differ on yours.

| Target | What | Cost | Device |
|---|---|---|---|
| `d8` | 1-D crossing flow | ~5 min | CPU |
| `d1` | Toy separation / NFE vs budget | ~2 h | CPU |
| `d2` | Missing-slice generalisation | ~1–2 h | CPU |
| `d3` | Matched-param MNIST | **1.9 h serial (measured)** | GPU |
| `d4` | Matched-param CIFAR-10 | **4.3 h serial (measured)** | GPU |
| `c1` | Solver dynamics | ~3.5 h serial | GPU |
| `c2` | bwd/fwd NFE vs tolerance | ~1–2 h | mixed |
| `c3` | NFE growth + stiffening | ~3 h | GPU |
| `c4` | O(1) memory vs NFE | ~0.5 h | GPU |

**Hardware the committed results were produced on.** Most: **NVIDIA RTX 3090 (24 GB)**,
AMD Ryzen 7 7700X (16 threads), 62 GB RAM, driver 595, CUDA 12.1, Python 3.11.15,
torch 2.5.1. C1 and the canonical D3/D4 runs ran on **A100 80GB PCIe MIG slices** on a
SLURM cluster; the first D3/D4 runs (RTX 3090) are kept under `results/d{3,4}/run1_rtx3090/`
as replicates and printed by the report scripts. **Every result
row records the GPU it ran on** in a `hardware` column — this is not decoration: it caught
a real confound, when a SLURM array scattered C1's seeds across two different MIG slice
sizes, making pooled wall-clock incomparable (the timing claim was re-checked within each
hardware group and survived).

Some experiments pin `CUDA_VISIBLE_DEVICES=""` deliberately: the 2-D toy problems are
faster on CPU and avoid GPU contention.

**Determinism.** Seeds control initialisation, shuffling and data generation, so a re-run
reproduces the reported numbers *within the documented spread*, not bit-for-bit; adaptive
solvers and cuDNN kernel selection are not bit-reproducible across machines. Figures
rebuilt from committed CSVs *are* deterministic (`results/c2/c2_surface.csv` regenerates
byte-identically). CPU toy runs are more reproducible still: the §6 grid re-ran Fig 9's
cells two months later, single- instead of multi-threaded, and matched the committed held-out
accuracies exactly.

### Cluster runs

`scripts/*.slurm` are **illustrative** and carry `CHANGE_ME` placeholders for
account/partition; the portable path is the `make` targets. `scripts/run_c1_solver_dynamics.slurm`
shows the per-seed array pattern: each task writes its own shard (`*_s<seed>.csv`) so
parallel tasks never race on one file, and the report script merges them.

---

## Datasets and network access

MNIST and CIFAR-10 are downloaded on first use and are **not committed** (datasets are
gitignored). Nothing else needs the network — logging defaults to a CSV backend and needs
no Weights & Biases account.

⚠️ **Known fragility.** The canonical MNIST host `yann.lecun.com` now returns **HTTP 404**.
torchvision silently falls back to the `ossci-datasets` S3 mirror, which works today, so
reproduction currently depends on a third-party mirror. If it is unreachable — offline
machine, firewalled cluster node, mirror retired — the loaders raise an explicit error
naming the directory to populate rather than failing obscurely. On an air-gapped machine,
copy the dataset directory in:

```bash
rsync -az data/cifar-10-batches-py/ host:/path/to/repo/data/cifar-10-batches-py/
```

---

## Verified environment

The **guaranteed-reproducible artifact is the Docker image**; the conda env is the
convenience GPU path.

```bash
make docker-smoke     # builds the CPU image and runs `make smoke` inside it
```

Pins live in `requirements.txt` (+ a fully-resolved `requirements-lock-cpu.txt` captured
inside the container) and `environment.yml`. If conda cannot solve the exact versions,
use Docker.

---

## Repository structure

```text
NeuralODEs/
├── Makefile                      # every reproduction entry point (`make help`)
├── .github/workflows/            # CI: the reviewer path in the pinned container, every push
├── REPLICATION_PLAN.md           # scope, claim-by-claim gap analysis, compute budget
├── DEVIATIONS.md                 # where we differ from the papers AS DESCRIBED
├── PROVENANCE.md                 # copying provenance: did we copy author code? (no)
├── OVERNIGHT_LOG.md              # pre-declared refutation checks + raw observations
├── OUT_OF_SCOPE.md               # the excluded Rubanova material, and why
├── results/                      # COMMITTED per-seed CSVs — the canonical artifacts
│   ├── budget/ budget_circles/   #   D1   toy separation, NFE vs budget
│   ├── slice_spheres/            #   D2   missing-slice generalisation
│   ├── d3/ d3_faithful/          #   D3   MNIST matched-param + faithful NFE
│   ├── d4/                       #   D4   CIFAR-10 matched-param
│   ├── crossing/                 #   D8   1-D crossing flow
│   ├── slice_grid/               #   §6   Fig 9 made systematic (p × width × geometry)
│   ├── c1/                       #   C1   solver dynamics (per-seed shards)
│   ├── c2/ c2_circles/           #   C2   bwd/fwd NFE surface
│   ├── mnist_nfe/ mnist_stiffening/  # C3 NFE growth + stiffening mechanism
│   ├── c4/                       #   C4   O(1) memory vs NFE
│   └── factorial/                #   WITHDRAWN verdict; provenance only
├── scripts/
│   ├── run_*.py, train_*.py      # experiment harnesses (one per claim)
│   ├── *_report.py, plot_*.py    # committed CSVs -> figures + refutation checks
│   └── *.slurm                   # illustrative cluster scripts (placeholders)
├── models/
│   ├── continuous.py             # ODEFunc, ODEBlock, ConvODEFunc
│   ├── networks.py               # ODENet, ConvODENet, EulerDiscretizedODENet
│   └── ode_rnn.py                # out of scope (Rubanova)
├── data/                         # loaders; datasets themselves are gitignored
├── training/                     # training/eval loops, NFE tracking, CSV logger
└── tests/                        # gated suite: `pytest -m "not extra"`
```

---

## Tests

```bash
pytest -m "not extra"    # gated suite (the `extra` mark is out-of-scope Rubanova)
```

The suite pins the invariants the claims rest on, not just plumbing: forward/backward NFE
split, adjoint-vs-direct gradient agreement, augmentation zero-init, ODE-Net ↔ Euler
parameter parity, the spheres geometry against Dupont App. F.2.1, seed determinism, flow
faithfulness, and the D4 resume logic (which must never treat a seed that died mid-training
as resumable — training is sequential and no checkpoint is saved).

---

## Licensing

Code, harnesses, report scripts and the committed result CSVs: **MIT** (see
[`LICENSE`](LICENSE)). ReScience C does not mandate a particular code licence — its
FAQ asks only that code be under an open licence, referring to the Debian Free
Software Guidelines — and MIT is DFSG-compliant.

`paper/` is the official ReScience C template redistributed under its own terms
(GPL-3+ for the template files, Apache-2.0 / SIL OFL for the bundled fonts); the
published article will be CC-BY-4.0. Details in `LICENSE`.

Copyright is held jointly by the six contributors in the repository's git history.

## Citation

This is a replication. Cite the original work:

```bibtex
@inproceedings{dupont2019augmented,
  title     = {Augmented Neural ODEs},
  author    = {Dupont, Emilien and Doucet, Arnaud and Teh, Yee Whye},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}
@inproceedings{chen2018neural,
  title     = {Neural Ordinary Differential Equations},
  author    = {Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2018}
}
```
