# A replication of Augmented Neural ODEs (Dupont et al., 2019)

[![reproducibility](https://github.com/VanniLeonardo/NeuralODEs/actions/workflows/reproducibility.yml/badge.svg?branch=rescience-c-replication)](https://github.com/VanniLeonardo/NeuralODEs/actions/workflows/reproducibility.yml)

A [ReScience C](https://rescience.github.io/) replication of Dupont, Doucet and Teh,
*Augmented Neural ODEs* (NeurIPS 2019), together with four solver and memory claims from
Chen, Rubanova, Bettencourt and Duvenaud, *Neural Ordinary Differential Equations*
(NeurIPS 2018).

Both models were reimplemented from the descriptions in the papers. No code was copied from
either author's release. See [`PROVENANCE.md`](PROVENANCE.md). Departures from the papers
are recorded in [`DEVIATIONS.md`](DEVIATIONS.md).

---

## Start here (no GPU required)

```bash
conda env create -f environment.yml && conda activate neural_odes   # or: pip install -r requirements.txt
make test        # unit tests                            (~1 min)
make smoke       # end-to-end pipeline check             (~4 min, CPU, no accounts)
make figures     # every figure, from the committed data (~2 min, CPU)
```

`make figures` is the command to start with. The per-seed CSV files under `results/` are the
committed artifact, and the figures are not. Every figure in the paper is rebuilt from those
files without a GPU, without training and without network access. Each target also prints
the condition that was recorded before the run, so the numbers can be audited rather than
only viewed.

The same path runs in CI on every push
([`.github/workflows/reproducibility.yml`](.github/workflows/reproducibility.yml)): a
no-cache build of the pinned container, `make smoke`, `make figures`, and a check that
rebuilding the figures modifies no committed result.

To re-run the experiments, see [Reproducing from scratch](#reproducing-from-scratch).
`make help` lists every target with its cost.

---

## How to read a result: `recon_ok`

This is the methodological core of the replication, and it explains why several numbers here
differ from what a direct re-run produces.

The vector field of a Neural ODE stiffens as it trains. At a loose solver tolerance the
adaptive integrator still returns an answer and a plausible NFE without tracking the flow.
Numbers measured there describe the solver rather than the model.

Every result therefore carries a reconstruction check. We integrate forward to *T*, integrate
back to 0, and require the input to be recovered to a relative error below `1e-2`. The
per-row flag is `recon_ok`.

- Tolerance is an experimental axis, not a fixed default. A quantity compared between two
  models is quoted at the loosest tolerance where both pass the check on every seed.
- Rows that fail the check are kept and reported. They are evidence about the tolerance
  axis. They are never used for a headline number.
- If no tolerance passes within the step cap, that is recorded as data, with `recon_ok=0`
  and the NFE lower-bounded.

Several conclusions in this project were withdrawn for this reason. The clearest case is the
one-dimensional crossing flow, first measured at `atol=rtol=1e-3`. At that tolerance the
check fails for all ten trained models, and the worst reconstructs its input with a relative
error above 10. The conclusion survived re-measurement, but it had not been established.

The image experiments show the same problem in a milder form. At the `1e-3` tolerance used
for training, the check fails on every trained model, on both datasets and all five seeds.
Re-measured at a tolerance that passes, accuracy moves by at most 0.11 points on average and
0.37 points for the worst seed. Accuracy is robust to integration error and cost is not, so
both are now quoted at checked tolerances (`make fig-d3` and `make fig-d4` print the
comparison, and `figures/d{3,4}/acc_vs_tol.png` show it).

---

## What is replicated

The identifiers below name the `make` targets. They correspond to claims 1 to 8 of the
paper. Raw numbers and the condition recorded before each run are in
[`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md).

### Dupont et al. 2019

| ID | Claim | Result | Figure |
|---|---|---|---|
| `d8` | A 1-D Neural ODE flow preserves order and cannot represent the crossing map (Prop. 1, Fig. 3) | Reproduced. At 1e-7, where both models pass the check: NODE MSE **1.0002**, ANODE-p1 **0.0007**. The NODE sits at the theoretical floor of 1.0 rather than below it. | `make fig-d8` |
| `d1` | NODE cost grows with the training budget, ANODE stays flat (§4.2, Fig. 6) | Reproduced on both geometries. Spheres at 50 epochs: NODE 218 NFE against ANODE 170. Growth ×1.74 against ×1.04 over 25 to 500 epochs. | `make fig-d1` |
| `d2` | NODE generalises poorly across an unobserved angular wedge (§5.1, Fig. 9) | Reproduced. Held-out accuracy NODE **0.619** against ANODE **1.000**, loss 6.089 against 0.000. The wedge `[0, π/5]` was verified against the paper text. | `make fig-d2` |
| `d3` | Matched-parameter ANODE beats NODE on MNIST and is cheaper (Table 1) | Reproduced, partial on the NODE number. At 1e-6, where both pass: ANODE **98.05 ± 0.19** against 98.2 ± 0.1, NODE **94.16 ± 0.44** against 96.4 ± 0.5. The NODE undershoot of about 2.2 points is consistent across two runs. ANODE is 1.65 to 2.10 times cheaper in NFE. | `make fig-d3` |
| `d4` | The same on CIFAR-10 (Table 1) | Reproduced, partial on the ANODE number. At 1e-5, where both pass: NODE **53.69 ± 0.81** against 53.7 ± 0.2, ANODE **60.04 ± 0.94** against 60.6 ± 0.4. An earlier run on other hardware gave 59.34, so the shortfall is smaller than our run-to-run spread. ANODE is 1.24 to 1.93 times cheaper in NFE. | `make fig-d4` |

### Chen et al. 2018

Chen's Table 1 is deliberately not reproduced. Our MNIST rows are our own seeded baselines
and are not a claim about that table (`DEVIATIONS.md`, row C4).

| ID | Claim | Result | Figure |
|---|---|---|---|
| `c1` | Error falls and cost rises as the tolerance tightens, and time is proportional to NFE (Fig. 3a-b) | Mostly reproduced. Error falls monotonically for every adaptive solver, and Spearman ρ between time and NFE is 0.98 to 0.99. The monotone-cost condition recorded before the run is violated: dopri5 NFE falls from 26 to 20 between 1e-1 and 1e-2. Both of those rows fail the reconstruction check, so the violation lies where the solver is not integrating. | `make fig-c1` |
| `c2` | Backward NFE is about half forward NFE (Fig. 3c) | Not reproduced. There is no single ratio. In the integrating regime it runs from 13 to 121 times, driven by the tolerance at which the adjoint is solved. `make fig-c2` reports it as a tolerance and field surface, and `scripts/c2_diagnosis_report.py` gives the diagnosis. | `make fig-c2` |
| `c3` | NFE increases during training (Fig. 3d) | Reproduced. At 1e-7 the MNIST convolutional field grows from **384 to 738** NFE over six epochs, and the checked tolerance itself tightens as training proceeds. | `make fig-c3` |
| `c4` | Adjoint memory is constant in effective depth (Table 1, memory) | Reproduced, using a corrected experiment. The coursework version swept the wrong axis. Adjoint memory is flat at **+0.0007 MB per NFE** against **+31.8** for direct backpropagation. | `make fig-c4` |

### Extension: Figure 9 made systematic

Dupont removes one wedge from one geometry. `make fig-slice-grid` sweeps the augmentation
dimension p ∈ {0, 1, 2, 3, 5}, the wedge width {π/8, π/5, π/3} and two geometries over ten
seeds, giving 300 runs. Each run uses the same reconstruction check, and six fail it and are
excluded. Held-out accuracy, median over seeds:

| | π/8 | π/5 | π/3 |
|---|---|---|---|
| spheres, NODE | 0.926 | 0.744 | 0.701 |
| spheres, best ANODE | 1.000 | 1.000 | 1.000 (p=2) |
| circles, NODE | 0.923 | 0.812 | 0.854 |
| circles, best ANODE | 1.000 | 1.000 | 0.998 (p=2) |

- ANODE generalises better at every width, on both geometries, for every p ≥ 1. No condition
  recorded before the run was violated.
- On the spheres the advantage grows from π/8 to π/5 and then flattens. The last step lies
  inside the seed spread, and measured by loss the advantage shrinks, because ANODE also
  degrades at the widest wedge.
- On the circles the gap is smaller and is not monotone in width. A one-seed pilot suggested
  there would be no gap at all, and we predicted the check would fail. Ten seeds showed the
  pilot seed was an outlier.
- The failure is specific to the unobserved region. Every model, including the NODE, scores
  at least 0.998 on the observed region.
- More augmentation is not uniformly better. p = 2 generalises best at the widest wedge on
  both geometries (`DEVIATIONS.md`, row A12).
- The grid reproduces the committed Figure 9 cells exactly, in all ten matched cells.

### Out of scope

Rubanova et al. 2019 (Latent ODE, ODE-RNN, sine and spiral) is excluded from the submission.
The code remains in the repository but nothing in the paper depends on it. SVHN and ImageNet
are also excluded. See [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md).

The pre-replication coursework experiments remain runnable through `make coursework`. They
are not part of the submission and are not cited by it. One of them, the stem by geometry by
head factorial, has a withdrawn result (`DEVIATIONS.md`, rows A1 and A3). It was measured at
a tolerance that does not integrate the field.

---

## Reproducing from scratch

```bash
make reproduce-all       # everything (~15-20 GPU-h)
make reproduce-dupont    # d8 d1 d2 d3 d4
make reproduce-chen      # c4 c2 c3 c1
make d4                  # or any single claim
```

Costs are listed by `make help`. The `d3` and `d4` figures are measured on an A100 MIG slice,
serially over five seeds. The others are estimates on the hardware below and will differ on
yours.

| Target | What | Cost | Device |
|---|---|---|---|
| `d8` | 1-D crossing flow | ~5 min | CPU |
| `d1` | Toy separation, NFE against budget | ~2 h | CPU |
| `d2` | Missing-slice generalisation | ~1-2 h | CPU |
| `d3` | Matched-parameter MNIST | 1.9 h serial (measured) | GPU |
| `d4` | Matched-parameter CIFAR-10 | 4.3 h serial (measured) | GPU |
| `c1` | Solver dynamics | ~3.5 h serial | GPU |
| `c2` | Backward against forward NFE | ~1-2 h | mixed |
| `c3` | NFE growth and stiffening | ~3 h | GPU |
| `c4` | Constant memory against NFE | ~0.5 h | GPU |

**Hardware.** Most results were produced on an NVIDIA RTX 3090 (24 GB) with an AMD Ryzen 7
7700X, 62 GB RAM, driver 595, CUDA 12.1, Python 3.11.15 and torch 2.5.1. The `c1` sweep and
the reported `d3` and `d4` runs used A100 80GB PCIe MIG slices on a SLURM cluster. The first
`d3` and `d4` runs, on the RTX 3090, are kept under `results/d{3,4}/run1_rtx3090/` and are
printed by the report scripts as replicates.

Every result row records the GPU it ran on, in a `hardware` column. This is not decoration.
It caught a real confound: a SLURM array scattered the `c1` seeds across two different MIG
slice sizes, which made pooled wall-clock times incomparable. The timing result was
re-checked within each hardware group and survived.

Some experiments set `CUDA_VISIBLE_DEVICES=""` deliberately. The two-dimensional problems are
faster on CPU and this avoids GPU contention.

**Determinism.** Seeds fix initialisation, shuffling and data generation, so a re-run
reproduces the reported numbers within the documented spread rather than bit for bit.
Adaptive solvers and cuDNN kernel selection are not bit-reproducible across machines. Figures
rebuilt from the committed files are deterministic: `results/c2/c2_surface.csv` regenerates
byte for byte. CPU runs are more reproducible still. The extension re-ran the Figure 9 cells
two months later, single-threaded rather than multi-threaded, and matched the committed
held-out accuracies exactly.

### Cluster runs

The files in `scripts/*.slurm` are illustrative and carry `CHANGE_ME` placeholders for
account and partition. The portable path is the `make` targets.
`scripts/run_c1_solver_dynamics.slurm` shows the per-seed array pattern. Each task writes its
own file (`*_s<seed>.csv`), so parallel tasks never race, and the report script merges them.

---

## Datasets and network access

MNIST and CIFAR-10 are downloaded on first use and are not committed. Nothing else needs the
network. Logging defaults to a CSV backend and needs no Weights and Biases account.

The canonical MNIST host `yann.lecun.com` now returns HTTP 404. torchvision falls back to the
`ossci-datasets` S3 mirror, which works today, so reproduction currently depends on a
third-party mirror. If that mirror is unreachable, the loaders raise an explicit error naming
the directory to populate. On an air-gapped machine, copy the dataset directory in:

```bash
rsync -az data/cifar-10-batches-py/ host:/path/to/repo/data/cifar-10-batches-py/
```

---

## Verified environment

The Docker image is the reproducible artifact. The conda environment is a convenience path
for GPU work.

```bash
make docker-smoke     # builds the CPU image and runs `make smoke` inside it
```

Pinned versions are in `requirements.txt`, with a fully resolved
`requirements-lock-cpu.txt` captured inside the container, and in `environment.yml`. If conda
cannot solve the exact versions, use Docker.

---

## Repository structure

```text
NeuralODEs/
├── Makefile                      # every reproduction entry point (`make help`)
├── .github/workflows/            # CI: the reviewer path, in the pinned container
├── REPLICATION_PLAN.md           # scope, claim-by-claim analysis, compute budget
├── DEVIATIONS.md                 # where we differ from the papers as described
├── PROVENANCE.md                 # whether author code was copied (it was not)
├── OVERNIGHT_LOG.md              # conditions recorded before each run, and raw results
├── OUT_OF_SCOPE.md               # the excluded material, and why
├── paper/                        # the ReScience C article and its template
├── results/                      # committed per-seed CSV files, the canonical artifact
│   ├── budget/ budget_circles/   #   d1  toy separation, NFE against budget
│   ├── slice_spheres/            #   d2  missing-slice generalisation
│   ├── d3/ d3_faithful/          #   d3  MNIST matched-parameter
│   ├── d4/                       #   d4  CIFAR-10 matched-parameter
│   ├── crossing/                 #   d8  1-D crossing flow
│   ├── slice_grid/               #       extension: Figure 9 made systematic
│   ├── c1/                       #   c1  solver dynamics, one file per seed
│   ├── c2/ c2_circles/           #   c2  backward against forward NFE
│   ├── c2_diagnosis/             #   c2  diagnosis of the ratio
│   ├── mnist_nfe/ mnist_stiffening/  # c3 NFE growth and stiffening
│   ├── c4/                       #   c4  constant memory against NFE
│   └── factorial/                #       withdrawn result, kept for provenance
├── scripts/
│   ├── run_*.py, train_*.py      # experiment harnesses, one per claim
│   ├── *_report.py, plot_*.py    # committed files to figures and checks
│   └── *.slurm                   # illustrative cluster scripts
├── models/
│   ├── continuous.py             # ODEFunc, ODEBlock, ConvODEFunc
│   ├── networks.py               # ODENet, ConvODENet, EulerDiscretizedODENet
│   └── ode_rnn.py                # out of scope
├── data/                         # loaders; the datasets themselves are gitignored
├── training/                     # training and evaluation loops, NFE tracking, logging
└── tests/                        # `pytest -m "not extra"`
```

---

## Tests

```bash
pytest -m "not extra"    # the `extra` mark covers the out-of-scope material
```

The suite pins the invariants the claims rest on rather than only the plumbing: the
forward and backward NFE split, agreement between adjoint and direct gradients, augmentation
zero-initialisation, parameter parity between the ODE-Net and the Euler baseline, the spheres
geometry against Dupont Appendix F.2.1, seed determinism, flow faithfulness, and the CIFAR-10
resume logic, which must never treat a seed that stopped part way as resumable.

---

## Licensing

The code, harnesses, report scripts and committed result files are MIT licensed, see
[`LICENSE`](LICENSE). ReScience C does not mandate a particular licence for code. Its FAQ
asks that code be under an open licence and refers to the Debian Free Software Guidelines,
which MIT satisfies.

`paper/` redistributes the official ReScience C template under its own terms: GPL-3 or later
for the template files, and Apache-2.0 or the SIL Open Font License for the bundled fonts.
The published article will be CC-BY-4.0. Details are in `LICENSE`.

Copyright is held jointly by the six contributors listed in the git history.

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
