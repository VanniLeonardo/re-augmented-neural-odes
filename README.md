# A replication of Augmented Neural ODEs (Dupont et al., 2019)

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
| **D3** | Matched-param ANODE beats NODE on MNIST and is cheaper (Table 1) | Reproduced. ANODE **98.18 ± 0.29%** vs Dupont's 98.2 ± 0.1 (match); NODE **94.53 ± 0.44%** vs 96.4 ± 0.5 (**undershoots ~1.9 pp** — partial). At the loosest common faithful tol (1e-7) ANODE is **2.1× cheaper** in NFE. | `make fig-d3` |
| **D4** | Same on CIFAR-10 (Table 1) | Reproduced. NODE **53.59 ± 0.50%** vs Dupont's 53.7 ± 0.2 (match within noise); ANODE **59.34 ± 0.70%** vs 60.6 ± 0.4 (**~1.3 pp low** — partial). ANODE 1.24× cheaper in NFE at the faithful tol (1e-6). | `make fig-d4` |

### Chen et al. 2018 — secondary (C1–C4 only)

Chen's Table 1 is **deliberately not reproduced**; our MNIST rows are our own seeded
baselines, not a Table-1 claim (`DEVIATIONS.md` C4).

| ID | Claim (paper location) | Result | Figure target |
|---|---|---|---|
| **C1** | Error falls and cost rises as tolerance tightens; forward time ∝ NFE (Fig. 3a–b) | Mostly reproduced. Error falls monotonically for all adaptive solvers; Spearman ρ(time, NFE) = **0.98–0.99**. **The monotone-cost check is refuted as pre-declared**: dopri5 NFE goes 26→20 between tol 1e-1 and 1e-2 — both rows `recon_ok` 0/5, i.e. the violation lies where the solver is not integrating. | `make fig-c1` |
| **C2** | Backward NFE ≈ ½ forward NFE (Fig. 3c) | **Not reproduced — headline withdrawn.** There is no single ratio: in the integrating regime it runs **≈13–121×** and is driven by tolerance and by how trained the field is. Reported as a tolerance × field surface. | `make fig-c2` |
| **C3** | NFE increases during training (Fig. 3d) | Reproduced. At a recon-faithful 1e-7 the MNIST conv field's NFE grows **384 → 738** over 6 epochs, and the faithful tolerance itself tightens as training proceeds. | `make fig-c3` |
| **C4** | ODE-Net memory is O(1) in effective depth via the adjoint (Table 1, memory) | Reproduced (**corrected experiment** — the coursework swept the wrong axis). Adjoint **+0.0007 MB/NFE** (flat) vs direct backprop **+31.8 MB/NFE**. | `make fig-c4` |

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

Per-claim costs (`make help`). D4 is measured; the others are estimates on the hardware
below and will differ on yours.

| Target | What | Cost | Device |
|---|---|---|---|
| `d8` | 1-D crossing flow | ~5 min | CPU |
| `d1` | Toy separation / NFE vs budget | ~2 h | CPU |
| `d2` | Missing-slice generalisation | ~1–2 h | CPU |
| `d3` | Matched-param MNIST + faithful NFE | ~1.5 h | GPU |
| `d4` | Matched-param CIFAR-10 | **1.9 h (measured)** | GPU |
| `c1` | Solver dynamics | ~3.5 h serial | GPU |
| `c2` | bwd/fwd NFE vs tolerance | ~1–2 h | mixed |
| `c3` | NFE growth + stiffening | ~3 h | GPU |
| `c4` | O(1) memory vs NFE | ~0.5 h | GPU |

**Hardware the committed results were produced on.** Most: **NVIDIA RTX 3090 (24 GB)**,
AMD Ryzen 7 7700X (16 threads), 62 GB RAM, driver 595, CUDA 12.1, Python 3.11.15,
torch 2.5.1. C1 ran on **A100 80GB PCIe MIG slices** on a SLURM cluster. **Every result
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
byte-identically).

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
