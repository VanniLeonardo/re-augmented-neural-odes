# Deviations from the papers as described

[`PROVENANCE.md`](PROVENANCE.md) answers whether we copied the authors' code. We did not.
This file answers the other question: does our implementation match the setup the papers
describe? It records every place we knowingly differ.

Each row gives what the papers specify, what this replication does, the effect on the
reproduced claim, and a status. Statuses are:

- **matches**, we changed our code to follow the paper;
- **deliberate**, a difference we chose and defend;
- **open**, a difference we have not resolved.

Rows reference the claim identifiers used by the `make` targets and the README.

---

## A. Toy separation experiments (`d1`, `d2`, `d8`, and the extension)

| # | The papers specify | This replication | Effect on the result | Status |
|---|---|---|---|---|
| A1 | The ODE integrates in data space. The topological argument requires no learned warp before the flow (Sections 3 and 4). | No encoder. The ODE integrates the raw 2-D input (`use_stem=False`). The coursework applied a `Linear(2,2)+Tanh` stem first. | An encoder is a diffeomorphism and would supply the warp the argument assumes is absent. | matches |
| A2 | Toy vector field of width 32, `d+1 -> 32 -> 32 -> d` (App. F.1.1). | Width 32. The coursework used 64. | A wider field lets the Neural ODE approximate the distorted flow more easily, which flatters it. | matches |
| A3 | Concentric spheres: a filled inner disk `‖x‖ ≤ 0.5` inside an annulus `1.0 ≤ ‖x‖ ≤ 1.5`, in a 1:2 ratio (App. F.2.1). | `make_spheres` in `data/synthetic.py`. Geometry, radii and ratio match. The thinner circles geometry is kept as a second geometry. | The enclosure is what makes the task hard, so the geometry carries the claim. | matches |
| A4 | Solver dopri5 at the library defaults, which are rtol 1e-7 and atol 1e-9. | Training at 1e-3, with every reported quantity re-measured on a tolerance ladder that carries a reconstruction check. | 1e-3 is four to six orders looser than the default and does not integrate these fields. Measured there, the forward-backward reconstruction error reaches about 0.9 on data of radius 1. This is the reason the check exists. | deliberate |
| A5 | 50 epochs, 20 repeats (App. F.2.1). | The budget is swept over {25, 50, 100, 200, 500, 1000} with 5 seeds, so the 50-epoch point is measured rather than assumed. | At 50 epochs: NODE dense accuracy 0.995 ± 0.006 at NFE 241, ANODE-p1 0.999 ± 0.001 at NFE 177. Five seeds give wider intervals than the original twenty repeats. | open (seeds) |
| A6 | Toy learning rate 1e-3 (App. F.2.1). | 3e-3. | Faster convergence. Does not change the NFE growth or the gap between the two models. | open |
| A7 | The toy comparison includes a ResNet baseline (Fig. 5). | NODE against ANODE only. Our discrete baseline exists on MNIST only. | One baseline curve is missing. The NODE against ANODE claim is unaffected. | open |
| A8 | The wedge removed from training is `[0, π/5]` (Section 5.1, Fig. 9). | `[0, π/5]`. | None. An earlier version of this file recorded `[0, π/13]`, which appears nowhere in the paper. Corrected after checking the text. | matches |
| A9 | Augmentation appends p zeros to the state (Section 5). | The same, applied to the data-space state. | None. | matches |
| A10 | The classifier is a single linear map applied to the terminal state (Section 2). | `nn.Linear(ode_dim, num_classes)`, with no nonlinearity between the ODE endpoint and the logits. | None. Audited because the crossing-flow experiment showed that a learnable readout lets a Neural ODE bypass the flow. A single hyperplane cannot separate nested classes, so the flow does the work. | matches |
| A11 | The toy task is regression to ±1 under MSE (Section 4.1). | Binary classification under cross-entropy on the same two regions. | The flow-level mechanism is the same, but our losses are not comparable to the published loss curves. The NFE signature is comparable. | open |
| A12 | The best augmentation for d=2 is p=5, searched over {1,2,5} (App. F.2.1). | The budget sweep uses p=1. The extension sweeps p ∈ {0,1,2,3,5}. | p=1 is enough to show the flat-NFE control. For held-out generalisation p=2 is best at the widest wedge on both geometries, and p=5 does not beat p=1 or p=2. The original searched for fit rather than for generalisation, so this is a different criterion, not a contradiction. | matches |

### The toy result, and two retractions

This result was stated wrongly twice before it was measured correctly. Both errors had the
same cause, and the history is kept here deliberately.

1. The first version claimed the Neural ODE showed an NFE explosion and that the geometry was
   load-bearing. That was retracted as GPU contention.
2. The replacement claimed there was no Neural ODE failure and that NFE stayed flat at about 44.
3. Both were measured at `atol=rtol=1e-3`, which does not integrate these fields. The
   forward-backward reconstruction error there is about 0.9 on data of radius 1, and an
   independent linear probe on the terminal state falls to chance while the model's own head
   reads near 1.0. The flat NFE was the solver never doing the work.

Re-measured at 1e-6 with a reconstruction check (`scripts/run_budget_sweep.py`,
`results/budget/`, 5 seeds):

| Budget | Model | Dense accuracy | Median NFE | Reconstruction |
|---|---|---|---|---|
| 50 | NODE | 0.995 ± 0.006 | 241 | 5/5 |
| 50 | ANODE-p1 | 0.999 ± 0.001 | 177 | 5/5 |
| 1000 | NODE | 0.992 ± 0.008 | 465 (stiffest seed 812) | 4/5 |
| 1000 | ANODE-p1 | 0.999 ± 0.001 | 184 | 5/5 |

This reproduces the original d=2 result rather than contradicting it. The Neural ODE
approximates the finite sample by stretching the inner disk into a thin tendril that threads a
gap in the annulus. The flow remains a homeomorphism: the winding number of the image of the
annulus inner boundary around the image of the inner disk is 1.000, the map is injective, and
it reconstructs to 1e-5 at an accurate tolerance. It cannot reach 100% accuracy, because the
topological obstruction forces at least 0.70% misclassification on the annulus inner boundary,
against 1.35% observed. Its NFE grows from 219 to 465 over 25 to 1000 epochs. The stiffest
seed reaches 812 and stops passing the check at 1e-6, which is reported rather than hidden.
ANODE-p1 stays flat at about 180 NFE and passes on every seed.

## B. Image experiments (`d3`, `d4`)

| # | The papers specify | This replication | Effect on the result | Status |
|---|---|---|---|---|
| B1 | The convolutional field injects time as an extra channel before every convolution, in a 1x1, 3x3, 1x1 stack (App. F.1.2). | Time is prepended before each convolution. The coursework concatenated it once at the input. | A single concatenation is a different vector field. Matching the paper changes the parameter count slightly. | matches |
| B2 | MNIST at matched parameters: 92 filters (84,395) against 64 filters with p=5 (84,816), batch 256, 5 runs (App. F.2.2, Table 1). | `scripts/run_d3_anode_mnist.py`. Parameter counts asserted matched at 85,316 against 85,462, about 1% above the stated targets because of the flattening readout. Batch 256, 5 seeds, 8 epochs. | ANODE **98.05 ± 0.19** against 98.2 ± 0.1, a match. NODE **94.16 ± 0.44** against 96.4 ± 0.5, about 2.2 points low and consistent across two runs. NFE ratio 1.65 to 1.91. An earlier version of this file quoted 98.18 and 94.53, which were medians printed as if they were means. The run-1 means are 98.00 and 94.32. | matches (partial on the NODE number) |
| B3 | CIFAR-10 at matched parameters: 125 filters (172,358) against 64 filters with p=10 (171,799), batch 256, 5 runs (App. F.2.2, Table 1). | `scripts/run_d4_anode_cifar.py`, same field and readout as MNIST. Parameter counts asserted matched at 173,611 against 172,452. 10 epochs, since the original does not state an epoch count for CIFAR-10. | NODE **53.69 ± 0.81** against 53.7 ± 0.2, a match. ANODE **60.04 ± 0.94** against 60.6 ± 0.4, 0.6 points low. An earlier run on other hardware gave 53.59 and 59.34, so the shortfall is smaller than our run-to-run spread. NFE ratio 1.24 to 1.93. | matches (partial on the ANODE number) |

Both image experiments train at a tolerance of 1e-3, at which the reconstruction check fails
on every model and seed. Accuracy re-measured at a tolerance that passes moves by at most
0.11 points on average and 0.37 points for the worst seed, so the accuracy comparison stands.
Cost is always quoted at a checked tolerance.

## C. Solver dynamics and memory (`c1` to `c4`)

| # | The papers specify | This replication | Effect on the result | Status |
|---|---|---|---|---|
| C1 | Implicit Adams through SciPy, at a classification tolerance of 1e-3 (Section 3). | dopri5 through torchdiffeq, at 1e-3. | A different solver gives different NFE dynamics. This is the basis of the claim-6 result rather than an oversight, and the diagnosis in `scripts/c2_diagnosis_report.py` tests it directly. | deliberate |
| C2 | Backward NFE is about half forward NFE (Fig. 3c). | Not reproduced. In the integrating regime the ratio runs from 13 to 121, and reaches 123 on the trained convolutional field. | The dominant cause is the tolerance at which the adjoint is solved, which torchdiffeq couples to the forward tolerance by default. Decoupling it reduces the ratio by 5 to 76 times, from 123 to 4.1 on the convolutional field, with gradients still correct to 1%. A stiff-capable solver does not help and is often far worse. The ratio also does not grow with training in the way we first claimed. | not reproduced |
| C3 | Adjoint memory is constant in the effective depth of the model (Table 1, memory column). | The coursework varied the depth of the discrete baseline, which is the wrong axis. `scripts/c4_memory_vs_nfe.py` holds the model fixed and raises its own NFE. | Adjoint memory is flat at +0.0007 MB per NFE, against +31.8 for direct backpropagation, over 3 seeds. | matches |
| C4 | Chen's ODE-Net: convolutional, two downsampling steps and six ODE blocks, about 0.22M parameters, 0.42% error, 100 or more epochs (Table 1). | Not attempted. Our MNIST rows are our own seeded baselines at 10 epochs. | Our MNIST numbers are not a claim about that table, and the paper says so. | deliberate |

## D. Baselines and general setup

| # | The papers specify | This replication | Effect on the result | Status |
|---|---|---|---|---|
| D1 | The discrete baseline is an independent residual network. | `EulerDiscretizedODENet`, one weight-tied `ODEFunc` reused across explicit Euler steps. | This buys exact parameter parity, but it makes "the discrete model and the ODE-Net reach comparable accuracy" close to tautological, since they are the same field under two integrators. The class is named for what it is. | deliberate |
| D2 | Image batch size 256. Optimiser, learning rate and epoch counts are largely unstated in both papers. | Batch 256 for both image experiments. Adam at 1e-3. | The unstated epoch budgets are the most likely cause of our two accuracy undershoots, and we cannot attribute them cleanly. | matches |

---

## Related files

- Whether author code was copied: [`PROVENANCE.md`](PROVENANCE.md).
- Scope and what is excluded: [`REPLICATION_PLAN.md`](REPLICATION_PLAN.md),
  [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md).
- Conditions recorded before each run, and raw results: [`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md).
