# PROVENANCE (copying-provenance only)

**Scope of this file:** did we *copy the authors' released code*? (No.) Whether our code
matches the architecture the paper *describes* is a **separate** question answered in
[`DEVIATIONS.md`](DEVIATIONS.md) — do not conflate the two. A module can be
independently-written (good provenance) and still deviate from the paper's described setup
(a deviation); those deviations live in `DEVIATIONS.md`, not here.

**Summary:** every model/experiment module in this repository was written from the
papers. `torchdiffeq` is used **only as the ODE-solver library** (`from torchdiffeq
import odeint_adjoint as odeint`) — the sanctioned, expected use. None of the
tell-tale code fingerprints of the three upstream example/model codebases appear in
our modules (verified against `rtqichen/torchdiffeq` examples,
`EmilienDupont/augmented-neural-odes`, and `YuliaRubanova/latent_ode`).

## Upstream references

| Repo | Used as | Notes |
|---|---|---|
| [`rtqichen/torchdiffeq`](https://github.com/rtqichen/torchdiffeq) v0.2.5 | **Library dependency** (solver) | We import `odeint_adjoint`. We do **not** use its `examples/odenet_mnist.py` model code (`ODEfunc`, `ConcatConv2d`, `RunningAverageMeter`, the GroupNorm `norm()` helper, `inf_generator`, `learning_rate_with_decay`) or `ode_demo.py`. |
| [`EmilienDupont/augmented-neural-odes`](https://github.com/EmilienDupont/augmented-neural-odes) | **Not used as code** | We take the ANODE *method* and the conv ODE-field *architecture* from the paper (App. F). We do not use their `Conv2dTime(nn.Conv2d)` subclass, `(device, time_dependent, non_linearity)` signatures, `fc1/fc2/fc3` layout, `options={'max_num_steps'}`, in-block augmentation, or `Trainer`/`histories` bookkeeping. |
| [`YuliaRubanova/latent_ode`](https://github.com/YuliaRubanova/latent_ode) | **Not used as code** | The Latent-ODE material is out of scope (see `OUT_OF_SCOPE.md`). Even so, our (excluded) ODE-RNN code uses a stock `nn.GRUCell` rather than their custom probabilistic `GRU_unit`, and calls `odeint` directly rather than via their `DiffeqSolver`. |

## Per-file / per-module verdict — *was it copied?*

Every verdict below is **independently written** (not copied). The last column gives the
concrete code difference from the closest upstream file that a reviewer can diff.
*(Architecture-vs-paper faithfulness is in `DEVIATIONS.md`, not here.)*

| Module | File | Copied? | Concrete difference from the closest upstream code |
|---|---|---|---|
| `ODEFunc` (MLP field) | `models/continuous.py` | **No** | `nn.Sequential(Linear(in+1,h),ReLU,Linear(h,h),ReLU,Linear(h,in))`; time concat **state-first** `cat([h,t])`. Dupont: `fc1/fc2/fc3` + time-**first** `cat([t,x])`. torchdiffeq demo: cubes the input, no time/NFE. Shared bits (`self.nfe`, `forward(t,h)`) are paper/API conventions. |
| `ODEBlock` | `models/continuous.py` | **No** | Registers `integration_time` as a **buffer** (neither upstream does); no `options={'max_num_steps'}`, no `@property nfe`; augmentation kept **out** of the block; own `return_trajectory=linspace(50)` path. `return out[1]` is a paper-level idiom. |
| `ConvODEFunc` (image field) | `models/continuous.py` | **No** | Plain `nn.Conv2d` layers + an explicit `_with_time` helper concatenating the time channel **state-first** before each conv. Dupont uses a `Conv2dTime(nn.Conv2d)` **subclass** that prepends time. (Per-layer time injection matches App. F.1.2 — see `DEVIATIONS.md` B1.) |
| ANODE augmentation | `models/networks.py` | **No** | Append-zeros `cat([h,zeros(augment_dim)])` is the defining ANODE op (paper-level), placed in `ODENet.forward`, not in an `is_conv`-branching `ODEBlock` as Dupont does. |
| `EulerDiscretizedODENet` (was `DiscreteResNet`) | `models/networks.py` | **No** | Original design: one **weight-tied** `ODEFunc` reused across explicit Euler steps. No upstream weight-ties this way (Dupont/torchdiffeq use independent per-layer blocks). *(That this makes it a discretisation of the same ODE, not an independent ResNet, is a design note in `DEVIATIONS.md` D1.)* |
| NFE counter + fwd/bwd split | `models/continuous.py`, `training/engine.py` | **No** | `self.nfe += 1` + reset-before-`backward`/re-read-after is the Chen-defined NFE metric, in every Neural-ODE codebase. Our snapshot is inline in `training/engine.py`, not a Dupont `Trainer._get_and_reset_nfes` dict-of-histories. |
| Synthetic 2-D data | `data/synthetic.py` | **No** | `make_circles/make_spirals/make_moons` are our own generators (`np.random.default_rng`), not Dupont's `ConcentricSphere`/`random_point_in_sphere`. |
| Loops / NFE stats / logging | `training/engine.py`, `training/utils.py`, `training/logging_backend.py` | **No (original)** | Standard PyTorch loops; the pluggable CSV/no-op/W&B logger is original to this replication. |
| Latent-ODE / ODE-RNN (OUT OF SCOPE) | `models/ode_rnn.py`, `models/continuous.py` (Latent classes), `main.py`, … | **No** | See `OUT_OF_SCOPE.md`. Stock `nn.GRUCell` (not Rubanova's `GRU_unit`), direct `odeint` (not `DiffeqSolver`), generic `h_to_mu`/`h_to_logvar` head (not the `std.abs()` scheme). |

## How to verify

Clone the three upstream repos and diff the class/function shapes against ours; the
distinguishing patterns above (buffer vs attribute integration time, `Sequential` vs
`fc1/fc2/fc3`, `_with_time` helper vs `Conv2dTime` subclass, `nn.GRUCell` vs `GRU_unit`,
no `max_num_steps`/`Trainer`/`RunningAverageMeter`) are the concrete checks.

See [`DEVIATIONS.md`](DEVIATIONS.md) for how the (independently-written) code differs from
the **architecture described in the papers**.
