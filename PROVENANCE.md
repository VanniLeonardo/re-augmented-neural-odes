# PROVENANCE

ReScience C requires that the reimplementation be derived from the **papers**, not
copied from the authors' released code. This file records, per module, whether the
code was *written from the paper*, *adapted from* a specific upstream repository, or
is a *library dependency*. Reviewers can (and should) diff against the upstream repos.

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
| [`YuliaRubanova/latent_ode`](https://github.com/YuliaRubanova/latent_ode) | **Not used as code** | The Latent-ODE material is out of scope (see `extra/README.md`). Even so, our (excluded) ODE-RNN code uses a stock `nn.GRUCell` rather than their custom probabilistic `GRU_unit`, and calls `odeint` directly rather than via their `DiffeqSolver`. |

## Per-file / per-module verdict

| Module | File | Verdict | Distinguishing evidence vs upstream |
|---|---|---|---|
| `ODEFunc` (MLP field) | `models/continuous.py` | **written-from-paper** | `nn.Sequential(Linear(in+1,h), ReLU, Linear(h,h), ReLU, Linear(h,in))`; time concatenated **state-first** `cat([h, t])`. Dupont uses `fc1/fc2/fc3` + time-**first** `cat([t, x])`; torchdiffeq's demo cubes the input and has no time/NFE. Only the conventional `self.nfe` counter and the API-mandated `forward(t, h)` are shared (paper-level, not fingerprints). |
| `ODEBlock` | `models/continuous.py` | **written-from-paper** | Registers `integration_time` as a **buffer** (neither upstream does); no `options={'max_num_steps'}`, no `@property nfe`; augmentation kept **out** of the block; original `return_trajectory=linspace(50)` path. `return out[1]` is a paper-level idiom. |
| `ConvODEFunc` (image field) | `models/continuous.py` | **architecture from ANODE App. F.1.2; implementation independent** | 1×1→3×3→1×1 @ 64 filters matches the paper. Written with plain `nn.Conv2d` in a `Sequential` and a **single** input-time concat, vs Dupont's `Conv2dTime` subclass injecting time into **every** conv (opposite order). Code comment credits the appendix. |
| ANODE augmentation | `models/networks.py` (`ODENet.forward`) | **from-paper, independently placed** | Append-zeros `cat([h, zeros(augment_dim)])` is the defining ANODE op (paper-level). We augment **after** a `Linear+Tanh` downsampling stem that Dupont's `ODENet` does not have. |
| `DiscreteResNet` | `models/networks.py` | **original design** | Weight-shares one `ODEFunc` across explicit Euler steps (`h + dt·f`, `dt=1/L`) so the param count matches the ODE-Net. No upstream weight-shares this way (Dupont/torchdiffeq use independent per-layer blocks). |
| NFE counter + fwd/bwd split | `models/continuous.py`, `training/engine.py` | **conventional / from-paper** | `self.nfe += 1` and reset-before-`backward` / re-read-after is the Chen-defined NFE metric, present in every Neural-ODE codebase — **not** evidence of copying. Our snapshot lives inline in `training/engine.py`, not in a Dupont-style `Trainer._get_and_reset_nfes` dict-of-histories. |
| Synthetic 2-D data | `data/synthetic.py` | **written-from-paper / standard** | `make_circles/make_spirals/make_moons` are our own generators (uniform-angle sampling, `np.random.default_rng`), not Dupont's `ConcentricSphere`/`random_point_in_sphere`. (Planned Stage-C change: align the circles geometry to Dupont's filled-disk + annulus "spheres"; this file's provenance is unaffected.) |
| Training/eval loops, NFE stats, logging | `training/engine.py`, `training/utils.py`, `training/logging_backend.py` | **written-from-scratch (this project)** | Standard PyTorch loops; the pluggable CSV/no-op/W&B logger is original to this replication. |
| Latent-ODE / ODE-RNN (OUT OF SCOPE) | `models/ode_rnn.py`, `models/continuous.py` (Latent classes), `main.py`, `training/timeseries_engine.py`, `data/timeseries.py` | **written-from-paper (concept), independent code; excluded** | See `extra/README.md`. Uses stock `nn.GRUCell` (not Rubanova's `GRU_unit`), direct `odeint` (not `DiffeqSolver`), generic `h_to_mu`/`h_to_logvar` VAE head (not the `std.abs()` scheme). |

## How to verify

Clone the three upstream repos and diff the class/function shapes against ours; the
distinguishing patterns above (buffer vs attribute integration time, `Sequential` vs
`fc1/fc2/fc3`, single vs per-conv time injection, `nn.GRUCell` vs `GRU_unit`, no
`max_num_steps`/`Trainer`/`RunningAverageMeter`) are the concrete checks.
