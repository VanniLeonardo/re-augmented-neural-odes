# Provenance: was the authors' code copied?

This file answers one question. Did we copy the released code of the original authors? No.

Whether our implementation matches the architecture the papers *describe* is a different
question, answered in [`DEVIATIONS.md`](DEVIATIONS.md). The two should not be conflated. A
module can be written independently, which is good provenance, and still depart from the
setup a paper describes, which is a deviation.

Every model and experiment module here was written from the papers. `torchdiffeq` is used
only as the solver library (`from torchdiffeq import odeint_adjoint as odeint`), which is its
intended use. We checked our modules against three upstream codebases and none of their
identifying patterns appear: `rtqichen/torchdiffeq`, `EmilienDupont/augmented-neural-odes`
and `YuliaRubanova/latent_ode`.

## Upstream repositories

| Repository | Used as | Notes |
|---|---|---|
| [`rtqichen/torchdiffeq`](https://github.com/rtqichen/torchdiffeq) v0.2.5 | Library dependency | We import `odeint_adjoint`. We do not use the model code in `examples/odenet_mnist.py` (`ODEfunc`, `ConcatConv2d`, `RunningAverageMeter`, the GroupNorm `norm()` helper, `inf_generator`, `learning_rate_with_decay`) or `ode_demo.py`. |
| [`EmilienDupont/augmented-neural-odes`](https://github.com/EmilienDupont/augmented-neural-odes) | Not used as code | We take the augmentation method and the convolutional field from the paper, Appendix F. We do not use their `Conv2dTime(nn.Conv2d)` subclass, their `(device, time_dependent, non_linearity)` signatures, the `fc1/fc2/fc3` layout, `options={'max_num_steps'}`, in-block augmentation, or the `Trainer` and `histories` bookkeeping. |
| [`YuliaRubanova/latent_ode`](https://github.com/YuliaRubanova/latent_ode) | Not used as code | This material is out of scope, see [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md). Even so, our excluded ODE-RNN code uses a stock `nn.GRUCell` rather than their probabilistic `GRU_unit`, and calls `odeint` directly rather than through their `DiffeqSolver`. |

## Per-module verdict

Every module below was written independently. The last column gives a concrete difference
from the closest upstream file, so a reviewer can check by diffing.

| Module | File | Copied | Difference from the closest upstream code |
|---|---|---|---|
| `ODEFunc`, the MLP field | `models/continuous.py` | No | `nn.Sequential(Linear(in+1,h), ReLU, Linear(h,h), ReLU, Linear(h,in))`, with the time concatenated state-first as `cat([h,t])`. Dupont uses `fc1/fc2/fc3` and concatenates time first. The torchdiffeq demo cubes its input and tracks neither time nor NFE. The shared parts, `self.nfe` and `forward(t,h)`, are API conventions. |
| `ODEBlock` | `models/continuous.py` | No | Registers `integration_time` as a buffer, which neither upstream does. No `options={'max_num_steps'}` and no `@property nfe`. Augmentation is kept outside the block. Adds a `return_trajectory` path over `linspace(50)`. |
| `ConvODEFunc`, the image field | `models/continuous.py` | No | Plain `nn.Conv2d` layers with a `_with_time` helper that concatenates the time channel state-first before each convolution. Dupont uses a `Conv2dTime(nn.Conv2d)` subclass that prepends time. Per-layer time injection follows Appendix F.1.2, see `DEVIATIONS.md` row B1. |
| Augmentation | `models/networks.py` | No | Appending zeros with `cat([h, zeros(augment_dim)])` is the defining operation of the method. We place it in `ODENet.forward` rather than in an `is_conv`-branching `ODEBlock`. |
| `EulerDiscretizedODENet` | `models/networks.py` | No | Our own design: one weight-tied `ODEFunc` reused across explicit Euler steps. No upstream ties weights this way. That this makes it a discretisation of the same ODE rather than an independent ResNet is discussed in `DEVIATIONS.md` row D1. |
| NFE counter and forward/backward split | `models/continuous.py`, `training/engine.py` | No | `self.nfe += 1`, reset before `backward` and re-read after, is the metric Chen defines and appears in every Neural ODE codebase. Ours is inline in `training/engine.py` rather than a `Trainer._get_and_reset_nfes` returning histories. |
| Synthetic 2-D data | `data/synthetic.py` | No | `make_circles`, `make_spirals` and `make_moons` are our own generators built on `np.random.default_rng`, not Dupont's `ConcentricSphere` and `random_point_in_sphere`. |
| Loops, NFE statistics, logging | `training/engine.py`, `training/utils.py`, `training/logging_backend.py` | No | Standard PyTorch loops. The pluggable CSV, no-op and Weights and Biases logger is original to this replication. |
| Latent ODE and ODE-RNN (out of scope) | `models/ode_rnn.py`, Latent classes in `models/continuous.py`, `main.py` | No | See `OUT_OF_SCOPE.md`. Stock `nn.GRUCell` rather than Rubanova's `GRU_unit`, direct `odeint` rather than `DiffeqSolver`, and a generic `h_to_mu` and `h_to_logvar` head rather than their `std.abs()` scheme. |

## How to check

Clone the three upstream repositories and diff the class and function shapes against ours.
The distinguishing patterns are concrete: a buffer rather than an attribute for the
integration time, `Sequential` rather than `fc1/fc2/fc3`, a `_with_time` helper rather than a
`Conv2dTime` subclass, `nn.GRUCell` rather than `GRU_unit`, and the absence of
`max_num_steps`, `Trainer` and `RunningAverageMeter`.
