"""Stage B correctness invariants a ReScience reviewer probes for a Neural ODE
reproduction: adjoint gradient correctness, augmentation zero-init/shape,
parameter-count parity, and seed determinism (with documented non-determinism).
"""

import pytest
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from data.synthetic import make_spheres
from models.continuous import ODEFunc
from models.networks import ODENet, EulerDiscretizedODENet
from training.utils import set_seed


def test_spheres_geometry_matches_dupont() -> None:
    """Dupont App. F.2.1: filled inner disk ||x||<=0.5 (class 0) enclosed by the
    annulus 1.0<=||x||<=1.5 (class 1), 1000 inner : 2000 outer."""
    x, y = make_spheres(3000, noise=0.0, seed=0)
    r = x.norm(dim=1)
    assert int((y == 0).sum()) == 1000 and int((y == 1).sum()) == 2000  # 1:2 ratio
    assert r[y == 0].max().item() <= 0.5 + 1e-5  # inner is a filled disk
    assert r[y == 1].min().item() >= 1.0 - 1e-5  # outer annulus inner radius
    assert r[y == 1].max().item() <= 1.5 + 1e-5  # outer annulus outer radius


def test_head_hidden_dim_builds_linear_or_mlp() -> None:
    """Default head is a single Linear (faithful to Dupont); head_hidden_dim -> MLP
    (diagnostic contrast only)."""
    linear = ODENet(data_dim=2, hidden_dim=2, num_classes=2, use_stem=False)
    mlp = ODENet(data_dim=2, hidden_dim=2, num_classes=2, use_stem=False, head_hidden_dim=16)
    assert isinstance(linear.fc, nn.Linear)
    assert isinstance(mlp.fc, nn.Sequential)


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _param_grads(fn, method, dtype, **kw):
    """Return (param grads, input grad) for one gradient method."""
    torch.manual_seed(0)
    f = ODEFunc(in_features=2, hidden_dim=8).to(dtype)
    torch.manual_seed(1)
    y0 = torch.randn(6, 2, dtype=dtype, requires_grad=True)
    t = torch.tensor([0.0, 1.0], dtype=dtype)
    out = fn(f, y0, t, method=method, **kw)
    out[-1].pow(2).sum().backward()
    return [p.grad.clone() for p in f.parameters()], y0.grad.clone()


# ---------------------------------------------------------------------------
# Adjoint gradient correctness.
# ---------------------------------------------------------------------------
def test_adjoint_matches_direct_backprop_fixed_step() -> None:
    """For a FIXED-STEP rk4 solver, direct backprop through the solver is the exact
    gradient of the discretised forward; the adjoint must match it. In float64 they
    agree to ~1e-7. Tolerance asserted: rtol=1e-5, atol=1e-6."""
    kw = dict(options={"step_size": 1.0 / 20})
    g_adj, gy_adj = _param_grads(odeint_adjoint, "rk4", torch.float64, **kw)
    g_dir, gy_dir = _param_grads(odeint, "rk4", torch.float64, **kw)
    for a, d in zip(g_adj, g_dir):
        assert torch.allclose(a, d, rtol=1e-5, atol=1e-6)
    assert torch.allclose(gy_adj, gy_dir, rtol=1e-5, atol=1e-6)


def test_adaptive_adjoint_differs_from_direct_backprop() -> None:
    """CAVEAT (documented, not a bug): for an ADAPTIVE solver, the continuous
    adjoint (optimise-then-discretise) does NOT equal backprop through the solver
    (discretise-then-optimise); forward and backward take different step sequences.
    We assert a real, bounded discrepancy so this property is pinned and visible."""
    kw = dict(rtol=1e-8, atol=1e-8)
    g_adj, _ = _param_grads(odeint_adjoint, "dopri5", torch.float64, **kw)
    g_dir, _ = _param_grads(odeint, "dopri5", torch.float64, **kw)
    max_abs = max((a - d).abs().max().item() for a, d in zip(g_adj, g_dir))
    # They agree only loosely (~1e-3 here), not to fixed-step precision (~1e-7)...
    assert max_abs > 1e-5
    # ...but the discrepancy is bounded (not a blow-up / sign error).
    assert max_abs < 1.0


# ---------------------------------------------------------------------------
# Augmentation: zero-init and shape.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("augment_dim", [1, 2, 5])
def test_augment_dims_initialise_to_zero(augment_dim: int) -> None:
    """The appended ANODE channels must start at exactly 0 (data-space toy setup)."""
    data_dim = 2
    model = ODENet(
        data_dim=data_dim, hidden_dim=data_dim, num_classes=2,
        augment_dim=augment_dim, ode_hidden_dim=8, use_stem=False,
    )
    traj = model(torch.randn(4, data_dim), return_trajectory=True)  # (50, B, ode_dim)
    initial_state = traj[0]  # t = 0
    assert model.ode_dim == data_dim + augment_dim
    assert torch.all(initial_state[:, data_dim:] == 0.0)  # augmented dims are zero
    assert torch.allclose(initial_state[:, :data_dim], torch.zeros_like(initial_state[:, :data_dim])) is False


def test_no_stem_integrates_in_data_space() -> None:
    """use_stem=False => identity 'stem', ODE state dimension == data_dim."""
    model = ODENet(data_dim=2, hidden_dim=99, num_classes=2, augment_dim=0, use_stem=False)
    assert isinstance(model.downsampling, nn.Identity)
    assert model.ode_dim == 2  # data_dim, not hidden_dim


# ---------------------------------------------------------------------------
# Parameter-count parity (assert it, across widths and depths).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_dim", [16, 64, 160])
@pytest.mark.parametrize("num_layers", [2, 5, 50])
def test_odenet_euler_param_parity(hidden_dim: int, num_layers: int) -> None:
    """The Euler baseline is parameter-matched to the MLP ODE-Net, independent of
    depth (because it weight-ties one ODEFunc)."""
    ode = ODENet(data_dim=784, hidden_dim=hidden_dim, num_classes=10, use_stem=True)
    euler = EulerDiscretizedODENet(
        data_dim=784, hidden_dim=hidden_dim, num_classes=10, num_layers=num_layers
    )
    assert _count(ode) == _count(euler)


def test_euler_param_count_is_depth_independent() -> None:
    counts = {
        L: _count(EulerDiscretizedODENet(784, 160, 10, num_layers=L))
        for L in (5, 200, 1000)
    }
    assert len(set(counts.values())) == 1  # identical across depths (weight-tied)


# ---------------------------------------------------------------------------
# Determinism (and what is NOT deterministic).
# ---------------------------------------------------------------------------
def test_seed_determinism_cpu_fixed_step() -> None:
    """Same seed => bit-identical output on CPU with a FIXED-STEP solver.

    NOT asserted (documented non-determinism): adaptive solvers (dopri5) + cuDNN on
    GPU are not bit-exact across runs/drivers, because the adaptive step controller
    and non-associative float reductions differ. Reproduction there is to a
    tolerance, stated in the README, not bit-exact.
    """
    def build_and_run():
        set_seed(0)
        model = ODENet(
            data_dim=2, hidden_dim=2, num_classes=2, solver_type="rk4",
            use_stem=False, solver_options={"step_size": 0.1},
        )
        set_seed(1)
        x = torch.randn(8, 2)
        return model(x)

    out_a = build_and_run()
    out_b = build_and_run()
    assert torch.equal(out_a, out_b)
