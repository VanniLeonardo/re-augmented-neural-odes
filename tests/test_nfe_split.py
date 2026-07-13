"""Adversarial tests for the forward/backward NFE split.

The promoted C2 headline ("backward NFE ~ forward NFE, not Chen's 1/2") rests
ENTIRELY on the correctness of our NFE counter and its forward/backward split.
These tests therefore assert the counts against values that are known
*analytically*, exactly (not approximately), for torchdiffeq 0.2.5:

  fixed-step forward NFE over t=[0,1] with step_size = 1/N:
      euler -> N,  midpoint -> 2N,  rk4 -> 4N          (one field eval per stage)
  adjoint backward (fixed-step rk4, N steps):
      backward NFE == forward NFE == 4N                (the C2 mechanism at unit level)

If any of these fail, the C2 counter is mis-instrumented and the headline must be
withdrawn -- that is the point of pinning them here.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint, odeint_adjoint

from models.continuous import ODEFunc, ODEBlock
from models.networks import ODENet
from training.engine import train_epoch

T01 = torch.tensor([0.0, 1.0])
# field evals per step for each fixed-step method (torchdiffeq 0.2.5)
_STAGES = {"euler": 1, "midpoint": 2, "rk4": 4}


def _fresh_func(dim: int = 2) -> ODEFunc:
    return ODEFunc(in_features=dim, hidden_dim=8)


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
@pytest.mark.parametrize("n_steps", [1, 3, 5, 10])
def test_forward_nfe_is_exactly_analytic(method: str, n_steps: int) -> None:
    """Forward NFE == stages * N exactly for a fixed-step solver."""
    func = _fresh_func()
    y0 = torch.randn(4, 2)
    assert func.nfe == 0
    _ = odeint(func, y0, T01, method=method, options={"step_size": 1.0 / n_steps})
    assert func.nfe == _STAGES[method] * n_steps


@pytest.mark.parametrize("n_steps", [3, 5, 10])
def test_odeint_and_adjoint_have_identical_forward_nfe(n_steps: int) -> None:
    """The adjoint's forward integration costs the same as plain odeint."""
    opts = {"step_size": 1.0 / n_steps}
    f1 = _fresh_func()
    _ = odeint(f1, torch.randn(4, 2), T01, method="rk4", options=opts)
    f2 = _fresh_func()
    _ = odeint_adjoint(f2, torch.randn(4, 2), T01, method="rk4", options=opts)
    assert f1.nfe == f2.nfe == 4 * n_steps


@pytest.mark.parametrize("n_steps", [3, 5, 10])
def test_adjoint_backward_nfe_exact_and_split_recovers_it(n_steps: int) -> None:
    """Fixed-step rk4 adjoint: forward == backward == 4N, recovered by snapshotting.

    This is the C2 mechanism at the unit level: torchdiffeq solves the augmented
    reverse-time system (state + adjoint + parameter sensitivities) with the same
    fixed-step structure as the forward, so backward NFE == forward NFE (NOT 1/2).
    """
    func = _fresh_func()
    y0 = torch.randn(4, 2, requires_grad=True)
    out = odeint_adjoint(func, y0, T01, method="rk4", options={"step_size": 1.0 / n_steps})

    forward_nfe = func.nfe  # snapshot after forward, before backward
    out[-1].sum().backward()
    backward_nfe = func.nfe - forward_nfe  # engine's split arithmetic

    assert forward_nfe == 4 * n_steps
    assert backward_nfe == 4 * n_steps
    assert backward_nfe == forward_nfe  # the finding: ratio 1.0, not 0.5


@pytest.mark.parametrize("n_steps", [4, 8])
def test_engine_snapshot_recovers_split_end_to_end(n_steps: int) -> None:
    """train_epoch's forward/backward NFE means equal the analytic 4N on a
    fixed-step ODE-Net -- proves OUR engine wiring, not just a bare odeint call."""
    torch.manual_seed(0)
    x = torch.randn(8, 2)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    model = ODENet(
        data_dim=2,
        hidden_dim=2,
        num_classes=2,
        solver_type="rk4",
        use_stem=False,
        solver_options={"step_size": 1.0 / n_steps},
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: don't change the field
    metrics = train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"))

    assert metrics["forward_nfe_mean"] == pytest.approx(4 * n_steps)
    assert metrics["backward_nfe_mean"] == pytest.approx(4 * n_steps)


def test_forward_reset_makes_cycles_independent() -> None:
    """ODENet.forward resets nfe=0, so repeated forward+backward cycles do not
    accumulate (the reset semantics the engine relies on)."""
    model = ODENet(
        data_dim=2, hidden_dim=2, num_classes=2, solver_type="rk4",
        use_stem=False, solver_options={"step_size": 0.25},  # N=4 -> 16 fwd
    )
    counts = []
    for _ in range(3):
        logits = model(torch.randn(6, 2))
        fwd = model.ode_func.nfe
        logits.sum().backward()
        bwd = model.ode_func.nfe - fwd
        counts.append((fwd, bwd))
    assert counts[0] == counts[1] == counts[2] == (16, 16)


def test_shared_odefunc_sums_across_blocks() -> None:
    """A single ODEFunc reused across TWO ODE blocks accumulates NFE across both
    (per-module counter). Documented adversarially: to attribute NFE to one solve
    you MUST reset around it -- which is why the engine/experiments use one block
    per measured field and reset in forward. Our models never share a func across
    blocks, so this does not corrupt real measurements."""
    shared = _fresh_func()
    block_a = ODEBlock(shared, solver_type="rk4", options={"step_size": 0.25})  # 16
    block_b = ODEBlock(shared, solver_type="euler", options={"step_size": 0.2})  # 5

    shared.nfe = 0
    block_a(torch.randn(3, 2))
    after_a = shared.nfe
    block_b(torch.randn(3, 2))
    after_b = shared.nfe

    assert after_a == 16
    assert after_b == 16 + 5  # summed, not reset between blocks
