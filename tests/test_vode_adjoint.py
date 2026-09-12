"""The VODE adjoint must agree with direct backpropagation, or its evaluation counts mean
nothing. A tiny field, a tight tolerance, on the CPU."""
import torch
from torchdiffeq import odeint

from models.continuous import ODEFunc
from scripts.run_c2_vode import vode_adjoint
from training.utils import set_seed


def test_vode_adjoint_matches_direct_backprop() -> None:
    set_seed(0)
    func = ODEFunc(in_features=2, hidden_dim=8)
    y0 = torch.randn(4, 2)

    grad, fwd, bwd, recon = vode_adjoint(func, y0, tol=1e-9, nsteps=100_000)

    func.zero_grad(set_to_none=True)
    out = odeint(func, y0, torch.tensor([0.0, 1.0]), method="dopri5", rtol=1e-11, atol=1e-11)
    out[1].pow(2).sum().backward()
    ref = torch.cat([p.grad.flatten() for p in func.parameters() if p.grad is not None])

    assert (grad - ref).norm() / ref.norm() < 1e-3
    assert fwd > 0 and bwd > 0
    assert recon < 1e-6  # the round trip returns to y0 at this tolerance
