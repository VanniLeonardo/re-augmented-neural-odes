"""Tolerance discipline as an executable invariant.

A Neural ODE result (an accuracy or an NFE) is only meaningful if the solver is
actually integrating the field. The decisive check is invertibility: the flow of an
ODE is a homeomorphism, so forward 0->1 then backward 1->0 must reconstruct the input
to solver tolerance. On the near-singular fields the nested-spheres task induces, the
code's historical default atol=rtol=1e-3 does NOT reconstruct (it is non-integrating),
while an accurate tolerance does. These tests pin both facts so no future experiment
silently reports numbers from a solver that is not solving.

This is the falsification check the standing rule requires attached to every NFE/acc
claim on this task.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint

from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import set_seed


def _reconstruction_error(ode_func, X, tol):
    """max ||back(fwd(x)) - x|| for the flow of ode_func, solved at `tol` (dopri5)."""
    with torch.no_grad():
        fwd = odeint(ode_func, X, torch.tensor([0.0, 1.0]), method="dopri5",
                     atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
        back = odeint(ode_func, fwd, torch.tensor([1.0, 0.0]), method="dopri5",
                      atol=tol, rtol=tol, options={"max_num_steps": 10_000_000})[1]
    return (back - X).norm(dim=1).max().item()


def _train_stiff_spheres_field():
    """A field that has learned to separate the nested spheres (hence near-singular),
    trained quickly and deterministically with a fixed-step solver so the test is fast."""
    set_seed(0)
    tr, va = get_dataloaders("spheres", n_samples=1200, batch_size=64,
                             val_split=0.2, noise=0.05, seed=0)
    model = ODENet(data_dim=2, hidden_dim=2, num_classes=2, augment_dim=0,
                   ode_hidden_dim=32, use_stem=False, head_hidden_dim=None,
                   solver_type="rk4", solver_options={"step_size": 1.0 / 20})
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(60):
        train_epoch(model, tr, opt, crit, torch.device("cpu"))
    Xval = torch.stack([va.dataset[i][0] for i in range(len(va.dataset))])
    return model, Xval


def test_accurate_tolerance_reconstructs_the_flow():
    """At an accurate tolerance the trained flow is invertible (reconstructs ~= 0).

    This is the check every faithful experiment must pass: if it fails, the reported
    numbers are not measuring a flow.
    """
    model, Xval = _train_stiff_spheres_field()
    err_accurate = _reconstruction_error(model.ode_func, Xval, tol=1e-7)
    assert err_accurate < 1e-2, f"flow not integrable even at 1e-7 (err={err_accurate:.2e})"


def test_loose_default_tolerance_does_not_integrate_this_field():
    """The historical default atol=rtol=1e-3 does NOT reconstruct the stiff field:
    it is non-integrating, so any NFE/accuracy measured there is an artifact. Pinned
    so the loose default can never quietly come back as 'faithful'."""
    model, Xval = _train_stiff_spheres_field()
    err_accurate = _reconstruction_error(model.ode_func, Xval, tol=1e-7)
    err_loose = _reconstruction_error(model.ode_func, Xval, tol=1e-3)
    # loose reconstruction is dramatically worse than accurate (orders of magnitude)
    assert err_loose > 10 * err_accurate
    assert err_loose > 1e-2  # absolute: a real, visible failure to return to the input
