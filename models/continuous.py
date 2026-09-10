from __future__ import annotations
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Dict, Optional
from torch import Tensor


class ODEFunc(nn.Module):
    r"""Parameterizes the continuous dynamics of the hidden state.

    Computes the time-dependent vector field $f_\\theta(h(t), t)$ for the IVP:
    $ \\frac{dh(t)}{dt} = f_\\theta(h(t), t) $

    Args:
        in_features (int): Dimensionality of the hidden state $h(t)$.
        hidden_dim (int): Dimensionality of the internal hidden layers.
    """

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.nfe = 0

        self.net = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Evaluates the vector field at state $h$ and time $t$."""
        self.nfe += 1

        t_expanded = torch.ones_like(h[:, :1]) * t
        h_time = torch.cat([h, t_expanded], dim=1)

        return self.net(h_time)


class ODEBlock(nn.Module):
    r"""Integrates the ODEFunc over time $t \\in [0, 1]$ via the adjoint method.

    Utilizes the adjoint sensitivity method to allow backpropagation with $\mathcal{O}(1)$
    memory footprint.

    Args:
        ode_func (nn.Module): The neural network parameterizing the vector field.
        solver_type (str): The ODE solver algorithm (e.g., 'dopri5', 'rk4', 'euler').
        atol (float): Absolute error tolerance for adaptive solvers.
        rtol (float): Relative error tolerance for adaptive solvers.
    """

    def __init__(
        self,
        ode_func: nn.Module,
        solver_type: str = "dopri5",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        options: Optional[Dict] = None,
    ):
        super().__init__()
        self.ode_func = ode_func
        self.solver_type = solver_type
        self.atol = atol
        self.rtol = rtol
        # Solver options passed through to torchdiffeq (e.g. {"step_size": 1/N} for a
        # fixed-step solver). Enables the C4 memory-vs-NFE experiment and analytic
        # NFE tests. None -> torchdiffeq defaults.
        self.options = options

        self.register_buffer("integration_time", torch.tensor([0.0, 1.0]).float())

    def forward(self, x: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        """Solves the IVP. If return_trajectory is True, returns intermediate states."""

        if return_trajectory:
            t = torch.linspace(0.0, 1.0, steps=50).type_as(x)
        else:
            t = self.integration_time.type_as(x)

        out = odeint(
            func=self.ode_func,
            y0=x,
            t=t,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver_type,
            options=self.options,
        )

        if return_trajectory:
            return out  # Shape: (50, batch_size, dim)

        return out[1]  # Shape: (batch_size, dim)


class ConvODEFunc(nn.Module):
    """Convolutional vector field with PER-LAYER time injection (ANODE App. F.1.2).

    Following the paper, the time ``t`` is appended as an extra channel *before each
    convolution* in the 1x1 -> 3x3 -> 1x1 (64-filter) stack. Written independently
    of Dupont's released code: plain ``nn.Conv2d`` layers with an explicit
    ``_with_time`` helper that concatenates the time channel STATE-FIRST, rather
    than his ``Conv2dTime(nn.Conv2d)`` subclass (which prepends time).

    (Concatenating the time once at the input instead would be a different vector
    field. See DEVIATIONS.md.)
    """

    def __init__(self, num_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.nfe = 0
        self.conv1 = nn.Conv2d(
            num_channels + 1, hidden_channels, kernel_size=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            hidden_channels + 1, hidden_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_channels + 1, num_channels, kernel_size=1, padding=0
        )

    def _with_time(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Append a constant time channel to the feature map (state-first)."""
        t_channel = (
            torch.ones(h.size(0), 1, h.size(2), h.size(3), device=h.device, dtype=h.dtype)
            * t
        )
        return torch.cat([h, t_channel], dim=1)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        h = torch.relu(self.conv1(self._with_time(t, h)))
        h = torch.relu(self.conv2(self._with_time(t, h)))
        h = self.conv3(self._with_time(t, h))
        return h
