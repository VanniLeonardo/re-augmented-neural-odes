from __future__ import annotations
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Dict, Optional, Tuple
from torch import Tensor
from config import LatentODEConfig


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
    ):
        super().__init__()
        self.ode_func = ode_func
        self.solver_type = solver_type
        self.atol = atol
        self.rtol = rtol

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
        )

        if return_trajectory:
            return out  # Shape: (50, batch_size, dim)

        return out[1]  # Shape: (batch_size, dim)


class ConvODEFunc(nn.Module):
    """Convolutional Vector Field for Image Data."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.nfe = 0
        # Architecture strictly following the ANODE paper Appendix F.1.2
        self.net = nn.Sequential(
            nn.Conv2d(num_channels + 1, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_channels, kernel_size=1, padding=0),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        # t is a scalar. Expand it to match the (Batch, 1, H, W) shape of h
        t_expanded = torch.ones(h.size(0), 1, h.size(2), h.size(3), device=h.device) * t
        h_time = torch.cat([h, t_expanded], dim=1)  # Concatenate on channel dimension
        return self.net(h_time)


class LatentODEFunc(nn.Module):
    """Latent vector field f_theta(z)."""

    def __init__(self, latent_dim: int, nhidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, nhidden),
            nn.ELU(),
            nn.Linear(nhidden, latent_dim),
        )
        self.nfe = 0

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        self.nfe += 1
        return self.net(z)


class EncoderODEFunc(nn.Module):
    """Encoder-side hidden dynamics f_enc(h)."""

    def __init__(self, nhidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nhidden, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden),
        )
        self.nfe = 0

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        self.nfe += 1
        return self.net(h)


class ODERNNEncoder(nn.Module):
    """Reverse-time ODE-RNN encoder using x_i, m_i, and t_i."""

    def __init__(
        self, input_dim: int, nhidden: int, latent_dim: int, config: LatentODEConfig
    ):
        super().__init__()
        self.nhidden = nhidden
        self.config = config
        self.ode_func = EncoderODEFunc(nhidden)
        self.rnn_cell = nn.GRUCell(input_dim * 2 + 1, nhidden)
        self.h_to_mu = nn.Linear(nhidden, latent_dim)
        self.h_to_logvar = nn.Linear(nhidden, latent_dim)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, int]:
        self.ode_func.reset_nfe()
        h = torch.zeros(1, self.nhidden, device=x.device, dtype=x.dtype)

        for i in range(t.size(0) - 1, -1, -1):
            if i < t.size(0) - 1:
                dt = t[i + 1] - t[i]
                if torch.abs(dt) > 1e-10:
                    t_span = torch.stack([t[i + 1], t[i]], dim=0)
                    h = odeint(
                        self.ode_func,
                        h,
                        t_span,
                        method=self.config.encoder_method,
                        rtol=self.config.rtol,
                        atol=self.config.atol,
                    )[-1]

            time_feature = t[i].view(1, 1)
            rnn_input = torch.cat(
                [x[i].view(1, -1), mask[i].view(1, -1), time_feature], dim=-1
            )
            h = self.rnn_cell(rnn_input, h)

        mu = self.h_to_mu(h).squeeze(0)
        logvar = self.h_to_logvar(h).squeeze(0)
        return mu, logvar, self.ode_func.nfe


class StandardGRUEncoder(nn.Module):
    """Discrete-time GRU baseline fed with x_i, m_i, and t_i."""

    def __init__(self, input_dim: int, nhidden: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim * 2 + 1, nhidden, batch_first=True)
        self.h_to_mu = nn.Linear(nhidden, latent_dim)
        self.h_to_logvar = nn.Linear(nhidden, latent_dim)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, int]:
        time_feature = t.unsqueeze(-1)
        features = torch.cat([x, mask, time_feature], dim=-1).unsqueeze(0)
        _, h_n = self.gru(features)
        h_last = h_n.squeeze(0)
        mu = self.h_to_mu(h_last).squeeze(0)
        logvar = self.h_to_logvar(h_last).squeeze(0)
        return mu, logvar, 0


class VanillaGRUEncoder(nn.Module):
    """
    Discrete-time GRU baseline fed ONLY with x_i and m_i (no timestamp).
    This is the classic "naive RNN" baseline used in the Latent ODE paper,
    which has no mechanism for handling irregular sampling intervals.
    """

    def __init__(self, input_dim: int, nhidden: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim * 2, nhidden, batch_first=True)
        self.h_to_mu = nn.Linear(nhidden, latent_dim)
        self.h_to_logvar = nn.Linear(nhidden, latent_dim)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, int]:
        features = torch.cat([x, mask], dim=-1).unsqueeze(0)
        _, h_n = self.gru(features)
        h_last = h_n.squeeze(0)
        mu = self.h_to_mu(h_last).squeeze(0)
        logvar = self.h_to_logvar(h_last).squeeze(0)
        return mu, logvar, 0


class LatentODE(nn.Module):
    """Latent ODE model with faithful Phase 4 evaluation hooks."""

    def __init__(self, config: LatentODEConfig):
        super().__init__()
        self.config = config

        encoder_type = getattr(config, "encoder_type", None)
        if encoder_type is None:
            encoder_type = "odernn" if config.use_ode_rnn else "gru_time"

        if encoder_type == "odernn":
            self.encoder: nn.Module = ODERNNEncoder(
                input_dim=config.input_dim,
                nhidden=config.nhidden,
                latent_dim=config.latent_dim,
                config=config,
            )
        elif encoder_type == "gru_time":
            self.encoder = StandardGRUEncoder(
                input_dim=config.input_dim,
                nhidden=config.nhidden,
                latent_dim=config.latent_dim,
            )
        elif encoder_type == "gru_notime":
            self.encoder = VanillaGRUEncoder(
                input_dim=config.input_dim,
                nhidden=config.nhidden,
                latent_dim=config.latent_dim,
            )
        else:
            raise ValueError(
                f"unknown encoder_type={encoder_type}; "
                f"expected one of 'odernn', 'gru_time', 'gru_notime'"
            )

        self.latent_ode_func = LatentODEFunc(config.latent_dim, config.nhidden)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.nhidden),
            nn.ReLU(),
            nn.Linear(config.nhidden, config.input_dim),
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _solve_latent_batched(
        self, z0_batch: Tensor, t_query_batch: Tensor
    ) -> Tuple[Tensor, int]:
        """
        Solve the latent ODE for an entire batch in a single odeint call.

        z0_batch:       (B, latent_dim)
        t_query_batch:  (B, T)  -- each row is the query grid for one sample

        Returns:
            pred_latent: (B, T, latent_dim)
            nfe:         total function evaluations for this batch
        """
        self.latent_ode_func.reset_nfe()
        B, T = t_query_batch.shape

        flat_times = t_query_batch.reshape(-1)
        pooled_times, inverse = torch.unique(
            flat_times, sorted=True, return_inverse=True
        )

        z_pooled = odeint(
            self.latent_ode_func,
            z0_batch,
            pooled_times,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )
        inverse = inverse.view(B, T)
        batch_idx = torch.arange(B, device=z0_batch.device).unsqueeze(1).expand(B, T)
        pred_latent = z_pooled[inverse, batch_idx, :]

        return pred_latent, self.latent_ode_func.nfe

    def forward(
        self,
        x: Tensor,
        t_obs: Tensor,
        mask: Tensor,
        t_query: Optional[Tensor] = None,
        sample_latent: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        if t_query is None:
            t_query = t_obs
        if sample_latent is None:
            sample_latent = self.training and self.config.is_variational

        batch_size = x.size(0)

        mu_list, logvar_list = [], []
        total_encoder_nfe = 0
        for b in range(batch_size):
            mu, logvar, enc_nfe = self.encoder(x[b], t_obs[b], mask[b])
            mu_list.append(mu)
            logvar_list.append(logvar)
            total_encoder_nfe += enc_nfe

        mu_tensor = torch.stack(mu_list, dim=0)
        logvar_tensor = torch.stack(logvar_list, dim=0)

        z0 = (
            self.reparameterize(mu_tensor, logvar_tensor)
            if sample_latent
            else mu_tensor
        )

        z_t, latent_nfe = self._solve_latent_batched(z0, t_query)

        pred_tensor = self.decoder(z_t)

        return {
            "predictions": pred_tensor,
            "mu": mu_tensor,
            "logvar": logvar_tensor,
            "nfe": torch.tensor(total_encoder_nfe + latent_nfe, device=x.device),
            "nfe_encoder": torch.tensor(total_encoder_nfe, device=x.device),
            "nfe_latent": torch.tensor(latent_nfe, device=x.device),
        }
