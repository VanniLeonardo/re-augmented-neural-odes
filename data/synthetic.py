from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_circles as sklearn_make_circles
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from dataclasses import dataclass
from torch import Tensor


def get_1d_crossing_data(batch_size: int) -> DataLoader:
    """Returns a 1D crossing dataset for sanity-checking trajectory crossings."""
    x = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def get_concentric_circles(
    batch_size: int,
    n_samples: int = 1024,
    noise: float = 0.05,
) -> DataLoader:
    """Returns a DataLoader of concentric circles via sklearn.

    Args:
        batch_size (int): Batch size for the loader.
        n_samples (int): Total number of points.
        noise (float): Std of Gaussian noise added to coordinates.
    """
    x, y = sklearn_make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def make_circles(
    n_samples: int = 1000,
    noise: float = 0.05,
    factor: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generates two concentric circles for binary classification.

    Points are sampled uniformly on $[0, 2\pi)$; the inner circle is scaled
    by `factor`. Standard toy benchmark from Chen et al. (2018).

    Args:
        n_samples (int): Total number of points across both classes.
        noise (float): Std of Gaussian noise added to $(x, y)$ coordinates.
        factor (float): Inner circle radius as fraction of outer ($r_{\text{inner}}$).
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Sample angles uniformly on $[0, 2\pi)$
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)

    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    inner = np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1) * factor

    X = np.concatenate([outer, inner], axis=0) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_spirals(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generates two interleaving Archimedean spirals for binary classification.

    Radial coordinate $r = \theta / (4\pi)$ with $\theta \in [0, 4\pi]$.
    Class 1 is rotated by $\pi$ relative to class 0.

    Args:
        n_samples (int): Total number of points across both classes.
        noise (float): Std of Gaussian noise added to $(x, y)$ coordinates.
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    # $r = \theta / (4\pi)$, so $r \in [0, 1]$ as $\theta$ spans $[0, 4\pi]$
    theta = np.linspace(0, 4 * np.pi, n_per_class)
    r = theta / (4 * np.pi)

    x0 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x1 = np.stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)], axis=1)

    X = np.concatenate([x0, x1], axis=0) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generates two interleaving crescent moons for binary classification.

    Upper moon spans $\theta \in [0, \pi]$; lower moon is offset by $(1, -0.5)$.

    Args:
        n_samples (int): Total number of points across both classes.
        noise (float): Std of Gaussian noise added to $(x, y)$ coordinates.
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    # Upper moon: $\theta \in [0, \pi]$
    theta_upper = np.linspace(0, np.pi, n_upper)
    upper = np.stack([np.cos(theta_upper), np.sin(theta_upper)], axis=1)

    # Lower moon offset by $(1, -0.5)$
    theta_lower = np.linspace(0, np.pi, n_lower)
    lower = np.stack([1 - np.cos(theta_lower), -np.sin(theta_lower) - 0.5], axis=1)

    X = np.concatenate([upper, lower], axis=0) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_upper), np.ones(n_lower)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_spheres(
    n_samples: int = 3000,
    noise: float = 0.0,
    seed: int = 42,
    r1: float = 0.5,
    r2: float = 1.0,
    r3: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Dupont's concentric-**spheres** dataset in d=2 (App. F.2.1).

    The FILLED inner disk ``||x|| <= r1`` (class 0) is enclosed by the annulus
    ``r2 <= ||x|| <= r3`` (class 1). Unlike two thin circles, the inner class is a
    solid region wrapped by the outer, so a NODE flow (a homeomorphism) composed with
    a linear classifier genuinely cannot separate them without tearing -- this is the
    hard topological bottleneck the paper's argument is about.

    Class ratio matches the paper (1000 inner : 2000 outer at n_samples=3000).
    Points are area-uniform within each region.

    Args:
        n_samples (int): total points; inner gets n//3, outer the rest (1:2).
        noise (float): std of Gaussian coordinate noise (paper uses 0).
        seed (int): RNG seed.
        r1, r2, r3 (float): inner-disk radius, annulus inner/outer radii.
    """
    rng = np.random.default_rng(seed)
    n_inner = n_samples // 3
    n_outer = n_samples - n_inner

    # Inner filled disk: area-uniform radius r = r1 * sqrt(U).
    theta_in = rng.uniform(0.0, 2 * np.pi, n_inner)
    rad_in = r1 * np.sqrt(rng.uniform(0.0, 1.0, n_inner))
    inner = np.stack([rad_in * np.cos(theta_in), rad_in * np.sin(theta_in)], axis=1)

    # Outer annulus [r2, r3]: area-uniform radius r = sqrt(U*(r3^2-r2^2)+r2^2).
    theta_out = rng.uniform(0.0, 2 * np.pi, n_outer)
    rad_out = np.sqrt(rng.uniform(0.0, 1.0, n_outer) * (r3**2 - r2**2) + r2**2)
    outer = np.stack([rad_out * np.cos(theta_out), rad_out * np.sin(theta_out)], axis=1)

    X = np.concatenate([inner, outer], axis=0)
    if noise > 0.0:
        X = X + rng.normal(0.0, noise, X.shape)
    y = np.concatenate([np.zeros(n_inner), np.ones(n_outer)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


@dataclass
class SampleTensors:
    observed_context: Tensor
    context_values: Tensor
    context_times: Tensor
    context_mask: Tensor
    interp_mask: Tensor
    full_times: Tensor
    ground_truth: Tensor
    future_mask: Tensor
