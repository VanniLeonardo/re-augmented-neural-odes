from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import ODEConfig


class IrregularSineWaveDataset(Dataset):
    """Synthetic irregular time-series dataset with batch fields."""

    def __init__(
        self,
        num_samples: int,
        context_start: float = 0.0,
        context_end: float = 5.0,
        future_end: float = 10.0,
        n_context_points: int = 20,
        n_future_points: int = 20,
        min_observed_context_points: int = 2,
        observation_prob: float = 0.7,
        noise_std: float = 0.1,
        input_dim: int = 1,
        signal_type: str = "sine",
        seed: int = 42,
    ) -> None:
        super().__init__()

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 0.0 <= context_start < context_end < future_end:
            raise ValueError("Expected context_start < context_end < future_end.")
        if n_context_points < min_observed_context_points + 1:
            raise ValueError(
                "n_context_points must exceed min_observed_context_points so "
                "interpolation targets exist."
            )
        if n_future_points <= 0:
            raise ValueError("n_future_points must be positive.")
        if not 0.0 < observation_prob < 1.0:
            raise ValueError("observation_prob must lie strictly between 0 and 1.")

        self.num_samples = num_samples
        self.context_start = context_start
        self.context_end = context_end
        self.future_end = future_end
        self.n_context_points = n_context_points
        self.n_future_points = n_future_points
        self.min_observed_context_points = min_observed_context_points
        self.observation_prob = observation_prob
        self.noise_std = noise_std
        self.input_dim = input_dim
        self.signal_type = signal_type

        generator = torch.Generator().manual_seed(seed)
        self.samples = [self._generate_sample(generator) for _ in range(num_samples)]

    def _sample_context_and_future_times(
        self,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples strictly increasing context/future times."""
        eps = 1e-4

        n_context_inner = self.n_context_points - 1
        if n_context_inner > 0:
            context_inner = (
                self.context_start
                + eps
                + (self.context_end - self.context_start - 2 * eps)
                * torch.rand(n_context_inner, generator=generator)
            )
            context_inner = torch.sort(context_inner).values
            context_times = torch.cat(
                [torch.tensor([self.context_start]), context_inner], dim=0
            ).to(torch.float32)
        else:
            context_times = torch.tensor([self.context_start], dtype=torch.float32)

        n_future_inner = self.n_future_points - 1
        if n_future_inner > 0:
            future_inner = (
                self.context_end
                + eps
                + (self.future_end - self.context_end - 2 * eps)
                * torch.rand(n_future_inner, generator=generator)
            )
            future_inner = torch.sort(future_inner).values
            future_times = torch.cat(
                [future_inner, torch.tensor([self.future_end])], dim=0
            ).to(torch.float32)
        else:
            future_times = torch.tensor([self.future_end], dtype=torch.float32)

        return context_times, future_times

    def _sample_observation_mask(self, generator: torch.Generator) -> torch.Tensor:
        """Samples a context mask and guarantees a usable interpolation task."""
        mask = (
            torch.rand(self.n_context_points, generator=generator)
            < self.observation_prob
        )

        mask[0] = True

        observed_count = int(mask.sum().item())
        if observed_count < self.min_observed_context_points:
            candidate_indices = torch.randperm(
                self.n_context_points, generator=generator
            )
            for index in candidate_indices.tolist():
                mask[index] = True
                observed_count += 1
                if observed_count >= self.min_observed_context_points:
                    break

        if int((~mask).sum().item()) == 0 and self.n_context_points > 1:
            mask[-1] = False

        return mask

    def _generate_sample(self, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Builds train/val/test dataloaders for the synthetic sine benchmark."""
        context_times, future_times = self._sample_context_and_future_times(generator)
        full_times = torch.cat([context_times, future_times], dim=0)

        ground_truth = self._generate_signal(full_times, generator)

        context_values = ground_truth[
            : self.n_context_points
        ] + self.noise_std * torch.randn(
            self.n_context_points,
            self.input_dim,
            generator=generator,
        )

        context_keep = self._sample_observation_mask(generator=generator)

        context_mask = context_keep.unsqueeze(-1).float()
        interp_mask = (~context_keep).unsqueeze(-1).float()
        future_mask = torch.ones(self.n_future_points, 1, dtype=torch.float32)

        observed_context = torch.where(
            context_mask.bool(),
            context_values,
            torch.zeros_like(context_values),
        )

        return {
            "observed_context": observed_context,
            "context_values": context_values,
            "context_times": context_times,
            "context_mask": context_mask,
            "interp_mask": interp_mask,
            "full_times": full_times,
            "ground_truth": ground_truth,
            "future_mask": future_mask,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {key: value.clone() for key, value in sample.items()}

    def _generate_signal(
        self,
        t: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Generates the clean signal on time grid t with shape [T, input_dim]."""
        signal_type = self.signal_type

        if signal_type == "sine" and self.input_dim == 1:
            phase = 2.0 * math.pi * torch.rand(1, generator=generator)
            return torch.sin(t.unsqueeze(-1) + phase)

        if signal_type == "sine" and self.input_dim > 1:
            phases = 2.0 * math.pi * torch.rand(self.input_dim, generator=generator)
            return torch.sin(t.unsqueeze(-1) + phases)

        if signal_type == "spiral":
            if self.input_dim != 2:
                raise ValueError("spiral requires input_dim=2")

            omega = 0.5 + 1.5 * torch.rand(1, generator=generator)
            phase = 2.0 * math.pi * torch.rand(1, generator=generator)
            alpha = 0.05 + 0.10 * torch.rand(1, generator=generator)
            r0 = 0.8 + 0.4 * torch.rand(1, generator=generator)

            r = r0 * torch.exp(-alpha * t)
            x = r * torch.cos(omega * t + phase)
            y = r * torch.sin(omega * t + phase)
            return torch.stack([x, y], dim=-1)

        if signal_type == "damped":
            if self.input_dim != 2:
                raise ValueError("damped requires input_dim=2")

            omega1 = 0.5 + 1.5 * torch.rand(1, generator=generator)
            omega2 = omega1 * (1.0 + 0.3 * torch.rand(1, generator=generator))
            phase1 = 2.0 * math.pi * torch.rand(1, generator=generator)
            phase2 = 2.0 * math.pi * torch.rand(1, generator=generator)
            alpha = 0.05 + 0.05 * torch.rand(1, generator=generator)

            damp = torch.exp(-alpha * t)
            x = damp * torch.sin(omega1 * t + phase1)
            y = damp * torch.sin(omega2 * t + phase2)
            return torch.stack([x, y], dim=-1)

        raise ValueError(
            f"unknown signal_type={signal_type!r} for input_dim={self.input_dim}"
        )


def get_irregular_sine_dataloaders(
    config: ODEConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train/val/test dataloaders for the synthetic irregular benchmark"""
    full_dataset = IrregularSineWaveDataset(
        num_samples=config.train_size + config.val_size + config.test_size,
        context_start=config.context_start,
        context_end=config.context_end,
        future_end=config.future_end,
        n_context_points=config.n_context_points,
        n_future_points=config.n_future_points,
        min_observed_context_points=config.min_observed_context_points,
        observation_prob=config.observation_prob,
        noise_std=config.noise_std,
        input_dim=config.input_dim,
        signal_type=config.signal_type,
        seed=config.seed,
    )

    split_generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [config.train_size, config.val_size, config.test_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
