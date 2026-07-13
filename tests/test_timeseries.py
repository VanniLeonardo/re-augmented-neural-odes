import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Rubanova / Latent-ODE time-series work is OUT OF SCOPE for the ReScience
# replication (see extra/README.md). These tests still pass and are kept, but are
# marked `extra` so the gated suite (`pytest -m "not extra"`) excludes them.
pytestmark = pytest.mark.extra

from training.timeseries_engine import (
    compute_context_training_metrics,
    compute_timeseries_metrics,
    evaluate_timeseries,
    train_timeseries_epoch,
)
from data.timeseries import IrregularSineWaveDataset
from models.ode_rnn import GRUNoTimeBaseline, GRUTimeSeriesBaseline, ODERNN


def _build_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    dataset = IrregularSineWaveDataset(
        num_samples=batch_size,
        n_context_points=6,
        n_future_points=4,
        min_observed_context_points=2,
        observation_prob=0.6,
        noise_std=0.01,
        seed=123,
    )
    samples = [dataset[index] for index in range(batch_size)]
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0].keys()
    }


def _build_manual_metric_batch() -> dict[str, torch.Tensor]:
    """Tiny hand-crafted batch for exact metric checks."""
    return {
        "observed_context": torch.tensor([[[10.0], [0.0]]]),
        "context_values": torch.tensor([[[10.0], [20.0]]]),
        "context_times": torch.tensor([[0.0, 1.0]]),
        "context_mask": torch.tensor([[[1.0], [0.0]]]),
        "interp_mask": torch.tensor([[[0.0], [1.0]]]),
        "full_times": torch.tensor([[0.0, 1.0, 2.0]]),
        "ground_truth": torch.tensor([[[100.0], [200.0], [300.0]]]),
        "future_mask": torch.tensor([[[1.0]]]),
    }


class RecordingConstantModel(nn.Module):
    """Minimal model that records query length and returns a learned constant."""

    def __init__(self, output_dim: int = 1, constant: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(constant)))
        self.output_dim = output_dim
        self.last_query_steps: int | None = None

    def forward(
        self,
        context_times: torch.Tensor,
        observed_context: torch.Tensor,
        context_mask: torch.Tensor,
        full_times: torch.Tensor,
    ) -> torch.Tensor:
        self.last_query_steps = full_times.size(1)
        batch_size, query_steps = full_times.shape
        return self.bias.view(1, 1, 1).expand(batch_size, query_steps, self.output_dim)

    def get_nfe(self) -> int:
        return 0


def test_irregular_sine_dataset_masks_are_consistent() -> None:
    dataset = IrregularSineWaveDataset(
        num_samples=4,
        n_context_points=8,
        n_future_points=5,
        min_observed_context_points=2,
        observation_prob=0.6,
        seed=7,
    )
    sample = dataset[0]

    assert sample["context_mask"].shape == (8, 1)
    assert sample["interp_mask"].shape == (8, 1)
    assert sample["future_mask"].shape == (5, 1)

    assert sample["context_mask"].sum().item() >= 2
    assert sample["interp_mask"].sum().item() >= 1

    assert torch.all(sample["context_times"][1:] > sample["context_times"][:-1])
    assert torch.all(sample["full_times"][1:] > sample["full_times"][:-1])
    assert torch.allclose(sample["full_times"][:8], sample["context_times"])

    assert sample["future_mask"].all()
    assert sample["observed_context"].shape == (8, 1)
    assert sample["context_values"].shape == (8, 1)
    assert sample["ground_truth"].shape == (13, 1)


def test_ode_rnn_shape_consistency() -> None:
    batch = _build_batch()
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.shape == batch["ground_truth"].shape


def test_gru_baseline_shape_consistency() -> None:
    batch = _build_batch()
    model = GRUTimeSeriesBaseline(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.shape == batch["ground_truth"].shape


def test_gru_no_time_baseline_shape_consistency() -> None:
    batch = _build_batch()
    model = GRUNoTimeBaseline(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.shape == batch["ground_truth"].shape


def test_ode_rnn_nfe_tracking() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    assert model.get_nfe() == 0

    _ = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert model.get_nfe() > 0


def test_ode_rnn_gradient_flow() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )
    loss = predictions.sum()
    loss.backward()

    has_gradients = any(parameter.grad is not None for parameter in model.parameters())
    assert has_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_ode_rnn_device_agnostic() -> None:
    device = torch.device("cuda:0")
    batch = {key: value.to(device) for key, value in _build_batch(batch_size=2).items()}
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    ).to(device)

    predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["full_times"],
    )

    assert predictions.device == device


def test_context_only_training_metrics_are_well_defined() -> None:
    batch = _build_batch(batch_size=2)
    model = ODERNN(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        solver_type="rk4",
    )

    context_predictions = model(
        context_times=batch["context_times"],
        observed_context=batch["observed_context"],
        context_mask=batch["context_mask"],
        full_times=batch["context_times"],
    )

    metrics = compute_context_training_metrics(context_predictions, batch)

    assert set(metrics.keys()) == {"loss", "observed_mse", "interpolation_mse"}
    assert metrics["loss"].ndim == 0
    assert torch.allclose(metrics["loss"], metrics["observed_mse"])


def test_compute_timeseries_metrics_matches_manual_values() -> None:
    batch = _build_manual_metric_batch()
    predictions = torch.tensor([[[1.0], [2.0], [3.0]]])

    metrics = compute_timeseries_metrics(predictions, batch)

    assert torch.isclose(metrics["observed_mse"], torch.tensor(81.0))

    assert torch.isclose(metrics["interpolation_mse"], torch.tensor(39204.0))

    assert torch.isclose(metrics["extrapolation_mse"], torch.tensor(88209.0))

    expected_full = torch.tensor(
        ((1 - 100) ** 2 + (2 - 200) ** 2 + (3 - 300) ** 2) / 3.0
    )
    assert torch.isclose(metrics["loss"], expected_full)


def test_compute_context_training_metrics_matches_manual_values() -> None:
    batch = _build_manual_metric_batch()
    context_predictions = torch.tensor([[[1.0], [2.0]]])

    metrics = compute_context_training_metrics(context_predictions, batch)

    assert torch.isclose(metrics["loss"], torch.tensor(81.0))
    assert torch.isclose(metrics["observed_mse"], torch.tensor(81.0))

    assert torch.isclose(metrics["interpolation_mse"], torch.tensor(39204.0))


def test_train_timeseries_epoch_observed_context_queries_context_only() -> None:
    batch = _build_batch(batch_size=2)
    dataloader = DataLoader([batch], batch_size=None)

    model = RecordingConstantModel(output_dim=1, constant=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    metrics = train_timeseries_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_mode="observed_context",
    )

    assert model.last_query_steps == batch["context_times"].size(1)
    assert "loss" in metrics
    assert "interpolation_mse" in metrics
    assert metrics["nfe_per_sample"] == 0.0


def test_train_timeseries_epoch_full_trajectory_queries_full_times() -> None:
    batch = _build_batch(batch_size=2)
    dataloader = DataLoader([batch], batch_size=None)

    model = RecordingConstantModel(output_dim=1, constant=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    metrics = train_timeseries_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_mode="full_trajectory",
    )

    assert model.last_query_steps == batch["full_times"].size(1)
    assert "extrapolation_mse" in metrics
    assert metrics["nfe_per_sample"] == 0.0


def test_evaluate_timeseries_queries_full_times() -> None:
    batch = _build_batch(batch_size=2)
    dataloader = DataLoader([batch], batch_size=None)

    model = RecordingConstantModel(output_dim=1, constant=0.0)

    metrics = evaluate_timeseries(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert model.last_query_steps == batch["full_times"].size(1)
    assert "loss" in metrics
    assert "extrapolation_mse" in metrics
    assert metrics["nfe_per_sample"] == 0.0


def test_evaluate_timeseries_reports_both_nfe_normalizations() -> None:
    batch = _build_batch(batch_size=2)
    dataloader = DataLoader([batch], batch_size=None)

    model = RecordingConstantModel(output_dim=1, constant=0.0)

    metrics = evaluate_timeseries(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert "nfe_per_sample" in metrics
    assert "nfe_per_batch" in metrics
    assert metrics["nfe_per_sample"] == 0.0
    assert metrics["nfe_per_batch"] == 0.0


def test_multidim_sine_dataset_shapes_are_consistent() -> None:
    dataset = IrregularSineWaveDataset(
        num_samples=2,
        n_context_points=6,
        n_future_points=4,
        input_dim=3,
        signal_type="sine",
        seed=123,
    )
    sample = dataset[0]

    assert sample["observed_context"].shape == (6, 3)
    assert sample["context_values"].shape == (6, 3)
    assert sample["context_mask"].shape == (6, 1)
    assert sample["interp_mask"].shape == (6, 1)
    assert sample["ground_truth"].shape == (10, 3)
    assert sample["future_mask"].shape == (4, 1)


def test_spiral_dataset_shapes_are_consistent() -> None:
    dataset = IrregularSineWaveDataset(
        num_samples=2,
        n_context_points=6,
        n_future_points=4,
        input_dim=2,
        signal_type="spiral",
        seed=123,
    )
    sample = dataset[0]

    assert sample["observed_context"].shape == (6, 2)
    assert sample["context_values"].shape == (6, 2)
    assert sample["context_mask"].shape == (6, 1)
    assert sample["interp_mask"].shape == (6, 1)
    assert sample["ground_truth"].shape == (10, 2)
    assert sample["future_mask"].shape == (4, 1)
