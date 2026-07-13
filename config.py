from __future__ import annotations
from typing import Tuple
import argparse
from dataclasses import asdict, dataclass
import torch


@dataclass
class SolverAblationConfig:
    """Configuration for the solver ablation study."""

    # Sweep parameters
    fixed_solvers: Tuple[str, ...] = ("euler", "midpoint", "rk4")
    adaptive_solvers: Tuple[str, ...] = ("bosh3", "dopri5", "dopri8")
    tolerances: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)  # atol == rtol

    dataset: str = "circles"
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05
    hidden_dim: int = 32
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50

    seed: int = 0
    results_dir: str = "results/solver_ablation"
    # Cap the number of (solver, tol) configurations run; 0 = all. Used by `make smoke`.
    max_configs: int = 0


@dataclass
class ANODECirclesConfig:
    """Configuration for NODE vs ANODE experiments on concentric circles."""

    dataset: str = "circles"
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05

    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3

    hidden_dim: int = 2
    ode_hidden_dim: int = 64
    augment_dims: Tuple[int, ...] = (0, 1, 2, 5)

    seeds: Tuple[int, ...] = (0, 1, 2)
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 200

    project: str = "neural-odes-30562"
    results_dir: str = "results/anode"
    log_every: int = 10
    write_results: bool = True


@dataclass
class ODEConfig:
    """Central configuration for Neural ODE experiments."""

    # Data Parameters
    dataset: str = "circles"  # one of: circles, spirals, moons
    n_samples: int = 1000
    val_split: float = 0.2
    noise: float = 0.05

    # ODE Solver Parameters
    solver_type: str = "dopri5"
    atol: float = 1e-3
    rtol: float = 1e-3
    integration_time: Tuple[float, float] = (0.0, 1.0)

    # Network Architecture
    in_features: int = 2  # fixed at 2 for all synthetic 2D datasets
    hidden_dim: int = 32

    augment_dim: int = 0  # Number of zero-dimensions to append
    is_time_series: bool = False  # Flag for ODE-RNN logic

    # Training Parameters
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50

    # Time-series model options
    ode_hidden_dim: int | None = None
    gru_time_hidden_dim: int | None = None
    gru_notime_hidden_dim: int | None = None
    input_dim: int = 1
    output_dim: int = 1
    signal_type: str = "sine"
    model_type: str = "ode_rnn"

    # Time-series training options
    seed: int = 42
    train_loss_mode: str = "observed_context"

    # Time-series benchmark parameters
    context_start: float = 0.0
    context_end: float = 5.0
    future_end: float = 10.0
    n_context_points: int = 50
    n_future_points: int = 50
    min_observed_context_points: int = 2
    observation_prob: float = 0.8
    noise_std: float = 0.01

    # Time-series split sizes
    train_size: int = 500
    val_size: int = 100
    test_size: int = 100


def _str2bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class LatentODEConfig:
    input_dim: int = 1
    signal_type: str = "sine"
    extrap_horizon: float = 10.0
    latent_dim: int = 20
    nhidden: int = 40

    use_ode_rnn: bool = True
    encoder_type: str = "odernn"
    is_variational: bool = True

    method: str = "rk4"
    encoder_method: str = "euler"
    rtol: float = 1e-3
    atol: float = 1e-4

    kl_coeff: float = 0.01
    lr: float = 1e-2
    batch_size: int = 64
    grad_clip_norm: float = 1.0

    num_train: int = 500
    num_val: int = 100
    num_test: int = 100
    epochs: int = 30
    eval_every: int = 5

    seq_len: int = 100
    noise_std: float = 0.01
    train_horizon: float = 5.0

    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    project_name: str = "latent-ode-phase4-faithful"

    missing_pct: float = 0.2
    burst_missing: bool = False
    burst_length_range: tuple = (2, 8)

    @property
    def num_samples(self) -> int:
        return self.num_train + self.num_val + self.num_test

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_args(cls) -> "LatentODEConfig":
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-dim", type=int, default=cls.input_dim)
        parser.add_argument("--latent-dim", type=int, default=cls.latent_dim)
        parser.add_argument("--nhidden", type=int, default=cls.nhidden)
        parser.add_argument("--use-ode-rnn", type=_str2bool, default=cls.use_ode_rnn)
        parser.add_argument(
            "--is-variational", type=_str2bool, default=cls.is_variational
        )
        parser.add_argument("--method", type=str, default=cls.method)
        parser.add_argument("--encoder-method", type=str, default=cls.encoder_method)
        parser.add_argument("--rtol", type=float, default=cls.rtol)
        parser.add_argument("--atol", type=float, default=cls.atol)
        parser.add_argument("--kl-coeff", type=float, default=cls.kl_coeff)
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--grad-clip-norm", type=float, default=cls.grad_clip_norm)
        parser.add_argument("--eval-every", type=int, default=cls.eval_every)
        parser.add_argument("--num-train", type=int, default=cls.num_train)
        parser.add_argument("--num-val", type=int, default=cls.num_val)
        parser.add_argument("--num-test", type=int, default=cls.num_test)
        parser.add_argument("--seq-len", type=int, default=cls.seq_len)
        parser.add_argument("--missing-pct", type=float, default=cls.missing_pct)
        parser.add_argument("--noise-std", type=float, default=cls.noise_std)
        parser.add_argument("--train-horizon", type=float, default=cls.train_horizon)
        parser.add_argument("--extrap-horizon", type=float, default=cls.extrap_horizon)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--num-workers", type=int, default=cls.num_workers)
        parser.add_argument("--device", type=str, default=cls.device)
        parser.add_argument("--project-name", type=str, default=cls.project_name)
        args = parser.parse_args()
        return cls(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            nhidden=args.nhidden,
            use_ode_rnn=args.use_ode_rnn,
            is_variational=args.is_variational,
            method=args.method,
            encoder_method=args.encoder_method,
            rtol=args.rtol,
            atol=args.atol,
            kl_coeff=args.kl_coeff,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            grad_clip_norm=args.grad_clip_norm,
            eval_every=args.eval_every,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            seq_len=args.seq_len,
            missing_pct=args.missing_pct,
            noise_std=args.noise_std,
            train_horizon=args.train_horizon,
            extrap_horizon=args.extrap_horizon,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
            project_name=args.project_name,
        )
