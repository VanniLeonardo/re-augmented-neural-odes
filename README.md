# Continuous-Depth Models vs. Discrete Networks: Neural ODEs

Course project for **30562 - Machine Learning and Artificial Intelligence**.

This repository contains the code used for the empirical study of Neural Ordinary Differential Equations (Neural ODEs) against discrete residual networks. The experiments cover:

1. MNIST classification with parameter-matched discrete and continuous-depth models.
2. Topological bottlenecks on concentric circles and Augmented Neural ODEs (ANODEs).
3. Solver dynamics and Number of Function Evaluations (NFE) ablations.
4. Irregular time-series modelling with ODE-RNNs, Latent ODEs, and timestamp-aware GRU baselines.

The final report should be read together with this repository. The code is organized so that the main report tables can be reproduced from the command-line scripts below.

---

## ReScience C reproduction (start here)

This repository is being converted into a **ReScience C** replication of **Dupont et al.
2019 (Augmented Neural ODEs)** (primary) and a subset of **Chen et al. 2018 (Neural ODEs)**
claims. See [`REPLICATION_PLAN.md`](REPLICATION_PLAN.md) for scope and
[`CHANGELOG_REPLICATION.md`](CHANGELOG_REPLICATION.md) for changes vs the coursework.
The Rubanova / Latent-ODE / sine / spiral experiments are **out of scope** and excluded
(see [`OUT_OF_SCOPE.md`](OUT_OF_SCOPE.md)).

**No external accounts are needed.** Experiment logging defaults to a local CSV backend
(`NODE_LOGGER=csv`); set `NODE_LOGGER=wandb` only to opt into Weights & Biases.

```bash
# 1. Fast pipeline check — no GPU, no accounts, < 5 min (verified in a clean container):
make smoke

# 2. Verify the clone-and-run acceptance gate in a fresh CPU Docker container:
make docker-smoke

# 3. Full reproduction at paper settings (GPU):
make reproduce-all            # or per-artifact: make table2 / table3 / solver-ablation / mnist-baselines

# 4. Gated unit tests:
make test
```

Pinned environment: [`environment.yml`](environment.yml) (conda/GPU),
[`requirements.txt`](requirements.txt) (pip), [`requirements-lock-cpu.txt`](requirements-lock-cpu.txt)
(fully-resolved CPU lock), and a [`Dockerfile`](Dockerfile) (CPU image used for the smoke gate).
Reference hardware for the committed numbers: **NVIDIA RTX 3090, driver 595, CUDA 12.1**.
Note: adaptive ODE solvers + cuDNN are not bit-exact across GPUs/drivers even at a fixed
seed — results reproduce within a documented tolerance, not to the last digit.

---

## Repository structure

```text
NeuralODEs/
├── .gitignore
├── config.py                     # Dataclass configs for the main experiments
├── environment.yml               # Conda environment specification
├── main.py                       # Integrated Neural ODE + Latent ODE runner
├── solver_ablation.py            # Solver/NFE ablation on synthetic 2D data
├── sweep_*.yaml                  # Weights & Biases sweep configs
├── data/
│   ├── dataloaders.py            # MNIST and synthetic 2D dataloaders
│   ├── synthetic.py              # Circles, spirals, moons, Latent ODE synthetic data
│   └── timeseries.py             # Standalone irregular time-series dataset
├── models/
│   ├── continuous.py             # ODEFunc, ODEBlock, ConvODEFunc, Latent ODE modules
│   ├── networks.py               # ODENet, ConvODENet, EulerDiscretizedODENet (weight-tied Euler baseline)
│   └── ode_rnn.py                # Standalone ODE-RNN and GRU baselines
├── scripts/
│   ├── train_discrete_mnist.py   # MNIST discrete residual baseline
│   ├── train_continuous_mnist.py # MNIST ODE-Net and Conv-ODE-Net
│   ├── train_anode_circles.py    # NODE vs ANODE on full concentric circles
│   ├── train_anode_slice_circles.py # Missing-slice generalization experiment
│   ├── make_plots.py            # Latent ODE encoder comparison plots
│   ├── plots_odernn.py           # Standalone ODE-RNN diagnostic plot
│   └── plot_*.py                 # Additional figure helpers
├── training/
│   ├── engine.py                 # Classification training/evaluation loops and NFE tracking
│   ├── timeseries_engine.py      # Standalone time-series training/evaluation loops
│   └── train_timeseries.py       # Standalone ODE-RNN/GRU time-series runner
├── tests/
│   ├── test_continuous.py        # ODE block shape, NFE, and gradient tests
│   └── test_timeseries.py        # Time-series data/model tests
└── references/                   

```

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate neural_odes
```

Logging defaults to a local **CSV backend** — no Weights & Biases account is required.
To opt into W&B instead, install it (`pip install wandb`) and set `NODE_LOGGER=wandb`
(offline by default; `WANDB_MODE=online` to sync). To silence logging entirely use
`NODE_LOGGER=none`.

MNIST is downloaded automatically by `torchvision` into `./data` the first time an MNIST script is run.

---

## Quick checks

The files `scripts/smoke_continuous_mnist.py` and `scripts/smoke_discrete_mnist.py` are manual one-forward-pass scripts (renamed from `test_*.py` so pytest does not collect them). Run them directly only when checking one MNIST forward pass:

```bash
python -m scripts.smoke_discrete_mnist
python -m scripts.smoke_continuous_mnist
```

---

## Experiment map

| Report item | Experiment | Main script |
|---|---|---|
| Table 1 | MNIST discrete ResNet vs ODE-Net vs Conv-ODE-Net | `scripts/train_discrete_mnist.py`, `scripts/train_continuous_mnist.py` |
| Appendix Table 7 | Memory scaling with residual depth | `scripts/train_discrete_mnist.py` with larger `--num_layers` |
| Tables 2-3 | NODE vs ANODE and missing-slice generalization | `scripts/train_anode_circles.py`, `scripts/train_anode_slice_circles.py` |
| Table 4 and Figure 7 | Standalone ODE-RNN vs GRU baselines | `training/train_timeseries.py`, `scripts/plots_odernn.py` |
| Table 5 and Figures 8-9 | Latent ODE encoder comparison on 1D sine | `main.py`, `scripts/make_plots.py` |
| Table 9 | Latent ODE encoder comparison on 2D spiral | `main.py` |
| Figure 10 | Solver/NFE dynamics | `solver_ablation.py` |
| Figure 11 / tolerance diagnostic | MNIST tolerance ablation | `scripts/train_continuous_mnist.py`, `scripts/plot_fig3.py` |

---

## Reproducing the MNIST baselines

### Weight-tied Euler baseline (`EulerDiscretizedODENet`)

The "discrete baseline" is a fixed-step **Euler discretisation of the same ODE**, with a single
`ODEFunc` weight-tied across steps for exact parameter parity with the ODE-Net. It is **not** an
independent ResNet — see [`DEVIATIONS.md`](DEVIATIONS.md). Naming it honestly matters: "the
discrete model and the ODE-Net reach comparable accuracy" is nearly tautological when they share
one vector field integrated two ways.

```bash
python -m scripts.train_discrete_mnist \
  --batch_size 64 \
  --hidden_dim 160 \
  --num_layers 5 \
  --lr 5e-4 \
  --epochs 10
```

### MLP ODE-Net

```bash
python -m scripts.train_continuous_mnist \
  --network_type mlp \
  --batch_size 64 \
  --hidden_dim 160 \
  --lr 1e-3 \
  --epochs 10 \
  --solver dopri5
```

### Convolutional ODE-Net

```bash
python -m scripts.train_continuous_mnist \
  --network_type cnn \
  --batch_size 64 \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 10 \
  --solver dopri5
```

The Conv-ODE-Net run also executes the post-training tolerance diagnostic and saves the corresponding plot under `plots/`.

### Memory-scaling diagnostic

Use the same discrete MNIST script with larger residual depths:

```bash
python -m scripts.train_discrete_mnist --batch_size 64 --hidden_dim 160 --num_layers 200 --lr 5e-4 --epochs 1
python -m scripts.train_discrete_mnist --batch_size 64 --hidden_dim 160 --num_layers 1000 --lr 5e-4 --epochs 1
```

The relevant quantity is the peak memory logged by the training loop. The deep runs are diagnostic memory measurements, not parameter-matched accuracy baselines.

---

## Reproducing NODE vs ANODE on concentric circles

The corrected ANODE experiments keep the base ODE state dimension fixed at 2 and use a separate vector-field width of 64. This preserves the intended two-dimensional topological bottleneck while giving the vector field enough capacity.

### Full concentric-circles sweep

```bash
python -m scripts.train_anode_circles \
  --epochs 500 \
  --lr 3e-3 \
  --n_samples 1000 \
  --batch_size 64 \
  --hidden_dim 2 \
  --ode_hidden_dim 32 \
  --augment_dims 0,1,2,5 \
  --seeds 0,1,2 \
  --log_every 20 \
  --write_results
```

Output:

```text
results/anode/circles_summary.csv
```

### Missing-slice generalization

```bash
python -m scripts.train_anode_slice_circles \
  --epochs 500 \
  --lr 3e-3 \
  --n_samples 1000 \
  --n_val_samples 3000 \
  --batch_size 64 \
  --hidden_dim 2 \
  --ode_hidden_dim 32 \
  --augment_dims 0,2 \
  --seeds 0,1,2 \
  --log_every 20 \
  --write_results
```

Output:

```text
results/anode/slice_circles_summary.csv
```

### ANODE figures

The dataset visualization is generated with:

```bash
python -m scripts.plot_anode_slice_dataset
```

The final bar plots are generated with:

```bash
python -m scripts.plot_anode_corrected_results
```

`plot_anode_corrected_results.py` expects the final aggregated CSV files under:

```text
results/anode/final/
```

If the raw per-seed CSVs are regenerated, first aggregate them by `model_name` over seeds, then place the flat summary tables in that directory.

---

## Reproducing the solver/NFE ablation

Run the solver sweep on concentric circles:

```bash
python solver_ablation.py \
  --epochs 50 \
  --n_samples 1000 \
  --batch_size 64 \
  --hidden_dim 32 \
  --lr 1e-3
```

This runs fixed-step solvers (`euler`, `midpoint`, `rk4`) and adaptive solvers (`bosh3`, `dopri5`, `dopri8`) across tolerances. NFE is logged during training through the `ODEFunc.nfe` counter and summarized by the training engine.

The MNIST tolerance diagnostic is run automatically at the end of `scripts/train_continuous_mnist.py`; the plotting helper is `scripts/plot_fig3.py`.

---

## Reproducing the standalone irregular time-series diagnostic

These commands reproduce the standalone ODE-RNN vs GRU comparison on the 1D sine task. They write JSON files under `results/`.

```bash
python -m training.train_timeseries \
  --run_name final_odernn_seed42_e10 \
  --model_type ode_rnn \
  --epochs 10 \
  --seed 42 \
  --wandb_mode offline

python -m training.train_timeseries \
  --run_name final_gru_time_seed42_e10 \
  --model_type gru_time \
  --epochs 10 \
  --seed 42 \
  --wandb_mode offline

python -m training.train_timeseries \
  --run_name final_gru_notime_seed42_e10 \
  --model_type gru_notime \
  --epochs 10 \
  --seed 42 \
  --wandb_mode offline
```

Generate the standalone interpolation learning-curve figure:

```bash
python -m scripts.plots_odernn
```

---

## Reproducing the Latent ODE encoder comparison

The Latent ODE comparison uses the integrated runner in `main.py`. To run only the Latent ODE part, pass `--ode-epochs 0`; this bypasses the synthetic 2D Neural ODE phase.

### 1D sine, three encoders, two seeds

```bash
for seed in 42 123; do
  for encoder in odernn gru_time gru_notime; do
    python main.py \
      --run-name 3way_${encoder}_seed${seed} \
      --ode-epochs 0 \
      --latent-encoder-type ${encoder} \
      --latent-seed ${seed} \
      --latent-input-dim 1 \
      --latent-signal-type sine \
      --latent-epochs 30 \
      --latent-num-train 500 \
      --latent-num-val 100 \
      --latent-num-test 100 \
      --latent-batch-size 64 \
      --latent-lr 1e-2 \
      --latent-kl-coeff 0.01 \
      --latent-method rk4 \
      --latent-encoder-method euler
  done
done
```

Generate the three-way encoder comparison plots:

```bash
python -m scripts.make_plots
```

Outputs are saved under:

```text
results/3way_*.json
figures/
```

### 2D spiral budget comparison

```bash
for epochs in 30 60; do
  for encoder in odernn gru_time; do
    python main.py \
      --run-name spiral_${encoder}_e${epochs}_seed42 \
      --ode-epochs 0 \
      --latent-encoder-type ${encoder} \
      --latent-seed 42 \
      --latent-input-dim 2 \
      --latent-signal-type spiral \
      --latent-epochs ${epochs} \
      --latent-num-train 500 \
      --latent-num-val 100 \
      --latent-num-test 100 \
      --latent-batch-size 64 \
      --latent-lr 1e-2 \
      --latent-kl-coeff 0.01 \
      --latent-method rk4 \
      --latent-encoder-method euler
  done
done
```

---

## Outputs

The scripts create the following directories as needed:

```text
results/       # JSON/CSV summaries
figures/       # Final figure files
plots/         # Tolerance and ODE-flow diagnostic plots
wandb/         # Offline W&B logs
```

Randomness is controlled through explicit seeds where the experiments average over seeds.

---

## Implementation notes

- `ODENet` and `ConvODENet` use `torchdiffeq.odeint_adjoint`, so the backward pass is computed through the adjoint method.
- `EulerDiscretizedODENet` (formerly `DiscreteResNet`) is a weight-tied Euler discretisation of the ODE-Net's vector field. It reuses one `ODEFunc` across steps for exact parameter parity; it is a controlled baseline, not an independent discrete architecture (see `DEVIATIONS.md`).
- NFE is counted inside the vector-field modules through an integer counter `nfe`. The training loop snapshots the counter before and after `loss.backward()` to separate forward and backward NFE.
- In the ANODE experiments, `hidden_dim=2` is the actual ODE state dimension before augmentation, while `ode_hidden_dim=32` is only the internal width of the vector-field MLP (Dupont toy field).
- The SLURM files under `scripts/` document the cluster runs used during development. They contain cluster-specific accounts and paths, so the local commands above are the portable way to reproduce the experiments.
