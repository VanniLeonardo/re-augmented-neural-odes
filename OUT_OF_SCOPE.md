# Material excluded from the replication

The submission replicates Dupont et al. 2019 (Augmented Neural ODEs) and four solver and
memory claims of Chen et al. 2018. The Latent ODE and ODE-RNN work of Rubanova et al. 2019,
together with the synthetic sine and spiral experiments from the original course project, is
excluded for three reasons.

1. Rubanova's headline claims are on PhysioNet, MuJoCo Hopper and Human Activity. The code
   does not run any of these, so a replication of that paper is not possible at our scope.
2. The two-dimensional spiral is an experiment from Chen et al. 2018, not from Rubanova et
   al. Presenting it under a Rubanova heading would be a citation error.
3. The synthetic one-dimensional sine can only be compared against Rubanova's toy
   supplementary Table 2, and with substantial setup drift. That is architecture validation
   rather than a controlled reproduction.

The full rationale is in `REPLICATION_PLAN.md`, sections 1.1 and 3.C.

## What this means

- These experiments are not part of `make reproduce-all` and not part of the test suite.
  `make test` and `make smoke` run `pytest -m "not extra"`, which excludes them.
- The code stays in the repository for provenance rather than being deleted. It is not
  maintained to the reproducibility standard of the rest: its entry points still use Weights
  and Biases directly and are not covered by the pinned smoke pipeline.
- Rubanova et al. 2019 is cited in the paper as related work only.

## Excluded files, left in place

| Path | Role |
|---|---|
| `main.py` | Integrated runner: synthetic Neural ODE and Latent ODE on sine and spiral. |
| `training/train_timeseries.py` | Standalone ODE-RNN and GRU time-series runner. |
| `training/timeseries_engine.py` | Time-series training and evaluation loops. |
| `data/timeseries.py` | `IrregularSineWaveDataset`. |
| `models/ode_rnn.py` | Standalone ODE-RNN and GRU baselines. |
| Latent classes in `models/continuous.py` | `LatentODEFunc`, `EncoderODEFunc`, `ODERNNEncoder`, `StandardGRUEncoder`, `VanillaGRUEncoder`, `LatentODE`. |
| `TimeSeriesDataset` in `data/synthetic.py` | Synthetic sine, spiral and damped signals. |
| `scripts/make_plots.py`, `scripts/plots_odernn.py` | Figures for this material. |
| `tests/test_timeseries.py` | Unit tests, marked `extra` and excluded from the gated suite. |
| `results/3way_*.json`, `results/C*_spiral.json`, `results/final_*_e10.json` | Committed result files. |

Running this material needs the optional `wandb` dependency, or `NODE_LOGGER=none`. The
pre-replication coursework experiments that are still wired up are available through
`make coursework`.
