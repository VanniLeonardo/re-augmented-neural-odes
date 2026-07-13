# OUT_OF_SCOPE — coursework material excluded from the ReScience C replication

The ReScience C submission replicates **Dupont et al. 2019 (Augmented Neural ODEs)**
(primary) and a subset of **Chen et al. 2018 (Neural ODEs)** claims (C1–C4). The
**Rubanova et al. 2019 (Latent ODEs) / ODE-RNN / synthetic sine & spiral** experiments
from the original course project are **excluded**, because:

1. Rubanova's headline claims are on **PhysioNet / MuJoCo Hopper / Human Activity** —
   none of which the code runs — so no replication of the paper is possible at our scope.
2. The 2-D **spiral is a Chen-2018 experiment, not a Rubanova one**; presenting it under
   a Rubanova heading would be a citation error.
3. The synthetic 1-D sine can only be compared to Rubanova's *toy* Suppl. Table 2, and with
   heavy setup drift — it is architecture validation, not a controlled reproduction.

See `REPLICATION_PLAN.md` §1.1 / §3.C and decision record D3 for the full rationale.

## What this means concretely

- These experiments are **not** part of `make reproduce-all` and are **not** in the gated
  test suite (`make test` / `make smoke`, which run `pytest -m "not extra"`).
- The code is **kept in the repository** (not deleted) for coursework provenance, but it is
  not maintained to the replication's reproducibility standard (its entry points still use
  W&B directly and are not part of the pinned smoke pipeline).
- Rubanova 2019 is cited only as **related work** in the paper.

## Excluded files (left in place)

| Path | Role |
|---|---|
| `main.py` | Integrated runner: Phase-1 synthetic Neural ODE + Phase-4 Latent ODE (sine/spiral). |
| `training/train_timeseries.py` | Standalone ODE-RNN / GRU time-series runner (report Table 4). |
| `training/timeseries_engine.py` | Time-series train/eval loops + metrics. |
| `data/timeseries.py` | `IrregularSineWaveDataset` (standalone time-series data). |
| `models/ode_rnn.py` | Standalone ODE-RNN + GRU baselines. |
| `models/continuous.py` (Latent classes) | `LatentODEFunc`, `EncoderODEFunc`, `ODERNNEncoder`, `StandardGRUEncoder`, `VanillaGRUEncoder`, `LatentODE`. |
| `data/synthetic.py` (`TimeSeriesDataset`) | Latent-ODE synthetic sine/spiral/damped signals. |
| `scripts/make_plots.py`, `scripts/plots_odernn.py` | Latent-ODE / ODE-RNN figures. |
| `tests/test_timeseries.py` | Time-series unit tests (marked `extra`; excluded from the gated suite). |
| `results/3way_*.json`, `results/C*_spiral.json`, `results/final_*_e10.json` | Committed time-series result JSONs. |

To run this material anyway you need the optional `wandb` dependency (or `NODE_LOGGER=none`)
and the coursework instructions in the top-level `README.md`.
