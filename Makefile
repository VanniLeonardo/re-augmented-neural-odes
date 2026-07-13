# ReScience C replication — reproduction entry points.
#
# Fast path (no GPU, no accounts, < 5 min):   make smoke
# Full reproduction (GPU, paper settings):    make reproduce-all
# Container acceptance gate:                   make docker-smoke
#
# Logging defaults to the CSV backend (no W&B account). Set NODE_LOGGER=wandb to
# opt into Weights & Biases. Scope: Dupont ANODE (primary) + Chen C1-C4. Rubanova
# is excluded (see extra/README.md). CIFAR-10 (D4) is a gated stretch, not here.

PY ?= python
SMOKE_DIR := .smoke
SEEDS ?= 0,1,2,3,4

.PHONY: help smoke test reproduce-all mnist-baselines table2 table3 anode-figures \
        solver-ablation fig3 env docker-build docker-smoke clean

help:
	@echo "Targets:"
	@echo "  smoke           Fast end-to-end pipeline check (<5 min, no GPU, no W&B)"
	@echo "  test            Gated unit tests (pytest -m 'not extra')"
	@echo "  reproduce-all   Full reproduction at paper settings (GPU)"
	@echo "  mnist-baselines Our seeded MNIST baselines (NOT Chen Table 1)"
	@echo "  table2          ANODE concentric-circles sweep (Dupont) + figures"
	@echo "  table3          ANODE missing-slice generalization (Dupont Fig 9) + figures"
	@echo "  solver-ablation Chen Fig 3 solver/NFE dynamics"
	@echo "  fig3            Chen Fig 3 tolerance diagnostic (conv ODE-Net)"
	@echo "  anode-figures   Aggregate ANODE CSVs -> figures (no manual step)"
	@echo "  docker-smoke    Build the CPU image and run 'make smoke' inside it"
	@echo "  clean           Remove scratch/generated outputs"
	@echo ""
	@echo "  (Stage C will add: C2 fwd/bwd-NFE sweep, C4 memory-vs-NFE, D3 ANODE-MNIST,"
	@echo "   D8 crossing-flow demo, and the missing-slice grid.)"

# --------------------------------------------------------------------------
# Fast verification — no GPU required, bounded by NODE_MAX_BATCHES.
# --------------------------------------------------------------------------
smoke:
	@echo ">>> [1/6] ANODE circles (tiny)"
	@NODE_MAX_BATCHES=3 $(PY) -m scripts.train_anode_circles --epochs 2 \
	  --augment_dims 0,2 --seeds 0 --n_samples 200 --log_every 1 \
	  --write_results --results_dir $(SMOKE_DIR)/anode
	@echo ">>> [2/6] ANODE missing-slice (tiny)"
	@NODE_MAX_BATCHES=3 $(PY) -m scripts.train_anode_slice_circles --epochs 2 \
	  --augment_dims 0,2 --seeds 0 --n_samples 200 --n_val_samples 300 \
	  --log_every 1 --write_results --results_dir $(SMOKE_DIR)/anode
	@echo ">>> [3/6] Aggregate + plot ANODE figures"
	@$(PY) -m scripts.aggregate_anode_results --results_dir $(SMOKE_DIR)/anode
	@$(PY) -m scripts.plot_anode_corrected_results --raw_dir $(SMOKE_DIR)/anode \
	  --results_dir $(SMOKE_DIR)/anode/final --figures_dir $(SMOKE_DIR)/figures
	@echo ">>> [4/6] Solver ablation (3 configs, tiny)"
	@$(PY) -m solver_ablation --epochs 2 --n_samples 200 --max_configs 3 \
	  --results_dir $(SMOKE_DIR)/solver
	@echo ">>> [5/6] MNIST baselines (mlp + discrete, capped batches)"
	@NODE_MAX_BATCHES=3 $(PY) -m scripts.train_continuous_mnist --network_type mlp \
	  --epochs 1 --hidden_dim 32 --seed 0 --results_dir $(SMOKE_DIR)/mnist
	@NODE_MAX_BATCHES=3 $(PY) -m scripts.train_discrete_mnist \
	  --epochs 1 --num_layers 5 --hidden_dim 32 --seed 0 --results_dir $(SMOKE_DIR)/mnist
	@echo ">>> [6/6] Unit tests (gated suite)"
	@$(PY) -m pytest -q -m "not extra"
	@echo ""
	@echo "SMOKE OK — full pipeline runs with the CSV logger and no W&B account."

test:
	$(PY) -m pytest -q -m "not extra"

# --------------------------------------------------------------------------
# Full reproduction (GPU; paper settings). Rubanova/CIFAR/SVHN excluded.
# --------------------------------------------------------------------------
reproduce-all: mnist-baselines table2 table3 solver-ablation
	@echo "reproduce-all complete. Figures under figures/ ; numbers under results/."

mnist-baselines:
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.train_discrete_mnist --num_layers 5 --hidden_dim 160 --lr 5e-4 --epochs 10 --seed $$s; \
	done
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.train_continuous_mnist --network_type mlp --hidden_dim 160 --lr 1e-3 --epochs 10 --seed $$s; \
	done
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.train_continuous_mnist --network_type cnn --hidden_dim 256 --lr 1e-3 --epochs 10 --seed $$s; \
	done

table2:
	$(PY) -m scripts.train_anode_circles --epochs 500 --lr 3e-3 --n_samples 1000 \
	  --batch_size 64 --hidden_dim 2 --ode_hidden_dim 64 --augment_dims 0,1,2,5 \
	  --seeds $(SEEDS) --log_every 50 --write_results
	@$(MAKE) anode-figures

table3:
	$(PY) -m scripts.train_anode_slice_circles --epochs 500 --lr 3e-3 --n_samples 1000 \
	  --n_val_samples 3000 --batch_size 64 --hidden_dim 2 --ode_hidden_dim 64 \
	  --augment_dims 0,2 --seeds $(SEEDS) --log_every 50 --write_results
	@$(MAKE) anode-figures

anode-figures:
	$(PY) -m scripts.aggregate_anode_results
	$(PY) -m scripts.plot_anode_corrected_results

solver-ablation:
	$(PY) -m solver_ablation --epochs 50 --n_samples 1000 --batch_size 64 --hidden_dim 32

fig3:
	$(PY) -m scripts.train_continuous_mnist --network_type cnn --epochs 10 \
	  --hidden_dim 256 --lr 1e-3 --seed 0 --tol-diagnostic

# --------------------------------------------------------------------------
# Environment / container.
# --------------------------------------------------------------------------
env:
	conda env create -f environment.yml

docker-build:
	docker build -t neural-odes-repro .

docker-smoke: docker-build
	docker run --rm neural-odes-repro

clean:
	rm -rf $(SMOKE_DIR) results/smoke results/logs figures plots \
	  __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
