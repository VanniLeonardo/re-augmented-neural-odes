# ReScience C replication — reproduction entry points.
#
#   make smoke        Fast end-to-end check (no GPU, no accounts, < 5 min)
#   make test         Gated unit tests
#   make figures      EVERY paper figure, regenerated from committed CSVs (< 2 min, no GPU)
#   make reproduce-all  Re-run the whole replication from scratch (GPU, ~15-20 h)
#   make docker-smoke Container acceptance gate
#
# Scope: Dupont et al. 2019 (ANODE) primary + Chen et al. 2018 claims C1-C4.
# Rubanova/Latent-ODE is excluded (see OUT_OF_SCOPE.md). Logging defaults to the CSV
# backend (no W&B account); set NODE_LOGGER=wandb to opt in.
#
# `figures` needs NO GPU and NO training: the committed per-seed CSVs under results/
# are the canonical artifacts and every figure is rebuilt from them. That is the
# command a reviewer wants.

PY ?= python
SMOKE_DIR := .smoke
SEEDS ?= 0,1,2,3,4

.PHONY: help smoke test figures reproduce-all reproduce-dupont reproduce-chen \
        d1 d2 d3 d4 d8 c1 c2 c3 c4 slice-grid fig-slice-grid \
        fig-d1 fig-d2 fig-d3 fig-d4 fig-d8 fig-c1 fig-c2 fig-c3 fig-c4 \
        coursework mnist-baselines table2 table3 anode-figures solver-ablation fig3 \
        factorial budget c2-guard env docker-build docker-smoke clean

help:
	@echo "Reviewer path (no GPU needed):"
	@echo "  smoke            Fast end-to-end pipeline check (<5 min)"
	@echo "  test             Gated unit tests (pytest -m 'not extra')"
	@echo "  figures          Regenerate EVERY paper figure from committed CSVs (<2 min)"
	@echo "  docker-smoke     Build the CPU image and run 'make smoke' inside it"
	@echo ""
	@echo "Full re-run from scratch (GPU; costs in parentheses are measured or estimated):"
	@echo "  reproduce-all    Everything below (~15-20 GPU-h)"
	@echo "  reproduce-dupont d1 d2 d3 d4 d8      Dupont ANODE (primary)"
	@echo "  reproduce-chen   c1 c2 c3 c4         Chen NODE claims C1-C4"
	@echo ""
	@echo "  d1  Toy separation / NFE growth vs budget   (~2 h, CPU)"
	@echo "  d2  Missing-slice generalisation, Fig 9     (~1-2 h, CPU)"
	@echo "  d3  Matched-param MNIST, Table 1           (1.9 h serial, A100 -- measured)"
	@echo "  d4  Matched-param CIFAR-10, Table 1        (4.3 h serial, A100 -- measured)"
	@echo "  d8  1-D crossing flow, Fig 3 / Prop 1       (~5 min, CPU)"
	@echo "  c1  Solver dynamics, Fig 3a-b               (~3.5 h serial, GPU)"
	@echo "  c2  bwd/fwd NFE ratio vs tolerance          (~1-2 h)"
	@echo "  c3  NFE growth over training + stiffening   (~3 h, GPU)"
	@echo "  c4  O(1) memory vs NFE                      (~0.5 h, GPU)"
	@echo ""
	@echo "  coursework       Pre-replication course experiments (NOT part of the submission)"

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
# FIGURES — every paper figure, rebuilt from the committed per-seed CSVs.
# No GPU, no training, no network. Each target also prints the pre-declared
# refutation check for its claim, so the numbers are auditable, not just drawn.
# --------------------------------------------------------------------------
figures: fig-d1 fig-d2 fig-d3 fig-d4 fig-d8 fig-c1 fig-c2 fig-c3 fig-c4
	@echo ""
	@echo "All figures regenerated under figures/ from committed results/*.csv."

fig-d1:  ## Dupont Fig 6 analog: NFE growth vs training budget (both geometries)
	$(PY) -m scripts.d1_report
	$(PY) -m scripts.plot_budget_sweep
	$(PY) -m scripts.plot_budget_sweep --results_dir results/budget_circles \
	  --figures_dir figures/budget_circles

fig-d2:  ## Dupont Fig 9: missing-slice generalisation
	$(PY) -m scripts.d2_report
	$(PY) -m scripts.plot_d2

fig-d3:  ## Dupont Table 1 (MNIST) + the recon-faithful NFE re-measurement
	$(PY) -m scripts.d3_report
	$(PY) -m scripts.d3_faithful_report

fig-d4:  ## Dupont Table 1 (CIFAR-10)
	$(PY) -m scripts.d4_report

fig-d8:  ## Dupont Fig 3 / Proposition 1: the 1-D crossing flow
	$(PY) -m scripts.d8_report

fig-c1:  ## Chen Fig 3a-b: solver dynamics (error and cost vs tolerance)
	$(PY) -m scripts.c1_report

fig-c2:  ## Chen Fig 3c recharacterised: bwd/fwd NFE as a tolerance x field surface
	$(PY) -m scripts.plot_c2_surface
	$(PY) -m scripts.plot_c2_recharacterise

fig-c3:  ## Chen Fig 3d: NFE grows during training (+ the stiffening mechanism)
	$(PY) -m scripts.plot_mnist_nfe
	$(PY) -m scripts.plot_mnist_stiffening

fig-c4:  ## Chen Table 1 memory column, corrected: O(1) memory vs NFE
	$(PY) -m scripts.plot_c4

# --------------------------------------------------------------------------
# FULL RE-RUN (GPU). Each target regenerates the CSVs its fig-* target consumes.
# Settings are the ones that produced the committed results (see OVERNIGHT_LOG.md).
# --------------------------------------------------------------------------
reproduce-all: reproduce-dupont reproduce-chen figures
	@echo "reproduce-all complete. Numbers under results/, figures under figures/."

reproduce-dupont: d8 d1 d2 d3 d4
reproduce-chen: c4 c2 c3 c1

d1: budget  ## toy separation + NFE-vs-budget, accurate tol + recon check
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_budget_sweep \
	  --seeds $(SEEDS) --budgets 25,50,100,200,500 --models NODE,ANODE-p1 \
	  --geometry circles --train_tol 1e-6 --eval_tol 1e-6 \
	  --results_dir results/budget_circles

d2:  ## Dupont Fig 9: remove the angular wedge [0, pi/5] from TRAINING only
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_missing_slice \
	  --geometry spheres --seeds $(SEEDS) --epochs 100 \
	  --train_tol 1e-6 --eval_tol 1e-6 --results_dir results/slice_spheres

slice-grid:  ## §6 extension: augmentation p x wedge width x {spheres, circles} x 10 seeds (CPU, resumable)
	$(PY) -m scripts.run_slice_grid

fig-slice-grid:  ## §6 extension: Fig 9 made systematic
	$(PY) -m scripts.slice_grid_report

# D3/D4 write one shard per seed (`*_s<seed>.csv`), the layout fig-d3/fig-d4 read. The
# ladder includes the 1e-3 TRAIN tolerance so the tolerance the accuracy is measured at is
# itself recon-checked. results/d3_faithful/ and results/d{3,4}/run1_rtx3090/ are the
# first run, kept as a replicate (provenance); they are not regenerated here.
d3:  ## matched-param NODE vs ANODE on MNIST
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.run_d3_anode_mnist --seeds $$s --tag _s$$s --epochs 8 \
	    --batch_size 256 --eval_tols 1e-3,1e-5,1e-6,1e-7 || exit 1; \
	done

d4:  ## matched-param NODE vs ANODE on CIFAR-10 (resumable per shard; see --fresh)
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.run_d4_anode_cifar --seeds $$s --tag _s$$s --epochs 10 \
	    --batch_size 256 --eval_tols 1e-3,1e-5,1e-6,1e-7 || exit 1; \
	done

d8:  ## 1-D crossing flow: NODE cannot cross, ANODE can (tolerance ladder + recon)
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.train_crossing_flow \
	  --seeds $(SEEDS) --epochs 300 --train_tol 1e-5 --eval_tols 1e-3,1e-5,1e-6,1e-7

c1:  ## Chen Fig 3a-b: adaptive x tolerance and fixed-step x step count
	$(PY) -m scripts.run_c1_solver_dynamics --seeds $(SEEDS)

c2: c2-guard  ## bwd/fwd NFE ratio across tolerance, both fields + MNIST
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.c2_recharacterise --seeds $(SEEDS)
	$(PY) -m scripts.c2_mnist_ratio

c3:  ## NFE growth over training, and the stiffening mechanism behind it
	$(PY) -m scripts.run_mnist_nfe --seeds $(SEEDS) --epochs 10
	$(PY) -m scripts.run_mnist_stiffening --seeds $(SEEDS) --epochs 8

c4:  ## O(1) memory as ODE-Net NFE rises: adjoint flat vs direct backprop rising
	$(PY) -m scripts.c4_memory_vs_nfe --seeds 0,1,2

budget:  ## D1 on the spheres geometry (the committed results/budget/)
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_budget_sweep \
	  --seeds $(SEEDS) --budgets 25,50,100,200,500,1000 --models NODE,ANODE-p1 \
	  --train_tol 1e-6 --eval_tol 1e-6

c2-guard:  ## C2 tolerance guard: the ratio is a tolerance axis, not one number
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.c2_tolerance_guard --geometry spheres --epochs 100 --seed 0
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.c2_tolerance_guard --geometry circles --epochs 100 --seed 0 \
	  --results_dir results/c2_circles

# --------------------------------------------------------------------------
# COURSEWORK — the pre-replication experiments. NOT part of the ReScience
# submission and not cited by it; kept so the conversion is auditable.
# The factorial's verdict is WITHDRAWN (loose-tolerance artifact, DEVIATIONS A1/A3).
# --------------------------------------------------------------------------
coursework: mnist-baselines table2 table3 solver-ablation

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
	  --batch_size 64 --hidden_dim 2 --augment_dims 0,1,2,5 \
	  --seeds $(SEEDS) --log_every 50 --write_results
	@$(MAKE) anode-figures

table3:
	$(PY) -m scripts.train_anode_slice_circles --epochs 500 --lr 3e-3 --n_samples 1000 \
	  --n_val_samples 3000 --batch_size 64 --hidden_dim 2 \
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

factorial:  # WITHDRAWN verdict; kept for provenance (DEVIATIONS A1/A3)
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_stem_geometry_factorial \
	  --seeds $(SEEDS) --epochs 500 --max_num_steps 1500 --time_budget_s 360 --with_mlp_head
	$(PY) -m scripts.print_factorial_table

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
