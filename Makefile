# Reproduction entry points for the ReScience C replication of Augmented Neural ODEs.
#
#   make test      unit tests
#   make smoke     end-to-end check on tiny settings, CPU only, no network
#   make figures   every figure, rebuilt from the committed per-seed results
#   make reproduce-all   re-run every experiment from scratch (GPU, 15-20 h)
#
# Logging defaults to a CSV backend and needs no account.

PY ?= python
SMOKE_DIR := .smoke
SEEDS ?= 0,1,2,3,4

.PHONY: help test smoke figures reproduce-all reproduce-dupont reproduce-chen \
        d1 d2 d3 d4 d8 c1 c2 c3 c4 slice-grid budget c2-guard \
        fig-d1 fig-d2 fig-d3 fig-d4 fig-d8 fig-c1 fig-c2 fig-c3 fig-c4 \
        fig-slice-grid env docker-build docker-smoke clean

help:
	@echo "No GPU needed:"
	@echo "  test             unit tests"
	@echo "  smoke            end-to-end check on tiny settings (~3 min, CPU, no network)"
	@echo "  figures          rebuild every figure from the committed results (~2 min)"
	@echo "  docker-smoke     build the pinned CPU image and run the smoke check inside it"
	@echo ""
	@echo "Full re-run (GPU). Costs are measured where marked:"
	@echo "  reproduce-all    everything below (~15-20 GPU-h, plus 1.6 h CPU)"
	@echo "  reproduce-dupont d8 d1 d2 d3 d4"
	@echo "  reproduce-chen   c4 c2 c3 c1"
	@echo ""
	@echo "  d8          1-D crossing flow                      (~5 min, CPU)"
	@echo "  d1          toy separation, NFE against budget     (~2 h, CPU)"
	@echo "  d2          missing-slice generalisation           (~1-2 h, CPU)"
	@echo "  d3          matched-parameter MNIST                (1.9 h, measured)"
	@echo "  d4          matched-parameter CIFAR-10             (4.3 h, measured)"
	@echo "  c1          solver dynamics                        (~3.5 h)"
	@echo "  c2          backward against forward NFE           (~1-2 h)"
	@echo "  c3          NFE growth and stiffening              (~3 h)"
	@echo "  c4          constant memory against NFE            (~0.5 h)"
	@echo "  slice-grid  the missing-region extension           (1.6 h, CPU, measured)"

# --------------------------------------------------------------------------
# Checks. No GPU, and the smoke check needs no network: it uses the generated
# two-dimensional data and the committed results, not MNIST or CIFAR-10.
# --------------------------------------------------------------------------
test:
	$(PY) -m pytest -q

smoke:
	@echo ">>> [1/4] crossing flow (1 seed, tiny)"
	@CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.train_crossing_flow --seeds 0 --epochs 20 \
	  --train_tol 1e-5 --eval_tols 1e-3,1e-5 --results_dir $(SMOKE_DIR)/crossing
	@echo ">>> [2/4] missing slice (1 seed, tiny)"
	@CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_missing_slice --geometry spheres --seeds 0 \
	  --epochs 5 --train_tol 1e-6 --eval_tol 1e-6 --results_dir $(SMOKE_DIR)/slice
	@echo ">>> [3/4] rebuild every figure from the committed results"
	@$(MAKE) figures
	@echo ">>> [4/4] unit tests"
	@$(PY) -m pytest -q
	@echo ""
	@echo "SMOKE OK"

# --------------------------------------------------------------------------
# Figures. Rebuilt from the committed per-seed results, without a GPU. Each
# target also prints the condition recorded before the run it reports.
# --------------------------------------------------------------------------
figures: fig-d1 fig-d2 fig-d3 fig-d4 fig-d8 fig-c1 fig-c2 fig-c3 fig-c4 fig-slice-grid
	@echo ""
	@echo "All figures rebuilt under figures/ from the committed results."

fig-d1:  ## toy separation, NFE against training budget, both geometries
	$(PY) -m scripts.d1_report
	$(PY) -m scripts.plot_budget_sweep
	$(PY) -m scripts.plot_budget_sweep --results_dir results/budget_circles \
	  --figures_dir figures/budget_circles

fig-d2:  ## missing-slice generalisation
	$(PY) -m scripts.d2_report
	$(PY) -m scripts.plot_d2

fig-d3:  ## matched-parameter MNIST
	$(PY) -m scripts.d3_report
	$(PY) -m scripts.d3_faithful_report

fig-d4:  ## matched-parameter CIFAR-10
	$(PY) -m scripts.d4_report

fig-d8:  ## the 1-D crossing flow
	$(PY) -m scripts.d8_report

fig-c1:  ## solver dynamics
	$(PY) -m scripts.c1_report

fig-c2:  ## backward against forward NFE, and the diagnosis
	$(PY) -m scripts.plot_c2_surface
	$(PY) -m scripts.plot_c2_recharacterise
	$(PY) -m scripts.c2_diagnosis_report

fig-c3:  ## NFE growth during training, and the stiffening behind it
	$(PY) -m scripts.plot_mnist_nfe
	$(PY) -m scripts.plot_mnist_stiffening

fig-c4:  ## constant memory against NFE
	$(PY) -m scripts.plot_c4

fig-slice-grid:  ## the missing-region extension
	$(PY) -m scripts.slice_grid_report

# --------------------------------------------------------------------------
# Full re-run. Settings are the ones that produced the committed results.
# --------------------------------------------------------------------------
reproduce-all: reproduce-dupont reproduce-chen slice-grid figures
	@echo "Done. Results under results/, figures under figures/."

reproduce-dupont: d8 d1 d2 d3 d4
reproduce-chen: c4 c2 c3 c1

d8:  ## 1-D crossing flow, tolerance ladder and reconstruction check
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.train_crossing_flow \
	  --seeds $(SEEDS) --epochs 300 --train_tol 1e-5 --eval_tols 1e-3,1e-5,1e-6,1e-7

d1: budget  ## toy separation and NFE against budget, on both geometries
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_budget_sweep \
	  --seeds $(SEEDS) --budgets 25,50,100,200,500 --models NODE,ANODE-p1 \
	  --geometry circles --train_tol 1e-6 --eval_tol 1e-6 \
	  --results_dir results/budget_circles

budget:  ## the spheres half of d1
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_budget_sweep \
	  --seeds $(SEEDS) --budgets 25,50,100,200,500,1000 --models NODE,ANODE-p1 \
	  --train_tol 1e-6 --eval_tol 1e-6

d2:  ## remove the wedge [0, pi/5] from training only
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.run_missing_slice \
	  --geometry spheres --seeds $(SEEDS) --epochs 100 \
	  --train_tol 1e-6 --eval_tol 1e-6 --results_dir results/slice_spheres

# d3 and d4 write one file per seed, the layout the report scripts read. The
# tolerance ladder includes the training tolerance, so the tolerance the accuracy
# is measured at is itself reconstruction-checked.
d3:  ## matched-parameter NODE against ANODE on MNIST
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.run_d3_anode_mnist --seeds $$s --tag _s$$s --epochs 8 \
	    --batch_size 256 --eval_tols 1e-3,1e-5,1e-6,1e-7 || exit 1; \
	done

d4:  ## matched-parameter NODE against ANODE on CIFAR-10, resumable per file
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  $(PY) -m scripts.run_d4_anode_cifar --seeds $$s --tag _s$$s --epochs 10 \
	    --batch_size 256 --eval_tols 1e-3,1e-5,1e-6,1e-7 || exit 1; \
	done

c1:  ## adaptive solvers against tolerance, fixed-step against step count
	$(PY) -m scripts.run_c1_solver_dynamics --seeds $(SEEDS)

c2: c2-guard  ## backward against forward NFE, and why it does not match
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.c2_recharacterise --seeds $(SEEDS)
	$(PY) -m scripts.c2_mnist_ratio
	$(PY) -m scripts.run_c2_diagnosis --field spheres --seeds $(SEEDS) --untrained
	$(PY) -m scripts.run_c2_diagnosis --field mnist --seeds 0,1,2 \
	  --solvers dopri5,bosh3,scipy:LSODA --tols 1e-3,1e-5 --batch 8 --ref_tol 1e-7

c2-guard:  ## the ratio across a tolerance axis, on both geometries
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.c2_tolerance_guard --geometry spheres --epochs 100 --seed 0
	CUDA_VISIBLE_DEVICES="" $(PY) -m scripts.c2_tolerance_guard --geometry circles --epochs 100 --seed 0 \
	  --results_dir results/c2_circles

c3:  ## NFE growth over training, and the stiffening behind it
	$(PY) -m scripts.run_mnist_nfe --seeds $(SEEDS) --epochs 10
	$(PY) -m scripts.run_mnist_stiffening --seeds $(SEEDS) --epochs 8

c4:  ## constant memory as the model's own NFE rises
	$(PY) -m scripts.c4_memory_vs_nfe --seeds 0,1,2

slice-grid:  ## augmentation by wedge width by geometry, ten seeds
	$(PY) -m scripts.run_slice_grid

# --------------------------------------------------------------------------
# Environment.
# --------------------------------------------------------------------------
env:
	conda env create -f environment.yml

docker-build:
	docker build -t neural-odes-repro .

docker-smoke: docker-build
	docker run --rm neural-odes-repro

clean:
	rm -rf $(SMOKE_DIR) results/logs figures \
	  __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
