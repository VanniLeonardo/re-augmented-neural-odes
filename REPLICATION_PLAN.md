# Replication plan

> **This is the plan as approved on 2026-07-13, before the experiments were run.** It is kept
> as a record of the scope decision and the compute budget. It is not a statement of results.
> For what was actually found, see [`README.md`](README.md), the article in `paper/`, and
> [`DEVIATIONS.md`](DEVIATIONS.md). Where this document predicted an outcome, the prediction
> was sometimes wrong, and the later files are the authority.


**Status:** Phase 1 (implementation). Plan approved 2026-07-13 with the shrunken scope recorded below.
**Author of plan:** engineering (code / experiments / reproducibility). The paper will be written separately.
**Scope of this document:** what to reproduce, how faithfully we currently target it, what is broken for reproducibility, a prioritised work plan, a compute budget, and what the current report must retract or weaken.

> ### APPROVED PHASE-1 SCOPE (overrides the exploratory scope discussion below)
> - **D1 — ANODE MNIST (D3) is CORE**; **CIFAR-10 (D4) is a GATED stretch** (start only after every P0 + every other P1 item is done, green, committed — then STOP and ask); **SVHN (D5) + ImageNet (D9) are OUT permanently.**
> - **D2 — NO faithful Chen conv Table-1 row.** Our MNIST rows are relabelled as *our own seeded baselines*; Chen's contribution to the submission is **C1–C4 only**.
> - **D3 — Rubanova is CUT entirely** (Latent-ODE / ODE-RNN / sine / spiral removed from the paper scope and from `make reproduce-all`). Code is *not deleted* — it is excluded from the reproduction pipeline and gated tests and documented in `OUT_OF_SCOPE.md`. Rubanova stays only as a related-work citation.
> - **C2 promoted to a first-class result:** Chen Fig 3c reports backward NFE at about half forward, and our adjoint did not reproduce that. *(The figure asserted here, backward approximately equal to forward, was itself withdrawn later: the measured ratio is 13 to 121 and depends on the adjoint tolerance. See `DEVIATIONS.md` row C2.)* This is *characterised* with a dedicated experiment (solver × tolerance × arch × ≥5 seeds), grounded in the torchdiffeq adjoint source, and gated behind a proven-correct NFE-split unit test.
> - **Assumed hardware for this machine: single NVIDIA RTX 3090 (24 GB), driver 595, CUDA 12.1.** Coursework env = conda `neural_odes` (torch 2.5.1 / torchvision 0.20.1 / torchdiffeq 0.2.5 / numpy 2.4.3 / scikit-learn 1.8.0 / matplotlib 3.10.9 / wandb 0.26.1 / rich 15.0.0 / pytest 9.0.3, python 3.11.15). Docker available.
>
> Execution order is fixed (§7): **Stage A (P0 infra) → Stage B (tests) → Stage C (experiments, cheapest first) → Stage D (CIFAR gate).** Stop and report after Stage A and after Stage C.

> **Design target.** ReScience C reviews are *hands-on*: a reviewer clones the repo and runs it on their own machine. The bar is **not** novelty or SOTA — it is (a) correctly targeting the *original* paper's claims, (b) a reimplementation faithful to the *paper* (not copied from the authors' code), and (c) code that *actually runs and reproduces the reported numbers*. Every recommendation below is designed against that bar, not against NeurIPS norms.

---

## 0. TL;DR (as-shrunk)

- **Primary: Dupont 2019 (ANODE).** Never formally replicated (only two informal, non-archival OpenReview 2019 reports, both MNIST/CIFAR-10). We replicate the *toy expressivity + generalization* half (separation, NFE growth, Fig-9 missing-slice) **plus the MNIST image row of Table 1** (matched-param NODE-vs-ANODE conv, 5 seeds). CIFAR-10 is a gated stretch; SVHN/ImageNet are out.
- **Secondary: Chen 2018, claims C1–C4 only.** Our solver/NFE ablation faithfully targets Chen Fig 3. The **O(1)-memory claim is on the wrong axis** and is redesigned (C4). Our MNIST rows are relabelled our own baselines — **not** a Chen Table-1 reproduction (no faithful conv row this round).
- **HEADLINE (C2, promoted):** Chen Fig 3c reports backward NFE ≈ ½ forward; our torchdiffeq adjoint gives **backward ≈ forward**. This is the most interesting result and goes in the abstract — but only after (a) a unit test *proves* our forward/backward NFE split is correct, and (b) a dedicated solver×tolerance×architecture×seed sweep characterises the ratio, grounded in the torchdiffeq adjoint source. If it turns out to be an instrumentation bug, we say so loudly.
- **Rubanova 2019: CUT** from the submission (kept in-repo, excluded via `OUT_OF_SCOPE.md`, related-work citation only). Its headline benchmarks (PhysioNet/MuJoCo/Human-Activity) are unreached and its paper has **no spiral** anyway (the 2-D spiral is Chen 2018's).
- **Reproducibility infrastructure is the main risk** and is Stage A: unpinned deps, unseeded MNIST tables, online-default `wandb.init`, personal-cluster SLURM files, a manual ANODE hand-aggregation (cluster job-IDs in filenames), and `scripts/test_*.py` smoke scripts pytest collects.
- **The report over-claims in ~9 places** (memory numbers, O(1) validation, single-seed spiral, slice-loss with std>mean, Rubanova framing) — retract/weaken, see §9.
- **Compute (shrunk):** the core submission is **~40–55 GPU-hours** on the RTX 3090; the CIFAR-10 stretch adds ~13–25. SVHN, a faithful Chen row, and any Rubanova benchmark are excluded. Per-experiment breakdown in §8.

---

## 1. Scope decision (the most important output)

### 1.1 Recommendation (APPROVED, shrunk)

| Paper | Role | What we replicate | What we do NOT claim |
|---|---|---|---|
| **Dupont 2019 (ANODE)** | **Primary** | (a) toy separation & NFE-growth (Figs 5–8 analogs); (b) missing-slice generalization = **Fig 9 reproduction**; (c) 1-D crossing-flow demo (Fig 3/D8); (d) **MNIST image row of Table 1** (matched-param NODE-vs-ANODE conv, 5 seeds) + Fig 10 (NFE-vs-loss) + Fig 13 (overfit NFE explosion); (e) **extension:** missing-slice grid | CIFAR-10 unless the gate is passed; SVHN, ImageNet; that we beat SOTA |
| **Chen 2018 (NODE)** | **Secondary (C1–C4 only)** | C1 solver/tolerance dynamics (Fig 3a–b); **C2 backward/forward-NFE characterisation (HEADLINE)**; C3 NFE-growth-in-training (Fig 3d); C4 *correctly-axised* O(1)-memory-vs-NFE experiment | that our MLP/CNN MNIST rows reproduce Chen's Table 1; CNFs; a faithful Chen conv Table-1 row (cut this round) |
| **Rubanova 2019 (Latent ODE)** | **CUT (related-work only)** | nothing — Latent-ODE/ODE-RNN/sine/spiral code excluded from the reproduction pipeline (see `OUT_OF_SCOPE.md`) | any replication of Rubanova; that the 2-D spiral is a Rubanova experiment |

This matches the approved shrunken scope, with three refinements the evidence forced (retained from the original analysis):

1. **The missing-slice experiment is *already a Dupont Fig 9 reproduction*, not a novel extension.** Dupont Fig 9 removes training points whose angle lies in `[0, π/13]` and shows NODE has a large generalization gap while ANODE reaches near-zero validation loss. Our `train_anode_slice_circles.py` removes an angular sector of width `π/5` and measures held-out-slice accuracy/loss — the same experiment. So the *replication* is Fig 9; the *added value* is turning Dupont's single qualitative slice into a systematic grid (see §6).
2. **Chen is only partly targeted today.** Our solver ablation genuinely reproduces Chen Fig 3; keep and harden it. But the MNIST accuracy/memory table is not Chen's Table 1 (different architecture, param count, epochs, solver, error rate). Treat it as our own baseline, or invest ~10–17 GPU-h to reproduce Chen's actual conv ODE-Net row.
3. **Rubanova cannot be a replication at our compute/scope.** Reframe honestly (§1.3). Human Activity is the *only* real benchmark cheap enough to attempt (UCI download, no registration, short sequences).

### 1.2 Argument, per paper

**Dupont — accept as primary.**
- *Not previously replicated formally* → maximal added value (see §2). The two existing informal reports cover only MNIST + CIFAR-10 classification + light ablation; we can subsume and extend them (toy expressivity claims, NFE analysis, generalization grid, archival peer-reviewed write-up).
- *We already target the toy half faithfully in spirit.* Report Table 2 (NODE 90.2±8.8 vs ANODE 100.0, NODE NFE median ~90 vs ANODE ~72–80) mirrors Dupont's story (Figs 5b, 6, 8: NODE unstable/high-NFE, ANODE solves it with flat NFE). Report Table 3 mirrors Fig 9.
- *The image tables are the paper's centrepiece and are absent.* Dupont Table 1: MNIST NODE 96.4±0.5 vs ANODE 98.2±0.1; CIFAR-10 53.7±0.2 vs 60.6±0.4; SVHN 81.0±0.6 vs 83.5±0.5 — all matched-parameter, 5 runs, conv ODE field (1×1→3×3→1×1). To be a *credible ANODE replication* we must reproduce at least MNIST + CIFAR-10. This is the single biggest piece of new work.

**Chen — accept as secondary, narrowed.**
- Our repo's real Chen contribution is the *solver-dynamics* story (Fig 3): report Fig 11 (NFE over training ↔ Chen Fig 3d), report Fig 12 (numerical error vs NFE, time vs NFE, backward vs forward NFE ↔ Chen Fig 3a–c). Keep, add seeds.
- **The O(1)-memory claim is on the wrong axis** and must be redesigned (see §3, §7). Chen's O(1) is asymptotic (Table 1 Memory column: ODE-Net `O(1)` vs RK-Net `O(L̃)` vs ResNet `O(L)`) and is specifically a property of the **adjoint** method. Our Table 7 sweeps a *ResNet's* discrete depth and gives the ODE-Net a single point — it demonstrates only the `O(L)` half. Correct test: fix the ODE-Net, drive its own forward+backward NFE up, show peak memory stays flat.
- Our MNIST Table 1 (MLP 204,650p/98.01%, CNN 75,722p/99.14%) is **not** Chen's Table 1 (conv ODE-Net, ~0.22M params, 0.42% error, implicit-Adams solver, ~100+ epochs, single run). Optional: reproduce Chen's real row from `torchdiffeq`'s documented recipe.

**Rubanova — reframe as extension; do not claim replication.**
- *Salvageability of the headline benchmarks:* PhysioNet (mortality AUC 0.833, interp MSE 2.361e-3), Human Activity (accuracy 0.846), MuJoCo Hopper (interp 0.36e-2 / extrap 1.44e-2). We run **none**. Reproducing them:
  - **PhysioNet:** free but needs account/registration; ~150–400 GPU-h for the full tables, ~30–80 for a single ODE-RNN-vs-GRU-D slice. **Recommend: drop** (registration + compute out of scope).
  - **MuJoCo Hopper:** data now easy (pre-generated `training.pt` or free MuJoCo), but Table 3 is a 6-model × 4-sparsity grid → ~100–250 GPU-h full, ~15–40 for a reduced slice. **Recommend: drop** (or a reduced slice only if we want one physics benchmark).
  - **Human Activity:** UCI download, no registration, short sequences → ~60–150 GPU-h full, **~15–40 GPU-h for a defensible single-slice** (ODE-RNN vs Latent-ODE-ODE-enc vs one RNN baseline, ≥5 seeds). **This is the only realistically-doable real Rubanova benchmark; keep as *optional* P2 to add one genuine data point.**
- Even our sine can only be compared to Rubanova's *toy* Suppl. Table 2, and with heavy setup drift (our latent_dim 20/nhidden 40 vs 10/20; RK4+Euler vs dopri5; fixed β=0.01 vs KL-annealed ELBO; 30 epochs vs unspecified; Adam vs Adamax). Present as "inspired-by / architecture validation," not a controlled reproduction.

### 1.3 What we explicitly do NOT claim

- We do **not** claim to reproduce Chen's Table 1 MNIST numbers (unless we run the faithful conv row).
- We do **not** claim Table 7 validates the O(1)-memory property (it doesn't; the corrected experiment does).
- We do **not** claim any Rubanova PhysioNet/MuJoCo/Human-Activity result unless we actually run it; the sine/spiral experiments are architecture validation.
- We do **not** attribute the 2-D spiral to Rubanova — it is from Chen 2018.
- We do **not** claim the Latent-ODE objective is Rubanova's ELBO (ours is MSE + β·KL, β=0.01).
- We do **not** make comparative claims from single-seed results (spiral Table 9) or from a metric whose std exceeds its mean (slice loss).

### 1.4 Decisions — RESOLVED (2026-07-13)

- **D1 — Image tables:** ANODE **MNIST is core**; **CIFAR-10 is a gated stretch** (Stage D — only after all P0 + all other P1 are done/green/committed, then stop and ask); **SVHN + ImageNet out permanently.**
- **D2 — Chen Table 1:** **NO** faithful conv row. Our MNIST rows are relabelled our own seeded baselines; Chen = C1–C4 only.
- **D3 — Rubanova:** **CUT entirely** from the submission (code kept but excluded from the pipeline, see `OUT_OF_SCOPE.md`; related-work citation only).

---

## 2. Existing replications (does ANODE already have one?)

**No — ANODE has not been formally replicated in a peer-reviewed reproducibility venue.** Evidence:

- **ReScience C published index** (`rescience.github.io/read/`): contains **no** NODE/ANODE/Latent-ODE replication.
- **ReScience NeurIPS-2019 special issue** (`github.com/ReScience/NeurIPS-2019`): the accepted-paper folders contain no ANODE paper; the only ODE-adjacent accepted item reproduces *Hamiltonian Neural Networks* (a different paper).
- **MLRC 2020/2021/2022** (`github.com/ReScience/MLRC` accepted.bib): none of the three papers.
- **TMLR reproducibility certifications:** none found for these papers.
- Two **informal** ANODE reproductions exist as NeurIPS-2019 Reproducibility Challenge *blind reports* on OpenReview only (not archival, not peer-reviewed):
  - "Reproducibility and Ablation Study of Augmented Neural ODEs" — `openreview.net/forum?id=u0UrD4Do_n` (MNIST/CIFAR-10 + fixed-step-solver ablation).
  - "Reproducibility of Augmented Neural ODEs" — `openreview.net/forum?id=O1IJIoWN6p` (MNIST 97.25±0.543 vs NODE 95.57±0.323; CIFAR-10 52.42±1.19 vs 51.72±0.571).

**Implication for added value.** The reproducibility-venue niche is open for all three papers. Chen NODE is the most reimplemented informally (torchdiffeq + tutorials), so it carries the least novelty; **ANODE and Latent ODE carry the most.** We must (i) *cite both OpenReview reports* and differentiate: broader scope (toy expressivity claims, NFE/compute analysis, generalization grid), independent-from-paper implementation, and a formal archival write-up; (ii) not rely on the two reports having "already done it" — they are narrow and non-archival.

---

## 3. Claim-by-claim gap analysis

Legend for **Faithful?**: ✅ faithful (minor drift) · ⚠️ partial / needs correction · ❌ not a reproduction as-is.
Compute = GPU-hours on one modern GPU (RTX 4090-class / A100-40GB); see §8 for assumptions.

### 3.A Dupont 2019 (ANODE) — PRIMARY

| # | Original claim | Fig/Table | Original setup (arch / data / params / epochs / seeds / solver / tol) | What our repo does now | Faithful? | Gap | Work required | Compute |
|---|---|---|---|---|---|---|---|---|
| D1 | NODE cannot separate nested regions; needs increasingly many NFE; ANODE solves it with ~flat NFE | Figs 5b, 6, 8 | MLP field `d+1→32→ReLU→32→ReLU→d`; concentric **spheres** (filled disk r≤0.5 + annulus 1.0≤r≤1.5; 1000 inner + 2000 outer); 50 epochs; **20 repeats**; RK45 **atol 1e-3 only**; lr 1e-3, batch 64 | `train_anode_circles.py`: MLP `ODEFunc` width **64**; sklearn-style thin **circles** (uniform angles, factor 0.5, 500/500); **500 epochs**; **3 seeds** {0,1,2}; dopri5 **atol=rtol=1e-3**; lr **3e-3** (Table 2) | ✅ (spirit) | vf width 64 vs 32; thin circles vs filled-disk+annulus; 3 seeds vs 20; 500 ep vs 50; lr 3e-3 vs 1e-3; adds rtol paper omits | Align dataset to Dupont's spheres (or justify circles); ≥5 seeds; document/justify hyperparam drift; commit raw+aggregated CSVs | ~2 |
| D2 | ANODE generalizes across an unobserved angular slice; NODE has large gap | Fig 9 | Concentric spheres, remove training points with angle ∈ `[0, π/13]`; NODE vs ANODE; 20 repeats | `train_anode_slice_circles.py`: remove sector width `π/5`; NODE vs ANODE-p2; held-out-slice acc/loss; **3 seeds** (Table 3) | ⚠️ | Slice width π/5 vs π/13; 3 seeds; **slice-loss std>mean** (0.2073±0.3591) unsupportable; full-val set includes the removed sector (dilutes the in-distribution metric) | ≥5–10 seeds; report median/IQR (skewed); add observed-region-only val metric; this is *the* Fig 9 reproduction — label it so | ~1 |
| D3 | **MNIST**: ANODE > NODE, matched params | Table 1 | conv field 1×1→3×3→1×1; NODE 92 filters=84,395p vs ANODE 64f+aug5=84,816p; batch 256; **5 runs**; RK45 atol 1e-3 | **Missing** (our MNIST is Discrete/NODE-MLP/NODE-CNN, not NODE-vs-ANODE-augmentation) | ❌ | Entire experiment absent | Build matched-param NODE-vs-ANODE conv on MNIST; 5 seeds | ~5 |
| D4 | **CIFAR-10**: ANODE 60.6±0.4 vs NODE 53.7±0.2 | Table 1 | conv field; NODE 125f=172,358p vs ANODE 64f+aug10=171,799p; batch 256; 5 runs; ~40 ep | **Missing** | ❌ | Entire experiment absent | Add CIFAR-10 pipeline (dataloader, aug), matched-param NODE/ANODE, 5 seeds | ~13–25 |
| D5 | **SVHN**: ANODE 83.5±0.5 vs NODE 81.0±0.6 | Table 1 | same conv arch as CIFAR-10; batch 256; 5 runs | **Missing** | ❌ | Entire experiment absent | Add SVHN pipeline; matched-param NODE/ANODE, 5 seeds | ~20–30 |
| D6 | ANODE reaches equal loss at ~half the NFE / far fewer iters on images | Fig 10 | conv field, p∈{0,1,10}, batch 256, 5 runs | **Missing** | ❌ | Absent (but cheap once D3/D4 exist) | Log NFE-vs-loss during D3/D4 training; plot | incl. in D3/D4 |
| D7 | ANODE trains stably; NODE NFE can exceed 1000 and loss explodes when overfitting MNIST | Fig 13 | conv MNIST, matched params, ~40 ep | **Missing** | ❌ | Absent | Extended-overfit run logging NFE/loss (falls out of D3) | ~2 |
| D8 | 1-D two-point `g_1d(-1)=1, g_1d(1)=-1` unrepresentable by a NODE flow (trajectories can't cross) | Fig 3, Prop 1 | 1-D MLP field; MSE | `data/synthetic.py:get_1d_crossing_data` exists but no trained experiment/figure | ❌ | Data helper present, experiment/plot absent | Add the 1-D crossing-flow demo + figure (very cheap, high pedagogical value) | <0.5 |
| D9 | ImageNet-64 (200 cls): ANODE ~10× faster, scales better | Fig 14 | conv field c=3, 164f=366k vs 64f+aug5=365k; batch 256 | **Missing** | ❌ | Out of scope | **Do not attempt** (~200 GPU-h alone) | — |

### 3.B Chen 2018 (NODE) — SECONDARY

| # | Original claim | Fig/Table | Original setup | What our repo does now | Faithful? | Gap | Work required | Compute |
|---|---|---|---|---|---|---|---|---|
| C1 | Solver dynamics: numerical error ↓ and forward time ∝ NFE as tol tightens | Fig 3a–b | trained MNIST ODE-Net, tol swept 1e-0…1e-5 | `plot_fig3.py` / `solver_ablation.py` sweep tol; report Fig 12 | ✅ | single unseeded run per config; solver state left mutated; wandb-only (no CSV) | ≥5 seeds; persist CSV; restore atol/rtol | ~1–2 |
| C2 | **Backward NFE ≈ ½ forward NFE** (adjoint cheaper than direct backprop) | Fig 3c | trained ODE-Net, implicit-Adams adjoint | report Fig 12 shows backward ≈ **forward** ("near-lockstep") | ⚠️ **divergence → HEADLINE** | Our torchdiffeq adjoint gives backward≈forward, contradicting Chen's ½ | **PROMOTED.** (1) unit-test the fwd/bwd NFE split first — headline can't rest on a buggy counter; (2) dedicated sweep {solver: euler/rk4/bosh3/dopri5/dopri8} × {tol 1e-1…1e-6} × {arch: MLP, conv} × {**torchdiffeq version**: 0.2.5 pinned + ≥1 earlier + ≥1 later if it exists, env otherwise fixed} × {≥5 seeds}, log fwd/bwd separately, report bwd/fwd ratio as primary quantity; (3) does the ratio ever approach 0.5? depend on solver order / tol / stiffness / torchdiffeq version? (4) explain via torchdiffeq `odeint_adjoint` augmented reverse system (state+adjoint+param-sens solved jointly) vs Chen's implicit-Adams — read the source, don't speculate. **The version axis is load-bearing: if the ratio holds only in 0.2.5 it is a bug report, not a replication finding.** If it's an instrumentation bug, report that loudly. | ~2–3 |
| C3 | NFE increases during training | Fig 3d | trained MNIST ODE-Net | report Fig 11 (NFE over training on circles) | ✅ | measured on circles not MNIST; seeds | Reproduce on MNIST conv ODE-Net; ≥3 seeds | ~2 |
| C4 | **O(1) memory** as a function of ODE-Net effective depth (adjoint) | Table 1 Memory col; §2 | asymptotic claim; adjoint reconstructs trajectory backward → O(1) regardless of NFE; contrast RK-Net O(L̃), ResNet O(L) | **Wrong axis:** Table 7 sweeps *ResNet* depth L (20.0→38.3→113.7 MB); ODE-Net is a single point (31.6 MB) | ⚠️ | The load-bearing half (ODE-Net memory flat as *its* NFE grows) is never measured; must confirm we use `odeint_adjoint` (we do — `continuous.py:4`) | New experiment: fix ODE-Net, drive NFE up (fixed-step step-count, or tol 1e-1→1e-7, or longer [0,T]); measure peak MB; contrast vs `odeint` (direct backprop = RK-Net) whose memory should rise | ~2–5 |
| C5 | ODE-Net matches ResNet accuracy on MNIST at fewer params, O(1) memory | Table 1 | conv ODE-Net downsample-twice + ODESolve(6 blocks); ~0.22M params; 0.42% err; implicit-Adams; tol 1e-3; ~100+ ep; **single run** | Our rows: NODE-MLP 204,650p/98.01%, NODE-CNN 75,722p/99.14%, Discrete-ResNet-MLP 204,650p/98.07% — **different experiment**; all **unseeded single runs** | ❌ | Not Chen's arch/params/epochs/solver/error; MNIST scripts have **zero seeding** | Either (opt. D2) reproduce Chen's faithful conv row (~10–17 GPU-h), or relabel ours as original baselines; add seeding regardless | ~3 (ours, seeded) / +10–17 (faithful) |
| C6 | Param-matched ResNet↔ODE-Net comparison is fair | §Model arch | — | DiscreteResNet weight-shares one `ODEFunc` → 204,650p == ODENet, independent of L | ✅ | invariant is untested; a refactor to per-layer weights would silently break fairness | Add assertion test (param parity across H, L) | <0.1 |

### 3.C Rubanova 2019 (Latent ODE) — **CUT from the submission** (table kept to document *why*)

> **Decision:** removed from paper scope and from `make reproduce-all`. Code stays in-repo but is excluded from the reproduction pipeline and gated tests (`OUT_OF_SCOPE.md`). Rationale below: we run none of Rubanova's headline benchmarks, and the 2-D spiral is not even a Rubanova experiment. Rubanova is cited as related work only.

| # | Original claim | Fig/Table | Original setup | What our repo does now | Faithful? | Gap | Work required | Compute |
|---|---|---|---|---|---|---|---|---|
| R1 | ODE-RNN/Latent-ODE beat RNNs on **PhysioNet** interp MSE (2.361e-3) & mortality AUC (0.833) | Tables 4–6 | 37-feat ICU, 8000 series, dopri5 rtol1e-3/atol1e-4, Adamax lr 0.01, KL-anneal 0.99 | **Missing** | ❌ | Entire benchmark absent | **Recommend drop** (registration + ~150–400 GPU-h) | — |
| R2 | Latent ODE best on **MuJoCo Hopper** interp/extrap | Table 3 | 14-dim, 10k sims, 3-layer 500-unit ODE field | **Missing** | ❌ | Entire benchmark absent | **Recommend drop** (or reduced slice ~15–40 GPU-h) | — |
| R3 | Latent-ODE(ODE-enc) best on **Human Activity** accuracy (0.846) | Table 7 | UCI, 12-feat, 6554×211, per-timepoint CE | **Missing** | ❌ | Entire benchmark absent | **Optional (D3):** single-slice, ODE-RNN vs Latent-ODE vs 1 RNN baseline, ≥5 seeds | ~15–40 |
| R4 | Toy **1-D sinusoid** interp/extrap (their *only* synthetic exp) | Suppl. Table 2 | 1000 1-D sines, 100 pts on [0,5], amp 1, freq~U[0.5,1]; latent 10/rec 20; 100-unit ODE; dopri5 rtol1e-3/atol1e-4; Adamax lr 0.01; KL-anneal 0.99 | `main.py` Latent ODE on sine (Table 5, **2 seeds**); standalone ODE-RNN/GRU (Table 4, 3 seeds) | ⚠️ | latent 20/40 vs 10/20; RK4+Euler vs dopri5; β=0.01 fixed vs annealed ELBO; 2–3 seeds; our baselines ≠ their baseline set (RNN-Δt/Impute/Decay/GRU-D/RNN-VAE) | Compare only to Suppl. Table 2; ≥5 seeds; label "architecture validation, not benchmark replication" | ~15–30 |
| R5 | 2-D spiral budget comparison | report Table 9 | **N/A — no spiral in Rubanova** (spiral is Chen 2018) | single-seed (42) spiral, 2 encoders × 2 budgets | ❌ (mis-attributed) | Single seed → no error bars; orderings flip with budget; wrong citation | Relabel as Chen-2018-derived; ≥5 seeds or drop comparative claim | ~10–20 |

---

## 4. Provenance / faithfulness audit summary (Step 4)

**Headline: our code is written *from the papers*, not lifted from the authors' code.** torchdiffeq is used strictly as the solver library (`from torchdiffeq import odeint_adjoint as odeint`) — the sanctioned use. None of the tell-tale fingerprints of the three upstream codebases appear in our modules. A `PROVENANCE.md` should record the following per module (reviewers can and will diff):

| Module (file) | Verdict | Why (vs upstream) |
|---|---|---|
| `ODEFunc` (`models/continuous.py`) | **written-from-paper** | `nn.Sequential` + state-first time concat `cat([h, t])`; Dupont uses `fc1/fc2/fc3` + `(device, time_dependent, non_linearity)` signature + time-first `cat([t, x])`; torchdiffeq demo cubes input, has no time/NFE. |
| `ODEBlock` (`models/continuous.py`) | **written-from-paper** | registers `integration_time` as a **buffer** (neither upstream does); no `options={'max_num_steps'}`, no `@property nfe`; own `return_trajectory` path. `return out[1]` is a paper-level idiom. |
| `ConvODEFunc` (`models/continuous.py`) | **architecture from ANODE App. F.1.2, implementation independent** | 1×1→3×3→1×1@64 matches the paper; but plain `nn.Conv2d` with a single input-time-concat vs Dupont's `Conv2dTime(nn.Conv2d)` subclass injecting time into *every* conv, opposite concat order. Our comment already credits the appendix. |
| ANODE augmentation (`models/networks.py`) | **from-paper, independently placed** | append-zeros is the defining ANODE op (paper-level); we augment *after* a `Linear+Tanh` downsampling stem Dupont's ODENet doesn't have. |
| `DiscreteResNet` (`models/networks.py`) | **original design** | weight-shares one `ODEFunc` across Euler steps (dt=1/L) for param-matched fairness; no upstream shares weights this way. |
| `ODERNNEncoder`/`LatentODE` (`models/continuous.py`) | **written-from-paper (concept), independent code** | stock `nn.GRUCell` (not Rubanova's custom probabilistic `GRU_unit`), direct `odeint` (no `DiffeqSolver` wrapper), generic `h_to_mu`/`h_to_logvar` VAE head (not Rubanova's `std.abs()` scheme). Reverse-time encoding is the paper's algorithm. |
| NFE counter (`self.nfe`/snapshot) | **conventional / from-paper** | the Chen-defined NFE metric; identical `self.nfe += 1` appears in all Neural-ODE codebases — *not* evidence of copying. Document as a community idiom. |

**Caveat to verify during `PROVENANCE.md` authoring:** confirm the training loop's NFE before/after-backward bookkeeping is *not* a near-verbatim copy of Dupont's `Trainer._get_and_reset_nfes` dict-of-histories structure (it is not, in the files read — our snapshot lives inline in `training/engine.py`).

---

## 5. Reproducibility infrastructure design (Step 5) — *where the submission lives or dies*

All of the following are **design only** (build after approval).

### 5.1 Single-command reproduction
- Add a top-level `Makefile`:
  - `make reproduce-all` — regenerate every table/figure from scratch.
  - Per-artifact targets: `make table2`, `make table3`, `make fig3`, `make anode-images`, `make memory-nfe`, etc.
  - **`make smoke`** — a `<5 min`, CPU-friendly path (tiny epochs / tiny n) that exercises every pipeline end-to-end so a reviewer can verify plumbing before committing GPU time. This is a strong positive signal to ReScience reviewers.
- Every target must run with **zero external accounts** (no W&B login).

### 5.2 Pinned environment (P0)
- `environment.yml` currently pins only `python=3.11` and `pytorch-cuda=12.1`; **torch/torchvision/torchdiffeq/numpy/scikit-learn/matplotlib/pandas/wandb are all unpinned** → a reviewer's `conda env create` resolves whatever is current and may not even solve against `pytorch-cuda=12.1`.
- Deliverables: pin exact versions; commit a resolved lockfile (`conda env export --no-builds` or `conda-lock`/`uv.lock`); ship a **Dockerfile or Apptainer `.def`** built from the lockfile. State the exact GPU/driver/CUDA and wall-clock for each committed number.

### 5.3 Determinism (audit + document)
- **MNIST scripts have no seeding at all**; `get_mnist_dataloaders` shuffles with no generator → Table 1/7 are unseeded single runs. Add `--seed` (python+numpy+torch+cuda) and a seeded `DataLoader` generator to both MNIST scripts.
- `main.py:set_seed` omits `np.random.seed` (inconsistent with `train_timeseries.py`); add it.
- Nothing sets `torch.use_deterministic_algorithms` / `cudnn.deterministic`. Add a `--deterministic` flag where feasible.
- **Document honestly:** adaptive `dopri5` + cuDNN are *not* bit-exact across GPUs/drivers. State the tolerance band within which results reproduce (e.g. accuracy within ±0.3%, NFE within a few evals) and the hardware the committed numbers came from.

### 5.4 Committed results → figures pipeline (P0 for ANODE)
- Sine/spiral figures already regenerate from committed `results/*.json` (good).
- **ANODE figures do not.** `plot_anode_corrected_results.py` reads hand-aggregated files `results/anode/final/circles_corrected_flat_table_job492199.csv` and `slice_corrected_flat_table_job492288_epochs500.csv` — **cluster-job-ID filenames, a directory that does not exist, and no committed input.** README instructs a manual "aggregate by `model_name` over seeds, then drop the file here" step. This is a hard review blocker.
- Deliverable: `aggregate_anode_results.py` doing `df.groupby('model_name').agg(['mean','std'])` → deterministic filenames; have the plotter call it; **commit raw + aggregated CSVs**.
- `.gitignore` currently ignores `results/`, `figures/`, `plots/` and the 19 JSONs were force-added against the rule → regenerated artifacts are silently untracked. Add explicit `!results/*.json` un-ignore rules (or a dedicated committed-artifacts dir).

### 5.5 Kill cluster-specific + make W&B optional
- **SLURM:** all 4 `.slurm` files hardcode `--account=3263572`, `--partition=stud`, `--qos=stud`, `module load /software/modules/miniconda3`, `cd /home/3263572/NeuralODEs`; `run_sweep_discrete.slurm:18` hardcodes a **personal W&B agent** `gaiagrossi-bocconi-university/NeuralODEs/cedblosn`. Genericise (env-var placeholders, `cd "$SLURM_SUBMIT_DIR"`) or move to an `examples/` dir with a "illustrative only" header, and remove the personal entity.
- **W&B is load-bearing:** every entrypoint does a top-level `import wandb`; `main.py`, `solver_ablation.py`, `train_anode_circles.py`, `train_anode_slice_circles.py` call `wandb.init` with **no offline mode** → interactive login/hang without an account. Deliverable: a pluggable logger with a **no-op/CSV backend as the default**, W&B opt-in via flag; make `import wandb` optional.

### 5.6 Tests (rename + expand)
- **Rename `scripts/test_continuous_mnist.py` / `scripts/test_discrete_mnist.py`** (→ `smoke_*.py` or `scripts/smoke/`): pytest collects them and they download MNIST + run a forward pass *at collection time*, breaking `pytest` for any reviewer. Add `pyproject.toml`/`pytest.ini` with `testpaths = tests`, `norecursedirs = scripts`.
- Existing `tests/` cover shapes / NFE>0 / grad-exists / hand-checked metrics. **Add:**
  - NFE counter correctness (forward vs backward split; exact count on a fixed-step solver where it's deterministic).
  - **Adjoint-vs-backprop gradient agreement** (`odeint_adjoint` vs `odeint` gradcheck on a tiny field).
  - Augmentation shape **and zero-init** of appended channels.
  - **Param-count parity** `ODENet == DiscreteResNet` across H and L (assert it — don't trust it).
  - Seed determinism (two seeded CPU forward passes bit-identical).

### 5.7 Archiving (submission-time)
- Tag the reviewed commit; mint a **Zenodo DOI**; add `CITATION.cff` + DOI badge; register with **Software Heritage** (`save code now`). No code defect — a submission checklist item.

### 5.8 Code hygiene (cheap, reviewer-visible)
- Delete empty dead file `models/discrete.py` (0 bytes).
- Remove dead imports: `from logging import config` (`continuous.py:5`), `from email import generator` (`synthetic.py:9`).
- Fix `train_discrete_mnist.py` CPU crash: it reads `train_metrics["memory_mb"]` unconditionally (`:131`, `:143`) but the engine only sets that key on CUDA → `KeyError` on any CPU-only reviewer machine. Use `.get("memory_mb", 0.0)` (as the continuous script does).
- Deduplicate the two `ODEFunc` definitions (`models/continuous.py` and `models/ode_rnn.py`) and consolidate the two parallel time-series stacks (`data/synthetic.py::TimeSeriesDataset` vs `data/timeseries.py::IrregularSineWaveDataset`) — or document why both exist.

---

## 6. Extension / added value: the missing-slice generalization grid (Step 6)

**Framing correction:** the base missing-slice experiment *is* a reproduction of **Dupont Fig 9**. The **added value** is making it systematic where Dupont was qualitative (one slice, no grid). Design to make it rigorous rather than anecdotal:

- **Grid (joint sweep):** augmentation dim `p ∈ {0, 1, 2, 3, 5}` × slice width `w ∈ {π/8, π/5, π/3}` × **≥5 seeds** (recommend 10, because the slice-loss distribution is heavy-tailed).
- **Metrics:** held-out-slice **accuracy AND loss**, reported as **median + IQR** (not mean±std — the current 0.2073±0.3591 with std>mean is exactly why). Also report an **observed-region-only** validation metric so in-distribution generalization is separated from held-out-sector extrapolation (currently the "full val" set includes the removed sector and blurs this).
- **Second geometry** (so the finding isn't a single-dataset artifact): add **two interleaving moons** (`make_moons` already exists) and/or **spirals** (`make_spirals` exists), each with an analogous held-out region. This is the key robustness upgrade.
- **Compute:** `5 (p) × 3 (w) × 5 (seed) × 2 (geometry) = 150` short runs (~2–5 min each) ≈ **~10 GPU-h**. With 10 seeds, ~20 GPU-h.
- **Claim we can then make:** "ANODE's generalization advantage over NODE across an unobserved region is monotone in the missing-region size and holds across ≥2 geometries" — a defensible, quantified extension of Dupont Fig 9. **Do not** keep the current single-metric mean±std claim.

---

## 7. Prioritised, dependency-ordered work plan

**Fixed execution order (do NOT reorder). Stop and report after Stage A and after Stage C.**

- **Stage A — P0 infra.** Everything in the P0 list below. **Acceptance gate:** a fresh clone in a fresh container, no W&B account, no network beyond dataset download, runs `make smoke` to completion (verified in a clean environment; command sequence + output reported).
- **Stage B — tests before experiments.** Expand `tests/` (P1-17). The **NFE forward/backward split test is written FIRST** because the C2 headline depends on it.
- **Stage C — experiments, cheapest first:** D8 → D1 → D2 → C1/C3 → C4 → C2 characterisation → D3/D6/D7 → §6 grid. After each: commit raw per-seed CSVs + aggregation script + figure script. No number reaches a figure by hand.
- **Stage D — GATE. Stop and ask before touching CIFAR-10 (D4).**

Standing rules: every table/figure regenerable from committed `results/*.csv|json` via a script via a `make` target (zero manual steps); report exact hardware + wall-clock per experiment; do not fake determinism (document what is/ isn't deterministic + the reproduction tolerance); any result contradicting the report is a *finding* (surface it, don't tune to agree); keep `CHANGELOG_REPLICATION.md` current.

**P0 — Stage A — blocks submission (a reviewer cannot run/reproduce without these).**
1. Pin environment + lockfile + container (§5.2).
2. Make W&B optional (no-op/CSV default); remove hard `import wandb` (§5.5).
3. Genericise/remove SLURM cluster specifics + personal W&B entity (§5.5).
4. Automate ANODE aggregation (`aggregate_anode_results.py`) and commit raw+aggregated CSVs; fix `plot_anode_corrected_results.py` (§5.4).
5. Un-ignore committed artifacts in `.gitignore` (§5.4).
6. Rename `scripts/test_*.py`; add `pytest.ini`/`pyproject.toml` `testpaths` (§5.6).
7. Seed the MNIST scripts + dataloaders; fix `train_discrete_mnist.py` CPU `KeyError` (§5.3, §5.8).
8. `Makefile` with `reproduce-all` + per-artifact + `make smoke` (§5.1).
9. `PROVENANCE.md` (§4).

**P1 — needed for a *credible* replication.**
10. Re-run at **≥5 seeds** + persist CSVs: ANODE circles (D1), missing-slice (D2), solver ablation (C1). *(Rubanova sine/spiral removed.)*
11. **C4 corrected O(1)-memory-vs-NFE** experiment: instrumented per-block memory (`§9-1` isolation), `odeint_adjoint` flat vs `odeint` (direct backprop) rising.
12. **C2 headline characterisation** — the fwd/bwd-NFE ratio sweep (see §3.B C2), *after* the NFE-split unit test in Stage B.
13. Reproduce **Chen Fig 3** dynamics on MNIST (C1, C3); relabel/refix the memory Table 1 CNN row and Table 7 (§9-1/§9-2).
14. **ANODE MNIST (D3) + NFE-vs-loss (D6) + overfit/NFE-explosion (D7)** — matched-param NODE-vs-ANODE conv, 5 seeds. *(D4 CIFAR-10 is Stage D, gated.)*
15. Align D1 dataset to Dupont's filled-disk + annulus "spheres"; vf width 32 per paper; document residual hyperparameter drift.
16. D2 as the Fig-9 reproduction: ≥5–10 seeds, median/IQR, observed-region-only val metric.
17. **Expand `tests/` (Stage B, first):** NFE fwd/bwd split correctness *(gates C2)*, adjoint-vs-direct-backprop gradient agreement, augment zero-init + shape, ResNet↔ODE-Net param parity, seed determinism.

**P2 — nice to have / strengthens submission.**
18. Missing-slice **grid + second geometry** extension (§6).
19. 1-D crossing-flow demo (D8, Stage C — cheap, high pedagogical value).
20. Determinism flags (`use_deterministic_algorithms`, `cudnn.deterministic`); code hygiene (delete empty `models/discrete.py`, dead imports, dedupe `ODEFunc`).
21. CI (GitHub Actions running `make smoke`); Zenodo/Software-Heritage at submission (§5.7).

**CUT this round (do not implement):** ~~faithful Chen conv Table-1 row (C5/D2)~~; ~~SVHN (D5), ImageNet (D9)~~; ~~any Rubanova benchmark — PhysioNet/MuJoCo/Human-Activity (R1–R3)~~; ~~sine/spiral re-runs (R4/R5)~~; ~~Latent-ODE faithfulness fixes~~ All are excluded, see `OUT_OF_SCOPE.md`.

**Dependency notes:** P0-1/2/3/5/6/7 unblock a clean clone-and-run; P0-4 unblocks the ANODE figures; Stage B P1-17 gates C2 (P1-12); P1-14 depends on P0-1 (pinned env) + P0-2 (offline logging); P2-18 depends on P1-10 infra; **D4 (CIFAR-10) depends on ALL P0+P1 done/green/committed → then stop and ask.**

---

## 8. Compute budget (GPU-hours)

**Assumed hardware:** the actual machine — one **NVIDIA RTX 3090 (24 GB), driver 595, CUDA 12.1** (matches the RTX 4090-class assumption in the original draft closely). **All numbers below must be re-timed on this machine and reported in the paper.** Ranges reflect NODE runs being dominated by rising/adaptive NFE.

| Experiment | Runs (configs × seeds) | GPU-h |
|---|---|---|
| D8 1-D crossing-flow demo | 1 | <0.5 |
| D1 ANODE circles, ≥5 seeds | 4 × 5 | ~2 |
| D2 missing-slice base, ≥5–10 seeds | 2 × 8 | ~1–2 |
| C1 solver/tolerance dynamics, ≥5 seeds | 15 × 5 | ~1–2 |
| C4 corrected O(1) memory-vs-NFE (adjoint vs direct) | ~15 short | ~2–5 |
| **C2 headline** fwd/bwd-NFE sweep (solver×tol×arch×≥5 seeds) | ~5×6×2×5 | ~4–8 |
| C3 + Chen Fig-3 dynamics on MNIST | few × 3 | ~2–4 |
| Our MNIST baselines, relabelled + seeded | 3 × 5, 10 ep | ~3 |
| **D3 ANODE MNIST** (+ D6 NFE-vs-loss, D7 overfit), 5 seeds | 2 × 5 | ~4–7 |
| §6 missing-slice **grid + 2nd geometry** | 150 | ~10–20 |
| **CORE SUBMISSION TOTAL (P0+P1+grid)** | | **≈ 30–55** |
| **D4 CIFAR-10 (gated stretch, Stage D)** | 2 × 5 | **~13–25** |
| ~~SVHN (D5), ImageNet (D9), faithful Chen Table-1 (C5), any Rubanova benchmark~~ | — | **excluded** |

**Bottom line:** the whole approved submission is **~30–55 GPU-hours** on the 3090; CIFAR-10, if the gate is passed, adds ~13–25. Cutting Rubanova removed ~25–50 GPU-h (and its slow per-sample encoder loop) from the previous ~100 GPU-h plan.

---

## 9. Things in the current report to retract or weaken

| # | Report claim | Problem | Action |
|---|---|---|---|
| 1 | **Table 1 CNN peak memory 3557.8 MB** (implicitly a continuous-depth memory number) | It is the whole-epoch `max_memory_allocated` dominated by 256×28×28 conv feature maps (~49 MB/tensor) + downsample Conv/BN + per-VJP autograd graphs + cuDNN workspace + Adam state — **not** adjoint activation memory. Comparing it to the MLP's 31.6 MB (160-float state) is apples-to-oranges (>1000× state-size/modality gap). *(NB: we checked — it is **not** contaminated by the post-training tolerance diagnostic, which runs after the per-epoch capture.)* | Remove the CNN number from the memory comparison, or re-measure with isolation (reset/read around the ODE block only, `cudnn.benchmark=False`, subtract baseline) and caption it as total conv training peak. |
| 2 | **Table 7 / App. A.2 "empirically validating the expected memory advantage"** | Table 7 only shows the *ResNet's* O(L) growth; the ODE-Net is a single point. The O(1)-in-NFE property is never tested. | Weaken to "we show ResNet memory grows with depth"; add the corrected experiment (C4) before claiming O(1) is validated. |
| 3 | **Table 1 CNN "aligning with original findings Chen et al."** | Our MNIST is not Chen's Table 1 (different arch/params/epochs/solver; 0.86% vs 0.42% error). | Drop the "aligns with Chen" phrasing; relabel as our own conv-ODE baseline (or run the faithful Chen row). |
| 4 | **Table 3 slice loss ANODE-p2 = 0.2073 ± 0.3591** | std > mean over n=3 → statistically meaningless; can't support "ANODE has low slice loss." | Re-run ≥5–10 seeds, report median/IQR; until then, drop the numeric slice-loss claim (keep slice accuracy, which is bounded). |
| 5 | **Table 9 (2-D spiral) comparative claims** ("ODE-RNN better at 30 ep, GRU catches up at 60 ep") | Single seed (42); orderings flip with budget → within-noise. | Re-run ≥5 seeds or demote to a labeled single-seed qualitative example with no comparative claim. |
| 6 | **Time-series section framed under Rubanova** (and spiral implicitly Rubanova) | Rubanova's claims are PhysioNet/MuJoCo/Human-Activity (none run); **Rubanova has no spiral** (spiral is Chen 2018). | Relabel: sine = architecture validation vs Rubanova's *toy* Suppl. Table 2 only; spiral = Chen-2018-derived. Remove any "failure/success to replicate Rubanova" language. |
| 7 | **"Latent ODE trained as a variational autoencoder" / ELBO** (App. C.3) | Objective is MSE + 0.01·KL with fixed β, not a calibrated Gaussian-NLL ELBO with annealing (Rubanova's). | Describe accurately as "MSE + β·KL, β=0.01," or implement the proper ELBO before claiming faithfulness. |
| 8 | **Table 4 vs Table 5 "ODE-RNN much better inside Latent ODE"** | Confounded: the two differ in epochs (10 vs 30), lr (1e-3 vs 1e-2), width, *and* architecture; the standalone extrap 0.921 is a free-run artifact of `observed_context` loss (future never supervised), not a fair ODE-RNN result. | Don't compare the tables directly; frame the standalone extrapolation blow-up as an expected free-run artifact. |
| 9 | **All MNIST numbers (Tables 1, 7)** | Produced by unseeded single runs (no `manual_seed` in the MNIST scripts, unseeded dataloader shuffle). | Re-run with fixed seeds (≥3), report mean±std; treat current numbers as provisional. |
| 10 | **Report Fig 12 "forward and backward NFE grow in near-lockstep"** presented as expected | Contradicts Chen Fig 3c (backward ≈ ½ forward). It's a genuine divergence, not a confirmation. | Reframe as a *replication finding*: "with the torchdiffeq adjoint we observe backward ≈ forward NFE, unlike Chen's ½" — characterize deliberately rather than assert it as expected. |

---

## Appendix — GitHub issue checklist (derived from §7)

Copy-paste as issues; labels: `P0` blocks-submission, `P1` credibility, `P2` nice-to-have; `infra`/`experiment`/`report`.

**P0 · infra**
- [ ] Pin exact versions in `environment.yml` + commit lockfile + Dockerfile/Apptainer (§5.2)
- [ ] Pluggable logging backend (no-op/CSV default); make `import wandb` optional; guard all `wandb.init` with offline default (§5.5)
- [ ] Genericise/remove all 4 `.slurm` files; delete personal W&B agent `gaiagrossi-bocconi-university/...` (§5.5)
- [ ] `aggregate_anode_results.py` (groupby model_name → mean/std, deterministic filenames); wire into `plot_anode_corrected_results.py`; commit raw+aggregated CSVs (§5.4)
- [ ] Un-ignore committed artifacts in `.gitignore` (`!results/*.json` etc.) (§5.4)
- [ ] Rename `scripts/test_continuous_mnist.py`/`test_discrete_mnist.py` → `smoke_*`; add `pytest.ini` `testpaths=tests` (§5.6)
- [ ] Seed MNIST scripts (python/numpy/torch/cuda) + seeded dataloader generator; add `np.random.seed` to `main.py:set_seed` (§5.3)
- [ ] Fix `train_discrete_mnist.py` CPU `KeyError('memory_mb')` (§5.8)
- [ ] `Makefile`: `reproduce-all`, per-artifact targets, `make smoke` (<5 min) (§5.1)
- [ ] Write `PROVENANCE.md` (per-file written-from-paper / adapted / library) (§4)

**P0/P1 · report**
- [ ] Retract/weaken items 1–10 in §9 (memory numbers, O(1) validation, slice loss, single-seed spiral, Rubanova framing, ELBO wording, Table4↔5, MNIST seeding, NFE-lockstep)

**P1 · experiment**
- [ ] Re-run ANODE circles, missing-slice, solver ablation, sine at **≥5 seeds** + persist CSVs
- [ ] **ANODE MNIST** matched-param NODE-vs-ANODE conv, 5 seeds (+ NFE-vs-loss, overfit/stability)
- [ ] **ANODE CIFAR-10** matched-param NODE-vs-ANODE, 5 seeds
- [ ] **Corrected O(1)-memory-vs-NFE** experiment + direct-backprop contrast + isolated memory measurement
- [ ] Chen **Fig 3** dynamics on MNIST (a–d), incl. backward-vs-forward-NFE characterization
- [ ] Relabel spiral as Chen-derived; ≥5 seeds or drop comparative claim
- [ ] Latent-ODE faithfulness: gate encoder GRU update on mask; fix objective description
- [ ] Expand `tests/` (adjoint-vs-backprop, augment zero-init, param parity, determinism, NFE)

**P2 · experiment / infra**
- [ ] Missing-slice **grid** (p × width × ≥5 seeds) + **second geometry** (moons/spirals) (§6)
- [ ] Dupont SVHN; 1-D crossing-flow demo
- [ ] *(D2)* Faithful Chen conv ODE-Net Table-1 row
- [ ] *(D3)* Human Activity single-slice (one real Rubanova benchmark)
- [ ] Determinism flags; batch the Latent-ODE encoder; code hygiene (delete `models/discrete.py`, dead imports, dedupe `ODEFunc`)
- [ ] CI (`make smoke`); Zenodo DOI + `CITATION.cff` + Software Heritage at submission

---

*End of plan. Awaiting review before any implementation (Phase 0).*
