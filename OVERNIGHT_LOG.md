# OVERNIGHT LOG — Stage C prep (autonomous session, 2026-07-15)

Append-only. Per entry: timestamp · what ran · **pre-declared refutation check** (written
BEFORE the run) · raw observation · pass/fail (decided by the observation, not by verdict
logic) · decisions. Lead the end-report with anything that contradicts a prior conclusion.

**Governing rule:** full autonomy over what to RUN; near-zero over what to CONCLUDE. Every
claim has a refutation written before the run. A result contradicting a theorem or a
Dupont/Chen paper number ⇒ STOP, leave plots + raw numbers, do NOT resolve alone.

**Environment:** RTX 3090 (24 GB, ~free at start), 16 CPU. Toy → CPU (faster for 2-D + no GPU
contention). MNIST → GPU. Accurate tol + reconstruction check on every flow result.

---

## 02:12 — Session start
- GPU free (1 GB/24 GB used, no large job). CPU 16 cores.
- Plan: D1 (NODE vs ANODE, spheres+circles, 50-ep budget) → D2 (missing-slice generalisation)
  → C2 recharacterisation → C1/C3 (MNIST NFE, GPU) → C4 (memory-vs-NFE, GPU). Stop before D3.
- Harness prep first: add `--geometry` + a per-(model,seed) wall-clock cap to
  `run_budget_sweep.py` so one pathologically-stiff seed cannot eat the night (record the cap
  as data, per instructions).

## 02:20 — Harness change (logged)
- `run_budget_sweep.py`: added `--geometry {spheres,circles}` and `--time_budget_s` per-(model,
  seed) wall-clock cap. When capped, still writes every budget row with `status=time_capped` +
  `epochs_done`, so the cap is DATA, not a gap. Fixed an epochs-tracking bug in the process
  (was mis-computing `done` when capped; now increments a real counter). Smoke-tested: cap fires
  at `@ep5`, both models, circles + spheres, every row carries `recon_ok`. Gated suite still
  green (will re-run before any commit).

## 02:22 — D1 PRE-DECLARED refutation checks (written BEFORE the run)
**Experiment:** NODE vs ANODE-p1, geometry ∈ {spheres (committed `results/budget/`), circles
(new `results/budget_circles/`)}, accurate tol 1e-6 + recon check, 5 seeds, budgets
{25,50,100,200,500}, cap 1200 s/seed. "Dupont d=2 reproduced" decomposed into falsifiable claims
— each with the observation that makes it FALSE (decided by the number, not by any verdict logic):

- **R1 (ANODE is cheaper — Dupont §5):** claim ANODE median fwd-NFE < NODE median fwd-NFE at 50 ep.
  FALSE if, aggregated over 5 seeds, median-NFE(ANODE) ≥ median-NFE(NODE), OR the gap is inside
  cross-seed spread (IQRs overlap with no separation).
- **R2 (NODE NFE grows, ANODE flat — Dupont §4.2 / Fig 6):** claim NODE median NFE rises ≥ +30%
  from budget 25→500 while ANODE rises < +30%. FALSE if NODE NFE is flat/decreasing, OR ANODE
  rises comparably to NODE.
- **R3 (both geometries carry the obstruction):** claim R1 and R2 hold on circles too. FALSE if
  circles shows no NFE gap and no NFE growth (signature would be spheres-specific).
- **R4 (ANODE ≥ NODE accuracy — Dupont §5 "better generalization/lower loss"):** claim ANODE
  dense-acc ≥ NODE dense-acc at 50 ep. FALSE if ANODE dense-acc < NODE dense-acc beyond noise.

**STOP-and-flag triggers (contradiction of a paper claim / theorem — do NOT resolve alone):**
- **S1:** any accurate-tol, `recon_ok=1` row with NODE dense-acc ≤ 0.70 (majority baseline) at
  ≥50 ep ⇒ NODE *fails to approximate* at d=2, contradicting Dupont's explicit "the NODE
  eventually learns to approximate g(x)" (§4.1). STOP.
- **S2:** ANODE strictly worse than NODE on BOTH accuracy AND NFE ⇒ contradicts Dupont's central
  thesis. STOP.
- (Note: dense-acc = 1.0 on the *finite* 12k test is NOT a theorem violation — the theorem
  forbids exact separation of the *continuum*; finite sets are threadable. The continuum check
  lives in the committed `topology_diagnostics` density sweep, not here.)

## 02:35 — Dupont D2 claim pinned + a doc discrepancy FLAGGED
Read Dupont §5.1 generalization (paper lines 300–305): *"create a validation set by removing
random slices of the input space (e.g. removing all points whose angle is in [0, π/5]) from the
training set. ... there is a large generalization gap for NODEs, presumably because the flow
moves through the gaps in the training set, ANODEs generalize much better and achieve near zero
validation loss."* His data is the concentric spheres; his example wedge is **[0, π/5]**.
- **FLAG for awake review (not rewriting committed docs):** `DEVIATIONS.md` A8 says "Dupont
  removes `[0, π/13]`"; the paper text I can read says **[0, π/5]**, which is exactly the width
  the coursework already uses. A8's "π/13" looks like an earlier misread. Leaving A8 as-is;
  D2 runs at π/5 (faithful to the paper text) and I note the mismatch here.

## 02:36 — D2 PRE-DECLARED refutation checks (written BEFORE the run)
**Experiment:** remove the angular wedge [0, π/5] from the TRAINING set; NODE vs ANODE-p1;
geometry ∈ {circles, spheres}; accurate tol 1e-6 + recon check; ≥5 seeds. Metrics on an
independent sample: full-val and **slice-val** (held-out wedge only = the sharp generalization
metric), acc + loss, accurate solver.
- **R-D2a (NODE has a generalization gap):** claim NODE slice-val loss ≫ NODE train loss. FALSE
  if NODE slice-val loss ≈ train loss (no gap), i.e. ratio slice-loss/train-loss < ~2.
- **R-D2b (ANODE generalizes better — Dupont's core D2 claim):** claim ANODE slice-val loss <
  NODE slice-val loss AND ANODE slice-val acc ≥ NODE slice-val acc, aggregated over ≥5 seeds
  beyond cross-seed spread. FALSE if ANODE slice-val loss ≥ NODE slice-val loss (medians, IQRs
  separated the wrong way).
- **STOP-and-flag S3:** ANODE generalizes *worse* than NODE on the held-out slice (higher loss
  AND lower acc, medians) ⇒ contradicts Dupont §5.1. STOP, leave plots + raw numbers.

## 02:50 — Background jobs launched
- D1 circles budget sweep (bg bs6k8941t) → `results/budget_circles/`.
- D2 missing-slice circles+spheres (bg bpinkgejq) → `results/slice_circles/`, `results/slice_spheres/`.
- D2 circles SMOKE (15 ep, 1 seed) lead: NODE slice-acc 1.000, gap -0.038 — **no NODE gap on
  circles**. NOT a conclusion (1 seed, 15 ep). Hypothesis: thin circles have no held-out 2-D
  *area* for the tendril to exploit (Dupont's data is *filled* spheres); spheres may differ. The
  5-seed/200-ep run + pre-declared R-D2a will decide.

## 02:52 — C2 recharacterisation PRE-DECLARED
**Experiment:** adaptive dopri5 bwd/fwd NFE ratio vs tolerance, both fields, ≥5 seeds, trained
100 ep at 1e-6, recon-checked (`scripts/c2_recharacterise.py`). Framing (headline/secondary/cut)
left to awake review — I only characterise.
- **Position under test (from tonight's committed withdrawal):** in the INTEGRATING regime
  (recon_ok) the ratio is ≫ 1 (not Chen's 0.5, not the old "≈1"), tolerance- and
  training-dependent, robust across seeds and both fields.
- **This position is REFUTED (i.e. the C2 withdrawal was premature) if:** the median
  integrating-regime ratio over seeds falls within **[0.5, 2.0]** on either field (which would
  make "backward ≈ forward" defensible after all). Band written before the run; observation
  decides. Raw per-seed ratios logged, never a bare verdict.
- Control: also measure an UNTRAINED (0-epoch) field for seed 0 — if it too shows a huge ratio,
  the effect is a pure solver property; if ~1-4×, it is training-induced stiffness. Either is data.

## 03:00 — C2 recharacterisation launched (bg bwgel2aqw) → results/c2/c2_recharacterise.csv

## 03:05 — C4 (O(1) memory vs NFE) PRE-DECLARED — GPU, self-contained
**Experiment:** fix a ConvODEFunc field (MNIST-shaped state), drive its forward NFE up by
tightening solver tolerance, and measure PEAK GPU memory for a forward+backward pass under
(a) `odeint_adjoint` (O(1) memory, recompute in backward) vs (b) plain `odeint` (direct
backprop, stores every intermediate state). `scripts/c4_memory_vs_nfe.py`.
- **Claim (Chen Table 1 / C3):** adjoint peak memory is ~flat as NFE grows; direct-backprop peak
  memory RISES with NFE.
- **REFUTED if:** adjoint memory rises with NFE with a slope comparable to direct (ratio of
  slopes > ~0.5), OR direct memory does NOT rise with NFE. Fit peak-mem vs NFE for both; compare
  slopes. Raw (NFE, mem_adjoint, mem_direct) rows logged.
- **Faithfulness checks (analog of recon):** (i) forward solve integrates at each tol (recon <
  1e-3 on the conv state); (ii) adjoint and direct gradients AGREE (max rel diff < 1e-2) so the
  two memory numbers are for the same computation. A row without both checks is not a result.
- **STOP-and-flag:** if adjoint gradients disagree with direct beyond tol, or adjoint memory
  exceeds direct at matched NFE (would contradict Chen's O(1) claim) ⇒ STOP, leave raws.

## 03:35 — C4 RESULT (3 seeds, `results/c4/`, `figures/c4/memory_vs_nfe.png`)
RAW (mean over seeds; NFE driven by tol 1e-3→1e-7): adjoint peak mem **442 MB flat** at NFE
14/20/26/38/92; direct backprop **689 → 3185 MB** over the same NFE. All 15 rows faithful
(recon < 5e-3, adjoint-vs-direct grad reldiff < 5e-3). Slopes: **adjoint +0.0007 MB/NFE**,
**direct +31.81 MB/NFE**, ratio 0.0000.
- **Pre-declared refutation NOT met** (adjoint/direct slope ratio 0.000 ≪ 0.5; direct slope ≫ 0).
  Chen's O(1)-memory-in-NFE claim (Table 1) **reproduced** by the corrected experiment (drives
  the ODE-Net's OWN NFE, not the Euler baseline depth the coursework varied). No theorem/paper
  contradiction. Faithfulness checks (recon + grad agreement) on every committed row.
- Decision: commit as a complete experiment (CSV + script + plot + this log entry). DONE (2b3d053).

## 03:40 — C1/C3 (MNIST NFE-over-training) PRE-DECLARED + feasibility gate
**Claim (Chen 2018, NFE grows):** the forward NFE of the MNIST conv ODE-Net INCREASES over
training as the dynamics get more complex. **Refuted if** forward NFE is flat/decreasing over
epochs (median over ≥5 seeds).
**Faithfulness:** the eval tolerance must integrate — per-epoch reconstruction check on a fixed
test batch (recon < 1e-2). A tolerance where recon fails is not measuring a flow.
**Feasibility gate (compute ceiling):** MNIST at *toy-tight* tol could explode the backward NFE
(C2 showed ~10^4 bwd NFE at 1e-6 on toy). Before a ≥5-seed run I PROBE 1 epoch × 1 seed: measure
epoch wall-clock + recon at candidate tols. Pick the LOOSEST tol whose recon still passes (the
conv field is smoother than the toy tear — C4 showed recon 4.7e-3 already at 1e-3 for an untrained
conv field). If no tractable faithful tol exists within ~3 GPU-h/run, RECORD THAT AS A FINDING
("MNIST NFE-over-training not faithfully measurable within the ceiling") and move on — do not
blow the night on one experiment.

## 03:50 — MNIST probe result + C1/C3 launched (bg b8k3xo0sm) → results/mnist_nfe/
PROBE (1 seed, 100 batches, tol 1e-3, eval_tol 1e-5): fwd NFE 22.6, **bwd NFE 25 (BOUNDED — no
explosion)**, recon 7.3e-4 (faithful), 31 s / 100 batches → ~2.4 min/epoch full.
- **Important cross-experiment finding:** the backward-NFE blow-up we saw on the toy tear
  (C2: ~10^4 at 1e-6) does NOT happen on the MNIST conv field — bwd ≈ fwd here. So the C2
  explosion is a property of the *near-singular toy tear*, not of the adjoint in general. This
  directly bears on C2 framing (the huge ratio is task-specific). Logged for the awake review.
- Feasibility gate PASSED: tol 1e-3 train (recon-checked at 1e-5), 5 seeds × 10 epochs ≈ 2 GPU-h.
  Launched `run_mnist_nfe.py`. Claim under test: forward NFE grows over training; faithful_fwd_nfe
  (recon-checked) logged alongside train NFE so the growth is measured in the integrating regime.

## 04:05 — DECISION: D2 relaunched leaner (killed the 200-ep/both-geom job)
The first D2 job (200 ep × 40 runs at 1e-6 under 3-way CPU contention) was on track for 10+ h —
would blow the ceiling. Killed it (bpinkgejq); the partial (2 circles NODE rows) is saved to
scratchpad/d2_old. Relaunched (b4x3m87mt) **spheres-first** (Dupont's actual data), 150 epochs
(the gap was clearly present at 200 ep on circles s0 — slice_loss 1.25 vs train 0.26 — and the
flow contorts progressively, so 150 should still show it), 5 seeds, NODE + ANODE-p1. This is a
scope/runtime trim, NOT a conclusion change; recorded here. Circles runs after spheres if time.

## 04:40 — D2 FINAL trim (bde0166p3): spheres-only, 100 ep, 5 seeds
The 150-ep/both-geom relaunch was still ~30 min/run under 3-way contention (would need ~5 h for
spheres alone). Final decision: **spheres-only** (Dupont's faithful data, where the gap is
dramatic — NODE s0 150 ep gave slice-acc 0.613 / gap +8.64 vs circles' modest 0.87/+1.0),
**100 epochs** (the gap is huge by 150, robust by 100), 5 seeds, NODE + ANODE-p1. Dropped
circles-D2 (secondary; the earlier partial + the circles budget sweep cover circles). Partial
150-ep spheres saved to scratchpad/d2_old. Two intervening restarts logged — trims to fit the
night, no conclusion changed. STOP triggers (S3) unchanged.

## 04:55 — D1 RESULT (spheres 5 seeds committed + circles; `scripts/d1_report.py`)
RAW (median over seeds, accurate tol 1e-6, recon-checked):
| | @50 dense_acc | @50 NFE | NFE growth 25→500 |
|---|---|---|---|
| spheres NODE | 0.996 | 218 | **×1.74** |
| spheres ANODE-p1 | 0.9996 | 170 | ×1.04 (flat) |
| circles NODE | 0.998 | 170 | **×1.37** |
| circles ANODE-p1 | 0.9996 | 128 | ×1.12 (flat) |
- **All pre-declared checks pass, NONE refuted, on BOTH geometries (R3):** R1 ANODE cheaper
  (ANODE NFE < NODE NFE); R2 NODE NFE grows ≥+30% while ANODE stays <+30% flat; R4 ANODE acc ≥
  NODE acc. **S1 STOP not triggered:** min NODE dense-acc ≥0.925 (≫0.70) — the NODE "eventually
  approximates" at d=2, matching Dupont §4.1; no accuracy-collapse, no paper contradiction.
- Honest caveats carried in the CSV: spheres NODE recon_ok 4/5 at budget ≥200 (stiffest seed
  exceeds 1e-6 faithfulness — same finding as the pre-overnight budget sweep); one circles NODE
  seed hit the wall-clock cap at budget 500 (status=time_capped, recorded not dropped).
- Circles ANODE was mid-flight when this was computed (3–5 seeds/budget); values are already
  tight (0.9996–0.9999, NFE 128–143) so the conclusion is stable. Final commit uses full data.
- D1 committed ff98e8e (full 5 seeds both geometries).

## 05:15 — D2 RESULT (spheres, 100 ep, 5 seeds; `results/slice_spheres/`, `figures/slice/`)
RAW (median[IQR], recon_ok 5/5 for BOTH models — all faithful):
| model | train_loss | SLICE acc | SLICE loss | full_val_acc |
|---|---|---|---|---|
| NODE | 0.000 | **0.619** [0.613, 0.921] | **6.089** [0.41, 7.50] | 0.963 |
| ANODE-p1 | 0.000 | **1.000** [1.000,1.000] | **0.000** | 1.000 |
- **Textbook Dupont §5.1 / Fig 9 reproduction.** NODE fits training perfectly yet collapses to
  ~chance (0.619) on the held-out wedge — the flow threads the inner class OUT through the removed
  wedge (a real 2-D hole in the training support), so held-out points there are misclassified.
  ANODE lifts the inner class into the augmented dim and generalises perfectly (slice-acc 1.000).
- **Pre-declared checks:** R-D2a (NODE gap) NOT refuted (slice_loss 6.09 ≫ train 0.00). R-D2b
  (ANODE better) NOT refuted (ANODE slice_loss 0.00 < NODE 6.09; ANODE slice_acc 1.00 > NODE
  0.62). **S3 STOP not triggered** (ANODE not worse). No paper contradiction — this is the
  direction Dupont predicts.
- Honest nuance: NODE slice-acc IQR [0.613, 0.921] — most seeds ≈ chance, one seed 0.921 (high
  cross-seed variance); median 0.619. ANODE is rock-solid 1.000 all seeds.
- geometry note: spheres shows a FAR larger gap than circles (circles NODE s0 earlier: slice-acc
  0.87 / gap +1.0). Thin circles lack a held-out 2-D *area*; the filled-sphere geometry is where
  Dupont's gap is dramatic. Circles-D2 dropped for time (secondary; partial in scratchpad/d2_old).
- FLAG (unchanged): `DEVIATIONS.md` A8 says Dupont's wedge is π/13; paper text says [0,π/5]. Used π/5.
- D2 committed d394a45.

## 05:30 — C1/C3 MNIST preliminary (seeds 0-1 done; 2-4 running) + a faithfulness caveat
RAW (mean seeds {0,1} by epoch): train-tol(1e-3) NFE **26.1→38.0** (peak ep3) then ~36; faithful
-tol(1e-5) NFE **86→122**; test acc 0.70→0.96.
- **Chen's NFE-growth direction is reproduced** (NFE rises over training; refutation "flat/
  decreasing" NOT met).
- **HONEST CAVEAT (recon check did its job):** the conv field STIFFENS with training — recon at
  eval_tol 1e-5 drifts from 2e-3 (ep1) to ~1.2e-2 (ep5+), so **recon_ok only 9/20 epochs**;
  past ~ep4 neither 1e-5 nor (a fortiori) the 1e-3 training tol is strictly integrating (~1%
  rel error). So the *late-epoch* NFE numbers are at the edge of faithfulness. This is the SAME
  tolerance-faithfulness phenomenon as the toy tear, milder/slower — it reinforces tonight's
  central theme rather than contradicting anything. A fully-faithful 10-epoch curve needs eval_tol
  ~1e-6/1e-7. NOT re-running tonight (compute/time); reporting with the recon_ok flag visible.
- Decision: let the 5-seed run finish; commit with the caveat + per-epoch recon_ok. This is a
  bounded, honest partial (NFE-growth shown in the recon_ok regime), NOT a clean full-budget claim.

## 06:40 — C1/C3 MNIST FINAL (5 seeds; `results/mnist_nfe/`, `figures/mnist_nfe/nfe_over_training.png`)
RAW (mean over 5 seeds, by epoch): train-tol(1e-3) NFE / faithful-tol(1e-5) NFE / recon_ok:
| epoch | 1 | 2 | 3 | 4 | 5 | ... | 10 |
|---|---|---|---|---|---|---|---|
| train NFE | 25.4 | 29.0 | 36.8 | 36.8 | 35.9 | | 37.2 |
| faithful NFE | 78.8 | 92.0 | 101.6 | 106.4 | 108.8 | | 120.8 |
| recon_ok (5 seeds) | 1.0 | 1.0 | **1.0** | 0.6 | 0.2 | | 0.0 |
- **Pre-declared refutation (NFE flat/decreasing) NOT met** — NFE grows: train 25→37, faithful
  79→121. **Chen's NFE-growth direction reproduced.**
- **Fully-faithful window = epochs 1–3 (recon_ok 5/5):** train 25.4→36.8, faithful 78.8→101.6.
  Past epoch 3 the conv field stiffens beyond 1e-5 (recon_ok falls to 0), so later-epoch NFE is
  edge-of-faithful — the growth continues but the numbers are at ~1% integration error. Same
  tolerance-faithfulness phenomenon as the toy tear, slower onset. Transparent via recon_ok/row.
- **This is NOT a clean full-budget C1/C3 claim.** For a faithful 10-epoch curve, re-run at
  eval_tol 1e-6/1e-7 (left for awake decision — it multiplies the recon-check cost). Committing
  the 5-seed run with the caveat and per-row recon_ok, not hiding the faithfulness limit.

## END OF AUTONOMOUS SESSION (part 1)
Delivered: D1, D2, C4 fully (≥5 seeds, committed, gated-green); C2 recharacterised (framing left
open); C1/C3 MNIST NFE-growth reproduced with a stated faithfulness caveat. No STOP triggered; no
prior conclusion inverted; one self-caught imprecision corrected (C2 "toy-specific"→tolerance-
driven). Stage D / CIFAR / paper untouched. Flags for review: C2 framing; DEVIATIONS A8 π/13 vs
paper [0,π/5]; MNIST faithfulness (re-run tighter?); spheres NODE recon 4/5 at budget≥200.
- Early MNIST signal (not yet a conclusion): fwd NFE 25.3→26.2→35.5 over epochs 1-3 (recon OK) —
  Chen's growth is appearing; will confirm at 10 ep × 5 seeds.

## 04:20 — C2 RECHARACTERISATION RESULT (5 seeds × 2 fields × tol axis; `results/c2/c2_recharacterise.csv`, `figures/c2/bwd_fwd_ratio_vs_tol.png`)
RAW (median bwd/fwd ratio over 5 seeds, trained 100 ep):
| tol | spheres | circles | recon_ok |
|---|---|---|---|
| 1e-3 | 6.1 | 6.3 | **0/5 (non-integrating)** |
| 1e-4 | 14.1 | 12.0 | 1/5 |
| 1e-5 | 32.6 | 29.4 | 5/5 |
| 1e-6 | 60.7 | 59.0 | 5/5 |
| 1e-7 | 93.7 | 96.6 | 5/5 |
Untrained control (seed 0): ratio 3.6→59 (spheres), 4.0→69 (circles) across 1e-3→1e-7.
- **Pre-declared refutation of the withdrawal NOT met:** integrating-regime median ratio 52–58
  (range 13–121), far outside the [0.5,2.0] "≈1" band. The C2 headline "backward ≈ forward"
  stays withdrawn.
- **Correcting my own earlier imprecise note (03:50):** the huge ratio is NOT "toy-specific" — it
  is **primarily tolerance-driven** and appears even in an **untrained** field (ratio ~32× at
  1e-6 with no training), so it is largely a property of the adjoint's reverse *augmented* system
  (state + adjoint + param sensitivities: higher-dim, stiffer) than of the forward. Training adds
  ~2× on top. The reason MNIST shows bwd≈fwd (~1.1) while the toy shows 30–100× is the **faithful
  tolerance differs**: the near-singular toy tear only integrates at tight tol (1e-5+), where the
  backward is expensive; the smooth MNIST conv field integrates at 1e-3, where it isn't. So the
  operative variable is "what tolerance does this field need to be integrated," and the answer is
  field-dependent.
- **The small-ratio regime is the non-faithful regime:** at 1e-3/1e-4 (ratio ~4–14, closest to
  the old "≈1" and Chen's 0.5) the trained field FAILS the recon check (0/5, 1/5) — those numbers
  are not a solve. Any bwd/fwd headline measured at loose tol is measuring a non-integration.
- **FRAMING (headline / secondary / cut) LEFT TO AWAKE REVIEW**, per instructions. My
  characterisation: C2 is real and interesting but is a *tolerance×field* surface, not a single
  ratio; it should not be an abstract headline as a number.
- Committing as a complete experiment.

---

# Overnight session — part 2 (continue: finish Dupont + Chen)

Order: 0 blocker (D2 wedge) → 1 C2 consolidation → 2 C1/C3 tighter-tol → 3 D3 ANODE-MNIST →
STOP at CIFAR gate. Same discipline: pre-declared refutations, raw observation decides,
STOP-and-flag on any theorem/paper contradiction.

## 07:30 — [0] BLOCKER RESOLVED: D2 wedge = [0, π/5], MATCHES paper (no re-run)
Rendered `references/1904.01681v3.pdf` p.6 to an image and read it directly (pdftotext mangles
the fraction). **§5.1, p.6 verbatim:** *"we can also create a validation set by removing random
slices of the input space (e.g. removing all points whose angle is in [0, π/5]) from the training
set... there is a large generalization gap for NODEs... ANODEs generalize much better and achieve
near zero validation loss."* Grepped the whole paper: **π/13 appears nowhere**; the coursework
report doesn't use it either. So A8's "[0, π/13] / wider wedge" was a sourceless misread.
- **Our committed D2 ran at π/5 → MATCHES the paper. No re-run needed.** A8 corrected (with the
  p.6 citation and a dated correction note; factual fix authorised by the continue-prompt item 0,
  done transparently not silently).
- Same page cross-confirms our other results: Fig 8-right = NODE NFE rising ~18→37 / ANODE flat
  (our D1); Fig 9-left = the NODE tendril threading the annulus (our topology finding); Fig 9
  mid/right = NODE train/val gap vs ANODE overlap (our D2). Note Dupont's Fig 9 loss plots run
  ~20 epochs; our D2 used 100 (longer; same direction, larger gap) — minor deviation, logged.
- Also captured for D3/D6 from p.6 §5.2: *"to compute a function which obtains a loss of 0.8 on
  CIFAR10, a NODE requires approximately 100 function evaluations whereas ANODEs only require 50"*
  and *"On MNIST... ANODEs... achieve the same loss in roughly 10 times fewer iterations."*

## 07:45 — [1] C2 consolidation PRE-DECLARED (framing DECIDED by co-author: demoted secondary, out of abstract)
Goal: ONE committed artifact = bwd/fwd NFE ratio as a **tolerance × field** surface across
{toy spheres, toy circles, MNIST conv} × {untrained, trained} with recon_ok per cell. I add
MNIST (new measurement) to the committed toy data (`c2_recharacterise.csv`).
- **Through-line claim (to characterise, not headline):** the adaptive backward/forward NFE
  ratio is governed by *the tolerance a field needs to be faithfully integrated*. Smooth fields
  (MNIST conv) integrate at loose tol → small ratio; near-singular fields (toy tear) need tight
  tol → large ratio. So it is tolerance-driven, not per-field magic.
- **Pre-declared refutation:** at MATCHED tolerance, MNIST and toy bwd/fwd ratios are the same
  order of magnitude (ratio ≈ f(tolerance), not f(field)). REFUTED if, at a matched recon_ok
  tolerance, MNIST ratio and toy ratio differ by >5×. (Also: if MNIST needs the SAME tight tol as
  the toy to integrate, the "smooth field integrates loose" half is wrong — that too is data.)
- Raw per-(field,state,tol) rows logged; recon_ok per cell; no bare verdict. Prose framing left
  as a stub for the co-author (I write the surface + result, not paper narrative).

## 08:30 — [3] D3 (ANODE vs NODE, MNIST, matched params) PRE-DECLARED + COMPUTE FLAG
Model faithful to App F.1.2 (conv field 1x1 k -> 3x3 k -> 1x1 c, time channel before each conv =
our ConvODEFunc) + Dupont image augmentation (add p zero channels to the input image, no
Chen-style downsampling) + flatten->linear head. **Param counts asserted & matched:** NODE k=92
= 85,316; ANODE-p5 (aug5, k=64) = 85,462 (146 apart, 0.17%); ~1% above Dupont's 84,395/84,816 (a
head/padding detail — documented deviation, not tuned). Pre-check: the D3 image field integrates
faithfully at eval-tol 1e-5 (recon 3.8e-4 at 2 ep) — unlike the C1/C3 64-channel field; per-row
recon_ok will confirm as it trains.
- **Pre-declared refutation (Dupont's core Table-1 claim):** ANODE reaches ≥ NODE test accuracy at
  matched params WITH lower/flatter forward NFE. REFUTED if ANODE test-acc < NODE test-acc
  (median, 5 seeds), OR ANODE forward NFE ≥ NODE forward NFE (not cheaper).
- Report raw vs Dupont Table 1 (**MNIST NODE 96.4±0.5 / ANODE 98.2±0.1**). A MISS is a legitimate
  partial-replication finding — will NOT tune toward the paper's numbers. D6 (NFE-vs-loss) and D7
  (train/test gap vs NFE) fall out of the same trajectory.
- **COMPUTE FLAG (per ceiling rule):** est. ~3 GPU-h for 5 seeds × 2 models × 8 epochs, batch 256
  (NODE ~2.8 min/ep, ANODE ~1.7). At/around the ~3-GPU-h single-run soft limit; total budget has
  room (~2 GPU-h used so far). Proceeding at 8 epochs (fewer than a fully-converged budget — so a
  shortfall vs Table-1 absolute numbers is expected and is a documented deviation, not a failure
  of the comparison). Incremental CSV → partial survives. Not CIFAR; stopping at the CIFAR gate.

## 08:45 — [1] C2 CONSOLIDATION RESULT (`results/c2/c2_surface.csv`, `figures/c2/c2_surface.png`)
Surface = bwd/fwd ratio (median) over {spheres, circles, mnist_conv} × {untrained, trained} × tol,
with recon_ok per cell. RAW highlights (trained; recon_ok in parens):
| tol | spheres | circles | mnist_conv |
|---|---|---|---|
| 1e-3 | 6.1 (0.0) | 6.3 (0.0) | 10.9 (0.0) |
| 1e-4 | 14.1 (0.2) | 12.0 (0.2) | 30.5 (0.0) |
| 1e-5 | 32.6 (1.0) | 29.4 (1.0) | 104.8 (1.0) |
| 1e-6 | 60.7 (1.0) | 59.0 (1.0) | — |
| 1e-7 | 93.7 (1.0) | 96.6 (1.0) | — |
- **Through-line CONFIRMED (tolerance-driven):** ratio grows steeply & monotonically with tighter
  tol on ALL three fields; present even untrained. The adjoint's reverse *augmented* system is
  inherently stiffer/higher-dim than the forward.
- **Pre-declared "same order across fields (<5×)" = PARTIAL PASS (honest):** at 1e-5 (all
  recon_ok) *trained* toy ~30–33 vs MNIST 105 = **~3×** (within band, not refuted); *untrained*
  toy 10.7 vs MNIST 65 = **~6×** (over band, refuted). ⇒ the ratio is tolerance-driven AND
  field-stiffness-dependent (conv field intrinsically stiffer). NOT purely f(tolerance). Reported
  as-is, not massaged.
- **Small-ratio regime = non-integrating:** for trained fields, 1e-3/1e-4 (ratios ~6–30, nearest
  Chen's 0.5 / the old "≈1") FAIL the recon check (recon_ok 0) on all three fields. Any bwd/fwd
  headline at loose tol is measuring a non-solve. This is the operative caution.
- Framing (demoted secondary, out of abstract) unchanged; prose stub left for co-author. C2 CLOSED.

## 09:00 — [2] C1/C3 stiffening confirmation PRE-DECLARED
`run_mnist_stiffening.py`: MNIST conv (64f), train tol 1e-3, per epoch measure recon + fwd NFE at
a LADDER of eval tols {1e-5,1e-6,1e-7} (capped solves), 5 seeds, 8 epochs.
- **Claim:** (a) faithful-NFE (at the tightest recon_ok tol) grows monotonically over training;
  (b) the LOOSEST tol that stays recon-faithful TIGHTENS with training.
- **Refuted if:** faithful-NFE is flat/decreasing once recon_ok, OR the loosest recon_ok tol does
  NOT tighten (e.g. 1e-5 stays recon_ok at all epochs -> no stiffening -> the C1/C3 finding was
  wrong). Raw recon_rel per (epoch, tol) logged.

## 10:30 — [2] C1/C3 STIFFENING RESULT (5 seeds; `results/mnist_stiffening/`, `figures/mnist_stiffening/stiffening.png`)
RAW recon_ok fraction (epoch × eval_tol):
| eval_tol \ epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1e-5 | 1.0 | 1.0 | 1.0 | **0.6** | **0.4** | **0.6** |
| 1e-6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1e-7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
Faithful NFE (at 1e-7): **384 → 738** over 6 epochs (×1.9), all recon_ok.
- **Both pre-declared claims CONFIRMED, refutation NOT met:** (a) faithful-NFE grows monotonically
  (384→499→574→644→699→738); (b) the recon-faithful tolerance TIGHTENS — 1e-5 recon_ok fraction
  falls 1.0→0.4 at epochs 4-6 while 1e-6/1e-7 hold. The stiffening is real (NOT removed by tighter
  tol: the NFE at 1e-6/1e-7 grows too). The transition is underway by epoch 6 (C1/C3's 10-epoch run
  showed 1e-5 fully failing) — honest partial, per-seed recon_ok logged.
- Confirms the C1/C3 finding is a MECHANISM (conv field stiffens with training), not a defect.

## 12:20 — [3] D3 RESULT (ANODE vs NODE, MNIST, matched params, 5 seeds; `results/d3/`, `figures/d3/`)
Params ASSERTED matched: NODE 85,316 vs ANODE-p5 85,462 (0.17% apart; ~1% above Dupont's
84,395/84,816 — head/padding detail, documented). Train tol 1e-3, eval/recon 1e-5, batch 256,
8 epochs. RAW (median[±std] over 5 seeds, epoch 8):
| model | test acc | train-tol NFE | faithful(1e-5) NFE | recon_ok |
|---|---|---|---|---|
| ANODE-p5 | **0.9818 ± 0.0029** | 26.0 | 50 | 5/5 |
| NODE | **0.9453 ± 0.0044** | 32.5 | 86 | **2/5** |
NFE growth 25→500... over training: NODE ×1.59, ANODE ×1.31.
- **Pre-declared refutation NOT met (both criteria) ⇒ Dupont's core Table-1 claim REPRODUCED:**
  ANODE ≥ NODE test-acc (0.982 vs 0.945) AND ANODE cheaper NFE (train-tol 26<32.5; faithful
  50<86, a 1.7× gap once you actually integrate). ANODE also learns faster (reaches its plateau
  by ~ep4).
- **vs Dupont Table 1 (NODE 96.4±0.5 / ANODE 98.2±0.1):** our **ANODE 98.18 ≈ Dupont 98.2 (match)**;
  our **NODE 94.53 UNDERSHOOTS by ~2%**. Honest partial: attributable to the reduced 8-epoch
  budget + NODE's slower convergence (which IS Dupont's mechanism — NODE learns slower). NOT
  tuned toward the paper. A NODE miss at fewer epochs is a legitimate partial-replication finding.
- **Faithfulness nuance (reinforces the direction):** NODE recon_ok only 2/5 at ep8 — the NODE
  flow STIFFENS past 1e-5 for 3/5 seeds (seed1 recon 0.33!), so its true faithful NFE is even
  higher than 86; ANODE stays faithful 5/5 (simpler flow). Per-seed recon logged.
- D6 (NFE-vs-loss) and D7 (train/test gap) in `figures/d3/{nfe_vs_loss,nfe_and_gap}.png`.
- NO STOP trigger: ANODE matches, NODE undershoot is a budget-attributable miss (a finding), not
  a contradiction of the claim (ANODE>NODE holds decisively) or a theorem.

---

# Overnight session — part 3 (D3 faithfulness fix → CIFAR under a cap)

## [1] D3 NODE faithful-NFE re-measurement PRE-DECLARED
The committed D3 NODE row is recon_ok 2/5 at 1e-5 -> "NODE faithful NFE 86" and the "1.7x" ratio
rest on a tol that does NOT integrate 3/5 NODE seeds. Re-measure BOTH D3 models
(`run_d3_faithful_nfe.py`, retrain deterministically = identical committed fields), per epoch
recording fwd NFE + recon at a LADDER {1e-5,1e-6,1e-7} (capped solves), 5 seeds, 8 epochs.
- **Find the loosest tol where BOTH models are recon_ok 5/5 THROUGH the budget** (or record
  honestly that the stiffest NODE seeds cannot be integrated even at 1e-7 -> that is the Dupont
  mechanism, reported as data with recon_ok=false and NFE lower-bounded).
- **Claim:** ANODE is cheaper in NFE at a tolerance where BOTH models are recon-faithful.
  **REFUTED if** the NFE gap vanishes (ANODE >= NODE) or reverses once the NODE is faithfully
  integrated. (Direction expected to hold; the point is a tol that survives the recon check.)
- Update the D3 artifact/figure + append correction to DEVIATIONS (do NOT rewrite the committed
  D3 narrative). ANODE was 5/5 at 1e-5 already; NODE is the binding column.

## [1] D3 FAITHFUL NFE RESULT (5 seeds; `results/d3_faithful/`, `figures/d3/faithful_nfe.png`)
RAW final-epoch median fwd NFE + recon_ok fraction (ladder):
| eval_tol | NODE NFE | NODE recon_ok | ANODE NFE | ANODE recon_ok |
|---|---|---|---|---|
| 1e-5 | 86 | 2/5 | 50 | 5/5 |
| 1e-6 | 146 | 3/5 | 92 | 5/5 |
| **1e-7** | **482** | **5/5** | **230** | **5/5** |
- **Loosest tol where BOTH models are recon_ok 5/5 = 1e-7.** At that common faithful tol:
  **NODE 482 vs ANODE 230 -> ANODE 2.10x cheaper.** Pre-declared claim ("ANODE cheaper at a tol
  where both recon-faithful") HOLDS; refutation (gap vanishes/reverses) NOT met. The direction is
  confirmed and STRENGTHENED vs the committed 1.7x-at-1e-5 (which understated NODE because 3/5
  NODE seeds weren't integrating at 1e-5).
- The stiffest NODE seeds DO integrate at 1e-7 (5/5) within the 500k step cap -- so no "cannot
  integrate" data point; the NODE flow is stiff but not beyond 1e-7. Submission-clean.

## [2] D4 CIFAR-10 PRE-DECLARED (under 18 GPU-h HARD CAP)
Matched params ASSERTED: NODE k=125 = 173,611 vs ANODE-p10 (aug10,k=64) = 172,452 (0.67% apart,
~0.5% above Dupont's 172,358/171,799). Batch 256. Faithful machinery = D3's.
- **Tolerance FIRST (probe before headline):** CIFAR's NODE stiffens more than MNIST's, so I probe
  timing + recon at a ladder {1e-5,1e-6,1e-7} on a few full epochs (1 seed) to (a) find the
  recon-faithful tol and (b) project the cap. The headline then measures per-epoch recon at that
  ladder so the faithful NFE and recon_ok are direct (no single-tol guess). A cell not integrable
  within the step cap is RECORDED AS DATA (recon_ok=false, NFE lower-bounded), never reported at
  loose tol as faithful.
- **Refutation:** ANODE ≥ NODE test acc at matched params WITH lower/flatter NFE, at a
  recon-faithful tol. REFUTED if ANODE acc < NODE acc (median) OR ANODE NFE ≥ NODE NFE at a tol
  where both are recon-faithful.
- **vs Dupont Table 1: NODE 53.7±0.2 / ANODE 60.6±0.4** — report raw; a MISS is a partial-
  replication finding, NOT tuned.
- **CAP DISCIPLINE:** ~18 GPU-h for the whole D4 block; the harness prints elapsed h. If a faithful
  run would blow it, STOP, commit what's recon-checked, flag — never drop to loose tol to fit.
  Seeds: ≥5 ideal; if the cap forces it, ≥3 (stated as a deviation).

## [2] D4 PROBE result + cap projection (CIFAR-10 downloaded; network-throttled ~47 min, wall-clock only)
Probe (1 seed, 2 ep, full batches, ladder): NODE ~130-148 s/ep, ANODE ~73-82 s/ep (incl. 3-tol
ladder recon). Early epochs BOTH recon_ok at all tols (1e-5 recon ~4e-4); CIFAR NODE will stiffen
with training (as D3 MNIST did) -> headline uses the ladder {1e-5,1e-6,1e-7} to capture faithful
NFE + recon_ok per epoch and report at the loosest COMMON faithful tol.
- **Cap projection: 5 seeds x 10 epochs ~= 3.5 GPU-h** (NODE ~27 min/seed, ANODE ~16 min/seed) --
  well under the 18 GPU-h cap. So **5 seeds, 10 epochs, batch 256** (no seed reduction needed).
- If a NODE cell can't integrate even at 1e-7 by late epochs (CIFAR NODE stiffer than MNIST's),
  that cell is recorded as DATA (recon_ok=false, NFE lower-bounded), NOT reported at loose tol.
- Headline config: `run_d4_anode_cifar.py --seeds 0-4 --epochs 10 --eval_tols 1e-5,1e-6,1e-7
  --batch_size 256 --cap 500000`. Launching now.

## [2] D4 headline INTERRUPTED then RESTARTED (process teardown, not a result)
The first headline launch was killed by a Claude Code process teardown after NODE seeds 0-1
(57 rows, 2/10 model-seeds) — NOT enough to conclude, and per the standing rule a partial killed
run is never patched into a result. CIFAR is now cached locally (the ~47 min network-throttled
download is done), GPU free, cap has ample room (~0.5 GPU-h consumed so far by the partial +
probes). Restarting as ONE clean invocation, fully detached (setsid) so a session boundary cannot
kill it again. Config unchanged: 5 seeds x 10 epochs x {NODE, ANODE-p10}, batch 256, ladder
{1e-5,1e-6,1e-7}, cap 500k steps.

## [3] D4 RESUMED — PRE-DECLARED (written BEFORE the run; supersedes the killed launches)
Two prior launches died (process teardown, then an unexplained death despite `setsid`). On
inspection the harness was **destructive on restart** (`traj.unlink()`), so it could never have
been completed incrementally — fixed in `2df1f0d` with resume keyed on the COMPLETE (model, seed).
Surviving data: NODE seeds 0,1,2 complete (90 rows); seed 3's 27 rows are orphans (died at epoch
9, no checkpoint → cannot resume mid-training) and are DROPPED and re-run from scratch.
- **Params ASSERTED (re-checked now):** NODE k=125 = **173,611** vs ANODE-p10 (aug10, k=64) =
  **172,452** — 0.67% apart, ~0.5–0.7% above Dupont's 172,358/171,799 (flatten-head detail, same
  as D3). Batch 256, 10 epochs, 5 seeds, train tol 1e-3, ladder {1e-5,1e-6,1e-7}, cap 500k steps.
- **Refutation (unchanged from the killed pre-declaration):** ANODE ≥ NODE test acc at matched
  params WITH lower/flatter NFE, at a tol where BOTH are recon-faithful. **REFUTED if** ANODE acc
  < NODE acc (median) OR ANODE NFE ≥ NODE NFE at a tol where both are recon_ok 5/5.
- **Reporting tol:** the loosest tol where BOTH arms are recon_ok 5/5 at the final epoch (D3's
  rule). If no tol achieves that, the cell is recorded as DATA (recon_ok=false, NFE
  lower-bounded) and reported as such — never a loose-tol number dressed as faithful.
- **vs Dupont Table 1: NODE 53.7±0.2 / ANODE 60.6±0.4** — report raw. A miss is a partial-
  replication finding; do NOT tune toward the paper.
- **Budget:** 18 GPU-h cap stands; projection ~3.5 GPU-h total, ~2.2 h remaining (2 NODE seeds
  ≈27 min each + 5 ANODE seeds ≈16 min each). **If elapsed exceeds 8 GPU-h, STOP and flag.**
- **Hardware:** local RTX 3090 for the whole arm, matching the 3 surviving NODE seeds — the D4
  comparison is deliberately NOT split across the A100 cluster. Every row now records its GPU.
- Note on the 3 surviving NODE seeds: all rows recon_ok at ALL three tolerances, i.e. CIFAR's NODE
  field had NOT stiffened past 1e-5 by epoch 10 (unlike MNIST's in D3). If that holds for the
  remaining seeds, the faithful reporting tol will be looser here than D3's 1e-7. Watch for it —
  a *convenient* result gets more scrutiny, not less.

## [3] D8 CROSSING FLOW — PRE-DECLARED re-run (was not reportable: no tol, no recon, no report script)
The committed `results/crossing/` result (NODE MSE 1.0000, ANODE-p1 0.0005) was measured at the
`ODEBlock` default **atol=rtol=1e-3** — the tolerance `DEVIATIONS.md` A4 records as
*non-integrating* for these toy fields — and the CSV carried **no tolerance column, no recon
check, and had no report/figure script**. By our own standing rule it is not reportable, however
clean it looks. Re-running with train_tol 1e-5 and an eval ladder {1e-3, 1e-5, 1e-6, 1e-7}, 5
seeds, every row carrying recon_ok + hardware. CPU (1-D toy; keeps the GPU free for D4).
- **Claim (Dupont Prop. 1 / Fig 3):** a readout-free 1-D NODE flow is order-preserving and cannot
  represent g(x): x<0→+1, x>0→−1; ANODE-p1 can.
- **Theory-predicted value:** the best order-preserving map collapses both classes toward a
  constant; with targets ±1 the optimal constant is 0, so **NODE MSE ≈ 1.0** is the floor the
  theorem predicts, not merely "bad".
- **REFUTED if** at a recon-faithful tolerance NODE MSE < 0.1 (the NODE solves it), **OR**
  ANODE-p1 MSE > 0.1 (augmentation fails to help).
- **STOP-and-flag:** NODE MSE materially below ~1.0 (say < 0.5) at a tol where recon_ok=1 would
  contradict Proposition 1 — a theorem, not just a paper number. Do NOT resolve alone; leave the
  raw numbers and flag.
- Reporting tol: loosest tol where BOTH arms are recon_ok 5/5, same rule as D3/D4. The 1e-3 rung
  is included deliberately so the old, non-integrating number stays visible as data.

## [3] D8 RESULT (5 seeds, ladder, recon-checked; `results/crossing/`, `figures/crossing/`)
RAW median over 5 seeds (MSE | fwd NFE | recon_ok fraction):
| eval_tol | NODE MSE | NODE NFE | NODE recon | ANODE-p1 MSE | ANODE NFE | ANODE recon |
|---|---|---|---|---|---|---|
| 1e-3 | 1.0004 | 56 | **0/5** | 0.0007 | 26 | **0/5** |
| 1e-5 | 1.0002 | 92 | **0/5** | 0.0007 | 50 | 5/5 |
| 1e-6 | 1.0002 | 158 | **3/5** | 0.0007 | 128 | 5/5 |
| **1e-7** | **1.0002** | **308** | **5/5** | **0.0007** | **272** | **5/5** |
- **Loosest tol where BOTH arms are recon-faithful: 1e-7.** There: NODE MSE **1.0002**
  [1.0000, 1.0003], ANODE-p1 **0.0007** [0.0001, 0.0013]. NFE 308 vs 272.
- **Pre-declared refutation NOT met** (NODE 1.0002 ≥ 0.1; ANODE 0.0007 ≤ 0.1). No STOP trigger:
  NODE sits *at* the Proposition-1 floor (1.0002 vs the predicted 1.0), never below it.
- **The old committed number was measured in a non-integrating regime.** At 1e-3 the recon check
  fails for **all 10 model-seeds**; the worst NODE seed reconstructs with **1.21e+01** relative
  error (≈1200%). So the previously committed D8 row was not integrating the field at all.
- **Scrutiny note (the conclusion was right for a reason we had not checked):** NODE MSE is
  1.0002–1.0006 at *every* rung, so the D8 conclusion happens to be tolerance-robust — but it had
  never been *established* faithfully, and robustness is a finding, not an assumption. The ANODE
  arm is the one that moves: it is recon-faithful from 1e-5 while the NODE arm needs 1e-7, i.e.
  the NODE's failing flow is also the stiffer one to integrate.

## [2] C1 SOLVER DYNAMICS — PRE-DECLARED (the genuinely missing experiment)
Canonical C1 (`REPLICATION_PLAN.md` §3.B: Chen Fig 3a-b — numerical error ↓ and cost ↑ as
tolerance tightens, forward time ∝ NFE) had **no committed CSV at all**. The overnight "C1/C3"
label covered C3 (NFE-over-training) only; this was masked by the naming. Building it now:
`scripts/run_c1_solver_dynamics.py`, ≥5 seeds, on the trained MNIST conv ODE-Net.
- **Design:** ONE trained model per seed (5 epochs at train tol 1e-3), then a frozen
  **evaluation** sweep — adaptive {bosh3, dopri5, dopri8} × tol {1e-1…1e-7} and fixed-step
  {euler, midpoint, rk4} × steps {1…128}. Fixed-step solvers are not tolerance-controlled, so
  their analogous cost axis is the step count; both give an accuracy-vs-cost curve.
- **Error is measured against a high-accuracy REFERENCE endpoint** (dopri8 @ 1e-10, ≥2 orders
  tighter than the tightest swept tol) on a fixed test feature batch — so "numerical error" is
  the *integrator's*, not the classifier's. Wall-clock is the solve alone, CUDA-synchronised,
  median of 5 repeats. Forward AND backward (adjoint) NFE recorded. Every row carries recon_ok.
- **R-C1a (error falls):** rel_err decreases monotonically as tol tightens, per adaptive solver.
  **REFUTED if** rel_err is flat or increases across the range (above the reference floor).
- **R-C1b (cost rises):** fwd NFE and wall-clock increase monotonically as tol tightens.
  **REFUTED if** NFE is flat or decreasing.
- **R-C1c (time ∝ NFE — Chen Fig 3b):** forward time and fwd NFE are strongly rank-correlated.
  **REFUTED if** Spearman ρ < 0.9 pooled within a solver.
- **STOP-and-flag:** a *higher-order* solver showing systematically LARGER error than a
  lower-order one at matched NFE would contradict standard numerical analysis, not just Chen.
- Rows failing recon are DATA (loose-tol rows are expected to fail; that is the point of the
  axis) and are excluded from the faithful-cost claims, never deleted.
- **Hardware:** A100 (Bocconi HPC) for the whole C1 block — a NEW experiment with no existing
  rows, so it is internally consistent; D4 stays on the local 3090. Every row records its GPU.

## [2] C1 REFERENCE-COST BENCHMARK (design probe, measured not guessed) — a stiffness finding
Two C1 attempts stalled inside the high-accuracy reference solve (>19 min, no output), so I
benchmarked it instead of guessing a third time (`--ref_bench`, 5-epoch-trained conv field,
batch 32, dopri8):
| ref tol | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 |
|---|---|---|---|---|---|
| NFE | 925 | 7,074 | 44,423 | 207,196 | (>20 min, cancelled) |
| wall-clock | 1.0 s | 8.1 s | 51.0 s | 237.1 s | — |
- **Growth is ≈5–7× per decade of tolerance.** For an 8th-order explicit method on a non-stiff
  field the cost should scale ≈ tol^(-1/8), i.e. **~1.33× per decade**. Observing 5–7× is a
  strong **stiffness** signature: dopri8 is an explicit method and the trained MNIST conv field
  is stiff (consistent with C3, where faithful NFE grew 384→738 over 6 epochs). Recording this as
  an observation; it is *not* the pre-declared C1 claim and is not being fitted to.
- **Design decision (from the measurement):** reference = **dopri8 @ 1e-8** (207k NFE, ~4 min/seed
  — affordable), sweep tolerances **{1e-1 … 1e-5}**, eval batch 32. That is **exactly Chen's Fig
  3a-b tolerance range (1e-0…1e-5)** with **3 orders of headroom** between the tightest swept
  tolerance and the reference, so the error axis is never reference-limited.
- Trained models are now cached per seed, so this design iteration cost one training run, not four.

## [3] D4 RESULT — CIFAR-10 matched-param NODE vs ANODE (5 seeds, COMPLETE; `results/d4/`)
Finished in **1.85 GPU-h** (cap 18 h; stop-flag 8 h — neither approached). 300 rows = 2 models ×
5 seeds × 10 epochs × 3 tols, **all on one RTX 3090**, resumed over the 3 surviving NODE seeds.
RAW final epoch (median over 5 seeds):
| eval_tol | NODE acc | NODE NFE | NODE recon | ANODE-p10 acc | ANODE NFE | ANODE recon |
|---|---|---|---|---|---|---|
| 1e-5 | 53.52 | 62 | 5/5 | 59.21 | 50 | **4/5** |
| **1e-6** | 53.52 | **122** | 5/5 | 59.21 | **98** | 5/5 |
| 1e-7 | 53.52 | 410 | 5/5 | 59.21 | 212 | 5/5 |
- **Loosest tol where BOTH arms are recon-faithful: 1e-6** → NODE NFE 122 vs ANODE 98,
  **ANODE 1.24× cheaper**. (At 1e-7: 410 vs 212 = 1.93×.)
- **Pre-declared refutation NOT met:** ANODE acc 59.21 ≥ NODE 53.52, AND ANODE NFE < NODE NFE at a
  tol where both are recon_ok 5/5.
- **vs Dupont Table 1 (raw, untuned):** NODE **53.59 ± 0.50** vs Dupont **53.7 ± 0.2** — matches
  within noise. ANODE **59.34 ± 0.70** vs Dupont **60.6 ± 0.4** — **~1.3 pp low**, just outside
  combined spread → a **partial** replication on the ANODE number, reported as such.
  Per-seed NODE [53.52, 54.48, 53.01, 53.25, 53.71]; ANODE [60.61, 58.80, 59.21, 58.61, 59.49].
- **SCRUTINY (the NODE number is suspiciously close, so it gets more, not less):**
  **`test_acc` is measured at the TRAIN tolerance 1e-3, and the recon ladder starts at 1e-5 — so
  the reconstruction check has NEVER been run at the tolerance the accuracy is measured at.**
  Since recon degrades as tolerance loosens (ANODE is already 4/5 at 1e-5), the accuracy numbers
  are plausibly measured in a regime that would fail the check. The **NFE** comparison is faithful
  (recon-checked at 1e-6); the **accuracy** comparison is *not yet* established as faithful.
  **This is inherited from D3's design and applies to D3's MNIST accuracies identically.**
  It does not change the refutation outcome (the ANODE>NODE accuracy gap is 5.7 pp, far larger
  than any plausible integration artifact), but it is a real gap in our own discipline and must
  not be reported as if recon-checked. FLAGGED for decision — the fix is a re-run with a 1e-3
  recon rung + accuracy re-measured across the ladder (D4 ≈1.9 GPU-h, D3 cheaper); the harnesses
  should also cache checkpoints (as C1 now does) so this never costs a full retrain again.
- Inversion vs D3 worth noting: on CIFAR the **ANODE** arm is the one that fails recon at 1e-5
  (4/5) while NODE is 5/5 at every rung — the opposite of D3/MNIST, where NODE was 2/5 at 1e-5.

## [2] C1 RESULT — solver dynamics, 5 seeds (`results/c1/`, `figures/c1/solver_dynamics.png`)
5 seeds × (3 adaptive × 5 tols + 3 fixed-step × 8 step counts) = 195 rows, A100s, reference
dopri8 @ 1e-8. RAW medians (adaptive):
| solver | tol | rel_err | fwd NFE | bwd NFE | time ms | recon |
|---|---|---|---|---|---|---|
| bosh3 | 1e-1 → 1e-5 | 3.47e-1 → 1.71e-4 | 20 → 227 | 47 → 10,166 | 22.7 → 279 | 0/5 → 5/5 |
| dopri5 | 1e-1 → 1e-5 | 2.08e-1 → 2.18e-4 | **26 → 20** → 38 → 68 → 116 | 74 → 3,248 | 27.4 → 127 | 0/5 → **3/5** |
| dopri8 | 1e-1 → 1e-5 | 6.01e-3 → 3.49e-5 | 41 → 1,055 | 470 → 180,520 | 54.8 → 1,254 | 0/5 → 5/5 |
- **R-C1a HOLDS** — error falls monotonically for all three adaptive solvers.
- **R-C1b REFUTED, as pre-declared.** dopri5 NFE goes **26 → 20** from tol 1e-1 to 1e-2. The
  pre-declared condition was "REFUTED if NFE is flat or decreasing", and it decreases. **Reporting
  the refutation as it stands — not restating the claim to make it pass.** Where it lives: the
  single violating step is dopri5's two loosest rungs, **both recon_ok 0/5** (the flow is not
  being integrated there at all, so the adaptive controller's step choice is not meaningful).
  Restricted to recon-faithful rows the cost is monotone (bosh3 112→227, dopri8 379→1,055), but
  dopri5 has only ONE faithful rung so monotonicity cannot be tested for it. The honest statement
  is: **Chen's cost-rises claim holds where the solver is actually integrating, and is violated
  at loose tolerance where it is not.**
- **R-C1c HOLDS** — Spearman ρ(time, NFE) = 0.983–0.992 pooled per solver.
- **Hardware confound CAUGHT by the per-row `hardware` column** (this is why it is recorded): the
  SLURM array scattered seeds across **two different A100 MIG slice sizes** — 4g.40gb (seeds 0,1,4)
  and 3g.40gb (seeds 2,3) — so pooled wall-clock mixes machines. Re-checked **within** each
  hardware group, ρ = 0.991–0.997, i.e. *tighter* than pooled. The confound weakened the
  conclusion rather than creating it; R-C1c survives. Now encoded as an automatic check in
  `scripts/c1_report.py` rather than left as a note.
- **STOP-check clean:** no higher-order solver is systematically worse at matched NFE.
- **Nominal tolerance is NOT comparable across solvers** (worth a paper sentence): at tol 1e-5,
  dopri5 spends 116 NFE and lands at recon_rel 9.8e-3 (3/5 pass, right at the 1e-2 threshold),
  while bosh3 spends 227 NFE for 1.1e-3 and dopri8 spends 1,055 for 1.6e-3. The same *requested*
  tolerance buys very different *actual* integration accuracy. `recon_ok` is the comparable axis;
  the tolerance number is not.
- Fixed-step arm behaves exactly as numerical analysis predicts: NFE = steps × stages (euler 1×,
  midpoint 2×, rk4 4×) and error falls at the method's order (rk4 7.1e-1 → 7.3e-6 over 1→128 steps).

## [4] D3 + D4 ACCURACY-FAITHFULNESS RE-RUN — PRE-DECLARED (closes the flag raised at D4)
The D4 result flagged that **`test_acc` is measured at the TRAIN tolerance 1e-3 while the recon
ladder started at 1e-5**, so the reconstruction check had never been run at the tolerance the
accuracy is measured at. Same defect in D3. Closing it by measurement, not argument.
- **Change:** the ladder now **includes the 1e-3 train rung** (so the accuracy tolerance is itself
  recon-checked every epoch), and at the **final epoch** the full test set is re-evaluated **at
  every rung** (`test_acc_at_tol`). Final epoch only — a full CIFAR test pass at 1e-7 costs ~20×
  one at 1e-3, and the headline number lives at the final epoch.
- **Ladder:** {1e-3, 1e-5, 1e-6, 1e-7}. D4: 5 seeds × 10 epochs. D3: 5 seeds × 8 epochs (matching
  the committed runs so the numbers are comparable).
- **What is under test:** whether the reported accuracies are artifacts of a non-integrating
  tolerance.
  - **R-ACC1 (accuracy is tolerance-robust):** at the final epoch, |acc(1e-3) − acc(faithful tol)|
    < 0.5 pp for both arms. **REFUTED if** the accuracy moves ≥0.5 pp between the train tolerance
    and the loosest recon-faithful tolerance — which would mean the committed D3/D4 accuracies
    were measured in a regime that was not integrating, and **both must be restated**.
  - **R-ACC2 (the ANODE>NODE gap survives):** the ANODE−NODE accuracy gap keeps its sign and stays
    >1 pp at a recon-faithful tolerance. **REFUTED if** the gap closes or inverts.
- **Prediction on record (so a miss is visible):** I expect R-ACC1 to HOLD — the D4 NFE at 1e-3 is
  ~20 vs ~410 at 1e-7, but a classifier argmax is far less sensitive than an NFE count. If it does
  NOT hold, the D3/D4 accuracy claims are retracted, not patched.
- **STOP-and-flag:** if the accuracy at a recon-faithful tol moves the NODE row AWAY from Dupont's
  53.7 by more than the 0.5 pp band, say so — the current agreement would then be a loose-tolerance
  coincidence, which is a finding about *our* method, not about Dupont.
- Hardware: A100 (HPC), per-seed array shards (`--tag`), every row records its GPU. The existing
  `results/d3/d3_trajectory.csv` and `results/d4/d4_trajectory.csv` are NOT overwritten — shards
  land beside them and are merged only after the checks above are evaluated.

## [4] PACKAGING — clean-room container verification (the check the last audit could not make)
The previous container check reused cached Docker layers and a surviving conda env, so it
proved "the existing environment works", not "a reviewer can build it". Done properly now:
- `docker build --no-cache` from scratch: **succeeds** (exit 0). The pinned stack resolves today;
  note the build pulls numpy 2.4.6 as a torch dependency and then correctly downgrades it to the
  pinned **2.4.3**, so the pin is doing real work.
- `docker run` (acceptance gate, `make smoke`) in the fresh image: **57 passed, 1 skipped**, exit 0.
- `docker run ... make figures` in the fresh image: **all 18 figures rebuilt from the committed
  CSVs**, exit 0. This is the reviewer's actual path — clone, build, regenerate every figure —
  and it now works end to end with no GPU, no training and no accounts.
- Determinism spot-check: `results/c2/c2_surface.csv` regenerates **byte-identically**, so the
  figure pipeline does not perturb committed artifacts.

## [4] D3 + D4 ACCURACY-FAITHFULNESS RESULT (5 seeds each, A100; shards in `results/d3/`, `results/d4/`)
All 10 array tasks COMPLETED exit 0 (D4 47–60 min/seed, D3 19–25 min/seed); row counts exact
(D4 2×10×4 = 80/seed, D3 2×8×4 = 64/seed). Internal check: at the 1e-3 rung `test_acc_at_tol`
re-computes `test_acc` — max difference **0.0**. RAW final epoch, accuracy RE-MEASURED at each rung
(mean ± sd, ddof=1):
| | tol | NODE recon | NODE acc | NODE NFE | ANODE recon | ANODE acc | ANODE NFE |
|---|---|---|---|---|---|---|---|
| D4 | 1e-3 (train) | **0/5** | 53.68 ± 0.85 | 26 | **0/5** | 60.15 ± 0.91 | 20 |
| D4 | **1e-5** | 5/5 | **53.69 ± 0.81** | 62 | 5/5 | **60.04 ± 0.94** | 50 |
| D4 | 1e-6 / 1e-7 | 5/5 | 53.70 / 53.69 | 116 / 272 | 5/5 | 60.04 / 60.05 | 92 / 206 |
| D3 | 1e-3 (train) | **0/5** | 94.18 ± 0.39 | 38 | **0/5** | 98.05 ± 0.21 | 26 |
| D3 | 1e-5 | 3/5 | 94.16 | 68 | 4/5 | 98.05 | 50 |
| D3 | **1e-6** | 5/5 | **94.16 ± 0.44** | 152 | 5/5 | **98.05 ± 0.19** | 92 |
| D3 | 1e-7 | 5/5 | 94.16 | 428 | 5/5 | 98.05 | 224 |
- **The flag was real:** the 1e-3 train tolerance fails recon on **0/5 seeds in all four arm ×
  experiment cells**. Every previously committed D3/D4 accuracy was measured in a non-integrating
  regime.
- **R-ACC1 HOLDS** (prediction on record held): accuracy at the loosest faithful tol differs from
  accuracy at 1e-3 by D4 NODE +0.012 / ANODE −0.108 pp, D3 NODE −0.024 / ANODE −0.008 pp; worst
  single seed 0.37 pp. Accuracy is robust to integration error; NFE is not. Same shape as D8:
  the answer was right, but it had not been established.
- **R-ACC2 HOLDS:** ANODE − NODE gap at the faithful tol = **+6.35 pp** (D4), **+3.89 pp** (D3).
  Pre-declared refutation not met in either: ANODE ≥ NODE accuracy and ANODE < NODE NFE.
- **vs Dupont (at the faithful tol):** D4 NODE 53.69 vs 53.7 (−0.01); D4 ANODE 60.04 vs 60.6
  (−0.56); D3 NODE 94.16 vs 96.4 (−2.24); D3 ANODE 98.05 vs 98.2 (−0.15). No STOP trigger: the
  D4 NODE agreement did not move away from Dupont — it is not a loose-tolerance coincidence.
- **Scrutiny — D4 ANODE moved +0.70 pp between runs** (run 1 RTX 3090: 59.34 ± 0.79; run 2 A100:
  60.04 ± 0.94), toward Dupont. Difference ≈1.4σ of the difference of two 5-seed means: run-to-run
  spread (seeds are not bit-reproducible across GPUs), not an improvement. **Not** upgrading
  "partial" to "match" on a favourable re-run: reported as a 0.6–1.3 pp undershoot across two runs.
- **Scrutiny — NFE ratios are less stable than accuracies.** D4 NODE NFE at 1e-7 was 410 in run 1
  and 272 in run 2; the faithful tol itself moved (run 1: 1e-6, run 2: 1e-5 for D4; D3 run 1's
  separate faithful re-measurement: 1e-7, run 2: 1e-6). ANODE is cheaper at **every** faithful rung
  in **both** runs, but by **1.24–1.93× (D4)** and **1.65–2.10× (D3)**. Reported as ranges.
- **Reporting slip found and corrected:** the committed D3 numbers 98.18 ± 0.29 / 94.53 ± 0.44 were
  run-1 **medians** printed next to a standard deviation; run-1 means are 98.00 / 94.32. All image
  accuracies are now mean ± sd (ddof=1), Dupont's convention. No conclusion changes.
- **Decision — run 2 is canonical, run 1 kept as a replicate** (`results/d{3,4}/run1_rtx3090/`,
  printed by the reports). Reason is methodological: run 2 is the only run whose accuracy is
  measured at a recon-checked tolerance. It is NOT chosen because it lands closer to Dupont.
- Hardware: both arrays again scattered across A100 MIG 3g.40gb and 4g.40gb slices. Irrelevant to
  accuracy and NFE; `epoch_s` is not comparable across rows and is not used for any claim.
- **Process note (near-miss):** the first submission queued jobs on a stale cluster checkout —
  `git pull` aborted on untracked C1 shards, SLURM accepted the jobs anyway. Caught from the same
  output and resubmitted at the right commit. A successful `sbatch` says nothing about which commit
  was queued; always check `git log -1` on the cluster before submitting.

## [5] §6 MISSING-SLICE GRID — PRE-DECLARED (written BEFORE the run; REPLICATION_PLAN P2-18)
Fig 9 shows ONE wedge on ONE geometry. §6 makes it systematic. `scripts/run_slice_grid.py`:
augmentation **p ∈ {0,1,2,3,5}** × wedge width **w ∈ {π/8, π/5, π/3}** × geometry **{spheres,
circles}** × **10 seeds** = 300 from-scratch cells, 100 epochs, train/eval tol 1e-6 + recon check
(each cell is D2's exact harness). Metrics: held-out-slice acc + loss, plus the §6-requested
**observed-region-only** validation (full val minus the wedge). Heavy-tailed, so **median [IQR]**.
Cells failing recon are excluded from the statistics and counted.
- **Second geometry = circles, not moons/spirals.** The standing rule is *no new datasets*; circles
  is already in the repo (D1, and D2's original design). Moons/spirals from §6 are deliberately not used.
- **G1 (Fig 9 at every width, spheres):** every ANODE (p≥1) median slice acc ≥ NODE's at every w.
  **REFUTED if** any p≥1 is below NODE at any w.
- **G2 (monotone in the size of the unobserved region — the §6 claim, spheres):** the advantage
  A_p(w) = med slice acc(ANODE-p) − med slice acc(NODE) is non-decreasing in w for the majority of
  p. **REFUTED if** not.
- **G3 (survives a second geometry):** on circles, A_p > 0.05 at the widest wedge for the majority
  of p. **REFUTED if** not.
- **STOP-and-flag:** any ANODE worse than NODE on BOTH median slice loss AND acc at any width on
  **either** geometry. Scope matches D2's S3, which was pre-declared over {circles, spheres}; it is
  deliberately NOT narrowed to spheres after the probe below. Do not resolve alone.
- **Consistency check:** the grid's (spheres, π/5, NODE / ANODE-p1, seeds 0–4) cells run D2's
  identical code and config, so they must reproduce the committed D2 result up to CPU float order.
- **Probe SEEN before this entry (1 seed, w = π/3, 4 cells — NOT a result):** spheres NODE slice acc
  0.578 vs ANODE-p5 1.000; **circles NODE 0.998 vs ANODE-p5 0.950** — no NODE gap, and ANODE-p5
  slightly *worse*. Recorded because it was observed before the checks were written; the checks
  are §6's, unchanged by it. **Prediction on record:** G1 holds; G2 likely holds; **G3 likely
  REFUTED**; the circles inversion may or may not survive 10 seeds — if it does on both loss and
  accuracy, the STOP fires and it goes to the co-author, not to me.
- **Budget:** 2.1–2.7 min/cell single-threaded → ~55 min on 14 workers (local Ryzen 7 7700X; the
  2-D toy is faster on CPU). **If wall-clock exceeds 3 h, stop and flag.** Every row records its CPU.

## [5] §6 MISSING-SLICE GRID — RESULT (300 cells; `results/slice_grid/`, `figures/slice_grid/`)
300/300 cells written, 0 failed, **1.64 h** (pre-declared ~55 min — the probe ran 4 workers; 14
workers share 8 physical cores, so cells took ~4 min instead of ~2.5; inside the 3 h stop-flag).
6/300 cells fail recon (≤1 per group; 5 of 6 marginal at recon 0.012–0.016 vs 0.01) and are
excluded. RAW held-out-slice accuracy, median [IQR] over 10 seeds:
| | π/8 | π/5 | π/3 |
|---|---|---|---|
| spheres NODE | 0.926 [0.841,1.000] | 0.744 [0.615,0.980] | 0.701 [0.549,0.998] |
| spheres ANODE p1/p2/p3/p5 | 1.000 all | 1.000 all | 0.972 / **1.000** / 0.986 / 0.983 |
| circles NODE | 0.923 [0.669,1.000] | 0.812 [0.651,0.997] | 0.854 [0.623,0.998] |
| circles ANODE p1/p2/p3/p5 | 1.000 all | 0.993 / 1.000 / 1.000 / 0.998 | 0.992 / **0.998** / 0.941 / 0.968 |
- **G1 HOLDS** — every ANODE ≥ NODE (median slice acc) at every width on spheres.
- **G2 HOLDS as declared** — A_p(w) non-decreasing for 4/4 p (+0.074 → +0.256 → +0.27–0.30).
  **But weaker than "monotone" suggests:** the π/5 → π/3 step is inside NODE's IQR, and on
  **loss** the advantage *shrinks* there (NODE 3.14 → 2.22 while ANODE rises 0.000 → 0.001–0.13).
  Honest reading: grows from π/8 to π/5, then plateaus. ANODE itself degrades on the widest hole.
- **G3 HOLDS — my prediction ("G3 likely REFUTED") was WRONG.** Circles advantage at π/3 is
  +0.087 to +0.144 for all p. The one-seed probe (circles NODE 0.998 vs ANODE-p5 0.950) was an
  outlier from NODE's heavy-tailed distribution — exactly why §6 asked for 10 seeds. Circles is
  **not** monotone in width (NODE 0.923 → 0.812 → 0.854); G2 was declared for spheres only, so this
  is a characterisation, not a refutation.
- **STOP: none.** No ANODE is worse than NODE on both median loss and accuracy anywhere.
- **Observed-region accuracy ≥ 0.998 for every model, NODE included** — NODE's failure is specific
  to extrapolation into the unobserved wedge, not general underfitting (the §6 metric doing its job).
- **Augmentation dose (characterisation; bears on DEVIATIONS A12):** p = 2 is best at the widest
  wedge on both geometries; p = 3/5 are slightly worse (circles p3 0.941 [0.842,0.981] vs p2 0.998
  [0.996,1.000] is the one clear separation). More augmentation is not monotonically better for
  held-out generalisation; Dupont chose p = 5 for fit, a different criterion.
- **Consistency: EXACT.** The grid's (spheres, π/5, NODE/ANODE-p1, seeds 0–4) cells reproduce the
  committed D2 held-out accuracies with max difference **0.0000** over 10 matched cells — fresh
  runs, two months later, single-threaded instead of multi-threaded, same machine.

## [6] C2 DIAGNOSIS — PRE-DECLARED (why does Chen Fig. 3c not reproduce?)
Claim 6 (backward NFE ≈ ½ forward NFE) is the one claim we report as NOT reproduced. Instead of
leaving it at "we measured 13–121×", we diagnose the cause. `scripts/run_c2_diagnosis.py`:
solver × forward tolerance × **adjoint tolerance** × field × seeds, every cell carrying the
reconstruction check AND a gradient-correctness check against direct backprop at 1e-9.
- **Why gradient correctness is in the design:** a configuration that is cheap because it solves
  the adjoint badly is not evidence about cost. Cells with grad_reldiff ≥ 1e-2 are recorded and
  excluded from any conclusion, never silently counted as cheap.
- **H1 (solver family):** Chen used implicit Adams (adaptive, stiff-capable); we used dopri5
  (explicit RK), and the adjoint's augmented reverse system is stiffer than the forward one.
  Claim: a stiff-capable adaptive solver gives a ratio ≥5× smaller than dopri5 at matched
  tolerance. **REFUTED if** its ratio is within 2× of dopri5's, or larger.
  (torchdiffeq's `implicit_adams` is FIXED-step and so is not a fair proxy; `scipy_solver` with
  LSODA/BDF is the closest adaptive stiff-capable analogue and stands in for Chen's solver.)
- **H2 (adjoint tolerance):** we solved the reverse system at the SAME tolerance as the forward
  pass. Claim: loosening the adjoint tolerance 100× relative to the forward reduces the ratio by
  ≥5×. **REFUTED if** <2×.
- **H3 (field stiffness):** the ratio grows with training, for every solver. **REFUTED if**
  untrained ≈ trained.
- **REVISION TRIGGER (can overturn our own negative result):** if any configuration faithful to
  Chen's description reaches ratio ≤1 at a tolerance that is BOTH recon-faithful and
  gradient-correct, then "Claim 6 does not reproduce" is WRONG as stated, and the paper must say
  the ratio is a property of the adjoint solver *configuration* — not that Chen's claim fails.
- **Probe SEEN before this entry (1 seed, spheres, 100 ep, tol 1e-5 — NOT a result):**
  dopri5 same-tol **24.4**; dopri5 adjoint ×100 looser **1.60**; scipy:LSODA same-tol **1988**;
  scipy:LSODA ×100 **3.79**; scipy:BDF **19.4 → 14.4**. All grad_ok. Recorded because it was seen
  before the hypotheses were written; the hypotheses are unchanged by it.
  **Prediction on record: H1 REFUTED (the stiff solver is far worse, not better), H2 SUPPORTED,
  and the revision trigger likely FIRES** — in which case our Claim 6 reporting changes from
  "not reproduced" to "reproduced only when the adjoint is solved at its own, looser tolerance",
  which is a scope condition on Chen's claim rather than a contradiction of it.
- Stage A: toy spheres, 7 solvers × 3 tols × 2 adjoint offsets × 5 seeds × {trained, untrained},
  CPU, sharded per seed. Stage B: MNIST conv field on the A100s, reduced grid, reusing the C1
  checkpoints. Per-cell 15 min timeout — a runaway cell is recorded as capped, not a hung job.

## [6] C2 DIAGNOSIS RESULT (456 cells; `results/c2_diagnosis/`, `figures/c2_diagnosis/`)
Toy spheres 420 cells (5 seeds × {trained, untrained} × 7 solvers × 3 tols × 2 adjoint offsets,
CPU, 29.5 min/seed) + MNIST conv 36 cells (3 seeds, A100, reusing the C1 checkpoints). 18 cells
hit the 15-min cap; of the remaining 438, **316 pass BOTH the reconstruction and the gradient
check** and carry the conclusions. Excluded cells are concentrated at tol 1e-3 (96 toy, 12 MNIST).
- **H1 (solver family) REFUTED — and my prediction was right for once.** Stiff-capable adaptive
  solvers are not cheaper: on the trained toy field at 1e-5, `scipy:LSODA` gives **2158.8** and
  `scipy:BDF` **28.0** against `dopri5`'s **24.6**; at 1e-7, 384.3 and 32.4 against 95.9. On the
  MNIST conv field **no SciPy cell completed within the cap at all**. Chen's implicit-Adams
  family does not explain our gap; if anything it is worse here.
- **H2 (adjoint tolerance) HOLDS, 15/17 solver × tolerance combinations.** Loosening the ADJOINT
  tolerance 100× relative to the forward pass cuts the ratio by 5.2×–76×: MNIST `dopri5`
  **123.3 → 4.06** (30×), `bosh3` **203.4 → 5.64** (36×); toy `dopri5` 24.6 → 1.96, `bosh3`
  43.5 → 1.75. The two exceptions are the SciPy solvers. **This is the dominant cause.**
- **H3 (ratio grows with training) REFUTED, 11/16.** `adaptive_heun`, `bosh3` and `dopri8` show
  the UNTRAINED field with the *higher* ratio. **This corrects our own earlier committed
  statement** that the bwd/fwd ratio is training-driven ("a random field is cheap; a trained
  separating field is not"). The likely reason is that the ratio conflates two quantities: on an
  easy field the *forward* NFE is small, so a modest backward cost still yields a large ratio.
  The ratio is a poor summary statistic, which is itself worth saying in the paper.
- **`torchdiffeq` defaults `adjoint_rtol = rtol`** (verified in the source): the expensive regime
  we originally reported is the library's *default*, not a choice we made. That matters for how
  the finding should be phrased.
- **REVISION TRIGGER — DOES NOT FIRE as pre-declared.** With the adjoint at the forward tolerance
  on a trained field, the lowest checked ratio is **5.96** (toy) and **118.9** (the conv field
  Chen used). Unrestricted, a checked configuration does reach **0.78** (`adaptive_heun`, toy,
  1e-5, adjoint ×100) — but that is neither Chen's solver nor his coupling, so it does not meet
  the condition as written. **Our "claim 6 does not reproduce" therefore stands**, now with a
  cause rather than just a number.
- **Limitation, stated plainly:** we could not evaluate Chen's actual solver. torchdiffeq's
  `implicit_adams` is fixed-step (so its NFE ratio is set by the step count, not by stiffness),
  and the adaptive stiff-capable SciPy solvers time out on the conv field. So we can say the gap
  is dominated by the adjoint tolerance coupling and is *not* explained by moving to a
  stiff-capable solver — but not that Chen's exact configuration would behave as ours does.
