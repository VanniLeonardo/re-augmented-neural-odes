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
