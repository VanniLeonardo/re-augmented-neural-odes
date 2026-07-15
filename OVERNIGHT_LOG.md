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
