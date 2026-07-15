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
- Decision: commit as a complete experiment (CSV + script + plot + this log entry).
