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
