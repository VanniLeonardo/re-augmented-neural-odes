# DEVIATIONS — where our implementation differs from the papers *as described*

`PROVENANCE.md` answers a different question ("did we copy the authors' code?" — no).
**This file answers "did we implement the architecture/setup the paper describes?"** and
records every place we knowingly differ, so the paper's "deviations from the original"
section can be written from it. This is a first-class ReScience deliverable.

Each row: **what the paper specifies · what we do · why · expected effect on the reproduced
claim · status** (`RESOLVED` = fixed to match the paper; `OPEN` = still differs, to fix or
justify in Stage C; `INTENDED` = a deliberate difference that IS the finding).

Legend for claim references: D* = Dupont experiments, C* = Chen claims (see `REPLICATION_PLAN.md` §3).

---

## A. Dupont ANODE — toy / concentric separation (D1, D2, D8, §6 grid)

| # | Paper specifies (cite) | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| A1 | **ODE integrates in data space**: the toy field is an MLP acting on the raw input; the homeomorphism argument (a NODE cannot separate nested regions) requires no learned warp before the flow (Sec 3–4). | **Was:** a `Linear(2,2)+Tanh` downsampling stem before the ODE. **Now:** no stem — ODE integrates the raw 2-D input (`use_stem=False`). | The stem is a diffeomorphism that hands the model a free learned warp before the flow; the faithful setup has none. | **Code change RESOLVED** (data-space integration is faithful and is what the budget sweep uses). **The earlier factorial verdict ("no consistent effect / NFE flat") is WITHDRAWN**: it was measured at `atol=rtol=1e-3`, which does not integrate these fields (fwd→bwd reconstruction error ≈0.9 — see A4 and the Finding note). Whether the stem changes the *NFE cost* is re-opened and not yet re-measured at an accurate tolerance. | **RESOLVED** (code); **OPEN** (measurement). |
| A2 | Toy vector-field width **32** (`d+1→32→32→d`, App. F.1.1). | **Was:** `ode_hidden_dim=64`. **Now:** 32. | Coursework widened the field "for capacity"; a wider field lets the NODE better approximate the distorted flow, again flattering the NODE baseline. 32 matches the paper. | Narrower field → NODE slightly less able to fake the tearing → larger NODE-vs-ANODE gap. | **RESOLVED** |
| A3 | Dataset: **concentric spheres** — filled inner disk `‖x‖≤0.5` + annulus `1.0≤‖x‖≤1.5`; **1000 inner + 2000 outer** (App. F.2.1). | **Was:** two thin **circles** only. **Now:** `make_spheres` added (`data/synthetic.py`, registered as `"spheres"`); `make_circles` kept for the coursework comparison. Geometry, radii and 1:2 ratio match Dupont exactly. | The paper specifies this geometry; the coursework used thin circles. | **The earlier "geometry not harder" verdict is WITHDRAWN** (loose-tol, accuracy-only reading). At an accurate tolerance the spheres task reproduces Dupont's d=2 result: the NODE approximates the finite sample but via an increasingly **stiff** flow whose **NFE grows during training** (median 219→465 over 25→1000 ep; Fig 6 analog `figures/budget/nfe_vs_epoch.png`). iid accuracy was the wrong lens; the cost (NFE) is the signature. | **RESOLVED** (implemented; measured at accurate tol — see Finding note). |
| A4 | Solver **RK45 (dopri5)**; torchdiffeq's own defaults are **rtol = 1e-7, atol = 1e-9** (Dupont uses the library defaults). | **Was:** `atol = rtol = 1e-3` (the `ODENet` default). **Faithful experiments now use an accurate tolerance + a reconstruction check.** | 1e-3 is **~4–6 orders looser** than the library default. For the near-singular fields this task induces it is **non-integrating**: fwd→bwd reconstruction error ≈0.9 on data of radius ~1, so accuracy/NFE measured there are artifacts (this is the whole falsification story). | Loose tol *hides* the very NFE-growth Dupont reports and produces a flow that is not being integrated. **Every NFE/accuracy number in the submission must carry a tolerance sweep or a passing reconstruction check** (`tests/test_flow_faithfulness.py`). | **RESOLVED-as-discipline** (tolerance is now a first-class axis; recon check is a permanent test). |
| A5 | **50 epochs**, **20 repeats** (App. F.2.1). | Budget is now a **swept axis** {25,50,100,200,500,1000} × **5 seeds** (`scripts/run_budget_sweep.py`), so Dupont's 50-epoch point is measured directly rather than fixed. | Makes the budget-dependence explicit instead of a single choice. | At Dupont's 50 ep (accurate tol): NODE dense acc 0.995±0.006 at NFE median ~241; ANODE-p1 0.999±0.001 at NFE ~177 — reproduces "NODE eventually approximates but struggles vs ANODE." 5 seeds < Dupont's 20 (still ≥5 per our standing rule). | **RESOLVED** (budget swept, ≥5 seeds). |
| A6 | Toy **lr = 1e-3** (App. F.2.1). | **lr = 3e-3** (factorial + budget sweep). | Coursework tuned up. | Minor; faster convergence. Does not change the qualitative NFE-growth / ANODE-gap result. | **OPEN** — document or set 1e-3 (low priority). |
| A7 | Toy comparison is **ResNet vs NODE vs ANODE** (Fig 5). | **NODE vs ANODE** only; no ResNet on the toy task (our Euler baseline is MNIST-only). | Scope. | Missing one baseline curve; does not affect the NODE-vs-ANODE claim. | **OPEN** — optional. |
| A8 | Missing-slice removes an angular slice from the **training** set; Dupont's example is **`[0, π/5]`** (§5.1, p.6, verbatim: *"removing all points whose angle is in [0, π/5]"*; Fig 9). | We remove **`[0, π/5]`** (`scripts/run_missing_slice.py --width π/5`). | — | **MATCHES the paper.** D2 result (spheres, 5 seeds, accurate/recon-checked): NODE held-out slice-acc 0.62 vs ANODE 1.00 — Dupont's "large generalization gap for NODEs" reproduced. | **RESOLVED — matches paper.** *(Corrected 2026-07-15: the earlier entry "[0, π/13] / coursework chose a wider wedge" was a misread with NO source in the paper — verified by rendering p.6 of `references/1904.01681v3.pdf` and grepping the full text; π/13 appears nowhere. See OVERNIGHT_LOG.md 2026-07-15 blocker note.)* |
| A9 | Augmentation appends **p zeros** to the state (Sec 5). | Same — append zeros; **now after data-space state** (was after the stem). | — | Faithful once A1 is applied. | **RESOLVED** |
| A10 | Classifier is a **single linear map** `L` applied to the terminal ODE state, `g(x)=L(φ(x))` (Sec 2). This linearity is what makes the topological argument bite. | `ODENet.fc = nn.Linear(ode_dim, num_classes)` — a single linear layer; no MLP/nonlinearity between the ODE endpoint and the logits (audited `models/networks.py`). | — | **Already faithful.** Audited because the D8 crossing demo showed a learnable readout can let a NODE cheat; here the head is a single hyperplane, which *cannot* separate topologically-nested classes, so it does **not** do the flow's work. (A diagnostic `head_hidden_dim` MLP head is available to *show* that a nonlinear head would trivialise the task — it is not used in the faithful runs; it is the "mlp" column of the factorial.) | **RESOLVED — already linear** (hypothesis "head does the work" ruled out). |
| A11 | Toy task is **regression** of `g(x)` to **±1** with MSE; Dupont's loss plots (Fig 5, 7) are MSE (Sec 4.1). | We do **binary classification** (cross-entropy, 2 logits) on the same two regions. | Classification is the coursework framing and makes the topological bound a clean accuracy floor. | Same flow-level mechanism (both need the homeomorphism to break apart the annulus → NFE growth). "Approximate" means *correct side* for us (~0.99) vs *close to ±1* for Dupont; our accuracy is not directly comparable to his MSE curves, but the **NFE-vs-epoch** signature is (Fig 6). | **OPEN** — document; an MSE-regression variant would make the loss curves directly comparable. |
| A12 | Best d=2 **ANODE augmented dim = 5** (searched {1,2,5}, App. F.2.1). | Budget-sweep control uses **ANODE-p1** (p=1). | p=1 already gives the flat-NFE / higher-accuracy control cleanly. | p=1 suffices to show the ANODE signature (flat NFE ~180, recon 5/5, acc 0.999); p=5 would be even simpler/flatter. Not a threat to the NODE-vs-ANODE contrast. | **OPEN** — optionally add p=5 to match the paper's best. |

### Finding (A1/A3/A10) — corrected twice; this is the falsified, accurate-tolerance version

**History (kept visible on purpose — the standing rule is not to bury corrections):**
1. First pass claimed the no-stem NODE showed *"NFE explosion / geometry is load-bearing."*
2. That was retracted as GPU contention; the replacement claimed *"no NODE-failure at faithful
   settings; NFE stays flat (~44)."*
3. **Both were wrong for the same root cause: every factorial number was measured at
   `atol=rtol=1e-3`, a tolerance that does NOT integrate these fields** (forward→backward
   reconstruction error ≈0.9 on data of radius ~1; an independent linear probe on the terminal
   state collapses to chance while the model's own head reads ~1.0). "Flat NFE ~44" was the
   loose solver never doing the work. Falsification battery: `scratchpad` scripts +
   `tests/test_flow_faithfulness.py`.

**The result, re-measured at an accurate tolerance (dopri5 1e-6, reconstruction-checked;
`scripts/run_budget_sweep.py`, `results/budget/`, 5 seeds, budgets 25–1000):**

| budget | model | dense acc | NFE median (mean over seeds) | recon ok |
|---|---|---|---|---|
| 50 (Dupont) | NODE | 0.995 ± 0.006 | 241 | 5/5 |
| 50 (Dupont) | ANODE-p1 | 0.999 ± 0.001 | 177 | 5/5 |
| 1000 | NODE | 0.992 ± 0.008 | **465** (stiffest seed 812) | **4/5** |
| 1000 | ANODE-p1 | 0.999 ± 0.001 | **184** (flat) | 5/5 |

**This is a faithful reproduction of Dupont's d=2 result — there is no contradiction with the
paper.** Dupont's own words: the NODE *"eventually learns to approximate g(x), but struggles
compared to ResNets"* (§4.1), because *"the flow could then squeeze through the gaps between
sampled points"* (§4.1) at the cost that *"as the ODE needs to break apart the annulus, the
number of function evaluations increases"* (§4.2). We observe exactly this:
- **NODE approximates the finite sample** (dense acc ~0.99, never 100%) by stretching the inner
  disk into a thin **tendril that threads a gap in the annulus** — the flow stays a genuine
  homeomorphism (winding number of φ(annulus-inner-boundary) around φ(inner) = **+1.000**,
  injective, reconstructs to 1e-5 at accurate tol). It cannot reach 100%: the topological
  obstruction gives a **computable lower bound of ≥0.70%** misclassified on the annulus inner
  boundary (observed 1.35%); error is *forced*, not incidental.
- **NODE NFE grows during training** — median 219→465 over 25→1000 epochs, rising steeply in the
  first ~20 epochs as the flow breaks apart the annulus, then continuing to climb with a widening
  cross-seed spread (`figures/budget/nfe_vs_epoch.png` = Dupont Fig 6). The **stiffest seed
  drives NFE to 812 and pushes past 1e-6 faithfulness (recon fails, 4/5)** — the flow becomes
  genuinely ill-posed, exactly Dupont's *"numerically expensive to solve."*
- **ANODE-p1 is the control**: flat NFE (~180, `nfe_vs_epoch` blue), *higher* accuracy, recon 5/5
  always — the simple lifted flow Dupont predicts.

**Item 9 settled (GPU-contention retraction re-examined):** the NFE growth is **real
field-stiffening**, plainly visible at accurate tolerance (Fig 6). The earlier "150-epoch
explosion vs contention" debate was ill-posed because at the loose 1e-3 the coursework used, NFE
does *not* grow at all — the growth only appears once the solver is actually integrating. So: the
growth is real; it was invisible at 1e-3; neither prior framing (loose-tol "flat" nor
contention "explosion") described the true accurate-tolerance behaviour.

**Falsification check attached (standing rule):** the accurate-tolerance claim is only valid where
the solver integrates — every budget-sweep row carries a fwd→bwd reconstruction error and a
`recon_ok` flag; the one regime where it fails (stiffest NODE seed at ≥200 ep) is reported as a
failure, not hidden. The homeomorphism claim is checked by winding number + injectivity, not
asserted.

## B. Dupont ANODE — images (D3 MNIST; D4 CIFAR-10)

| # | Paper specifies | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| B1 | Conv ODE field injects time as an extra channel **before every convolution** (1×1→3×3→1×1, 64 filters; App. F.1.2). | **Was:** time concatenated **once** at the input. **Now:** time channel prepended before **each** conv (independently written; state-first concat, not Dupont's `Conv2dTime`). | The single-concat version is a different vector field. | Changes the conv field's time-dependence to match the paper; changes param count slightly (conv2/conv3 gain +1 input channel). | **RESOLVED** |
| B2 | Matched-param **NODE 92 filters (84,395) vs ANODE 64 filters + aug 5 (84,816)** on MNIST; batch **256**; **5 runs** (App. F.2.2, Table 1). | **BUILT (D3, 2026-07-15):** `scripts/run_d3_anode_mnist.py` — Dupont image aug (p zero channels on the input) + App F.1.2 conv field + flatten→linear head. **Param counts asserted matched: NODE 85,316 vs ANODE-p5 85,462** (~1% above Dupont's targets — a flatten-head detail). Batch 256, 5 seeds, 8 epochs. | — | **RESULT:** ANODE **98.18±0.29%** vs NODE **94.53±0.44%** test acc (Dupont Table 1: ANODE 98.2±0.1 — *matches*; NODE 96.4±0.5 — our NODE *undershoots ~2%*, attributable to the 8-epoch budget + NODE's slower convergence). **Faithful NFE (re-measured, `results/d3_faithful/`): at the loosest common recon-faithful tol 1e-7, NODE 482 vs ANODE 230 → ANODE 2.1× cheaper** (5/5 recon_ok both). Pre-declared refutation not met. | **RESOLVED — D3 reproduces the Table-1 claim** (partial on NODE's absolute number; faithful NFE at 1e-7). *(NFE comparison corrected 2026-07-15: the initial 1e-5 measurement was recon_ok only 2/5 for NODE and understated its cost; the faithful comparison is at 1e-7 — see OVERNIGHT_LOG.)* *(Updated 2026-09-10: re-run with the 1e-3 train tolerance on the recon ladder and accuracy re-measured at every rung. The train tolerance fails recon on 0/5 seeds for both arms, yet accuracy moves ≤0.03 pp at the faithful tol 1e-6. Canonical (run 2, A100): ANODE **98.05 ± 0.19**, NODE **94.16 ± 0.44** (~2.2 pp low), NFE ratio 1.65–1.91×. Also corrects a reporting slip: the 98.18/94.53 above are run-1 **medians** printed as mean ± sd; run-1 means are 98.00/94.32. `results/d3/`, OVERNIGHT_LOG.)* |
| B3 | Matched-param **NODE 125 filters (172,358) vs ANODE 64 filters + aug 10 (171,799)** on CIFAR-10; batch 256; 5 runs (App. F.2.2, Table 1). | `scripts/run_d4_anode_cifar.py`, same field and head as D3; **params asserted matched: NODE 173,611 vs ANODE-p10 172,452** (~0.5–0.7% above Dupont — the flatten-head detail). 10 epochs (Dupont's CIFAR epoch count is not stated). Trained at tol 1e-3 (Dupont: library defaults, see A4). | Compute; same protocol as D3. | The 1e-3 training tolerance does not integrate the trained field (recon 0/5), but accuracy re-measured at a checked tolerance moves ≤0.11 pp, so the accuracy comparison is unaffected; NFE is always quoted at a checked tolerance. | **RESOLVED — reproduces the Table-1 ordering; partial on ANODE's absolute number.** Run 2 (A100, faithful tol 1e-5): NODE **53.69 ± 0.81** (Dupont 53.7 ± 0.2, match); ANODE **60.04 ± 0.94** (Dupont 60.6 ± 0.4, 0.6 pp low). Run 1 (RTX 3090, `results/d4/run1_rtx3090/`): 53.59 / 59.34. The 0.6–1.3 pp ANODE miss is within our run-to-run spread. NFE ratio 1.24–1.93× across rungs and runs. *(Added 2026-09-10.)* |

## C. Chen NODE — solver dynamics & memory (C1–C4)

| # | Paper specifies | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| C1 | Table 1 MNIST uses **implicit Adams (scipy)**; classification tolerance **1e-3** (Sec 3, p.8). | **dopri5** (torchdiffeq adjoint), tol 1e-3. | Modern, GPU-friendly solver; nobody reruns the scipy/TF/autograd stack. | Different solver → different NFE dynamics; **this is the basis of the C2 finding**, not an accident. | **INTENDED** (documented). |
| C2 | **Backward NFE ≈ ½ forward NFE** (Fig 3c). | **Headline WITHDRAWN.** The previous claim "backward ≈ forward (ratio ≈1)" does not survive a tolerance guard. | Our adjoint solves the augmented reverse system (state + adjoint + parameter sensitivities), which is *stiffer and higher-dimensional* than the forward. | Guard (`scripts/c2_tolerance_guard.py`, `results/c2/`): in the **integrating** regime (recon-checked) the adaptive dopri5 ratio is **bwd/fwd ≈ 25→115× as tol tightens 1e-5→1e-8** (same on circles and spheres) — **neither Chen's 0.5 nor our claimed ≈1**, and strongly tolerance- **and** training-dependent (a random field is cheap; a trained separating field is not). The only regime where bwd==fwd is **fixed-step** (same grid backward, by construction — the `tests/test_nfe_split.py` invariant, which stands). So C2 cannot be a single headline number; it needs per-setup characterisation with a recon check. | **WITHDRAWN as headline; OPEN** — recharacterise across tolerance (never near an abstract as a single ratio). *(Updated 2026-09-10: recharacterisation DONE — `results/c2/c2_recharacterise.csv`, `c2_surface.csv`; reported as a tolerance × field surface, ≈13–121×, demoted out of the abstract.)* |
| C3 | ODE-Net memory is **O(1) in effective depth (NFE)** via the adjoint (Table 1 Memory col). | Report Table 7 varies the **Euler-baseline depth L**, not the ODE-Net's NFE; ODE-Net is a single point. | Coursework tested the wrong axis. | Does not actually test O(1); C4 rebuilds it (fix ODE-Net, drive its NFE up, show flat peak memory vs `odeint` rising). | **OPEN** — Stage C C4. *(Updated 2026-09-10: DONE — `results/c4/`: adjoint +0.0007 MB/NFE vs direct backprop +31.8 MB/NFE, 3 seeds.)* |
| C4 | Chen's ODE-Net: conv, downsample-twice + ODESolve(6 blocks), **~0.22M params, 0.42% error, ~100+ epochs**. | Our MNIST rows: MLP 204,650 / a conv baseline; 10 epochs. | We are **not** reproducing Chen's Table 1 (decision D2). | Our MNIST rows are **our own seeded baselines**, explicitly not a Chen Table-1 claim. | **RESOLVED-by-relabel** |

## D. Baseline design & general setup

| # | Paper specifies | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| D1 | A discrete **ResNet** baseline is an independent residual network (He et al.). | Our "discrete baseline" is a **weight-tied Euler discretisation of the same ODE field** (`EulerDiscretizedODENet`), one shared `ODEFunc` across Euler steps. | Buys exact parameter parity with the ODE-Net. | "Discrete and ODE-Net reach comparable accuracy" is close to tautological (same field, two integrators). Renamed honestly; documented as a controlled baseline, not an independent architecture. | **RESOLVED-by-honest-naming** |
| D2 | Image batch size **256** (Dupont); optimiser/lr/epochs largely unstated in both papers. | MNIST batch **64**, Adam, lr 1e-3, 10 epochs. | Coursework defaults. | Minor for our own baselines; D3 (Stage C) should use batch 256 to target Dupont. | **OPEN** — Stage C for D3. *(Updated 2026-09-10: D3 and D4 both run at batch 256.)* |

---

## Cross-references
- Copying-provenance (not covered here): `PROVENANCE.md`.
- Scope, budget, and which items are Stage C: `REPLICATION_PLAN.md`.
- The stem-removal (A1) validation numbers and every code change: `CHANGELOG_REPLICATION.md`.
