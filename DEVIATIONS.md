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
| A1 | **ODE integrates in data space**: the toy field is an MLP acting on the raw input; the homeomorphism argument (a NODE cannot separate nested regions) requires no learned warp before the flow (Sec 3–4). | **Was:** a `Linear(2,2)+Tanh` downsampling stem before the ODE. **Now:** no stem — ODE integrates the raw 2-D input (`use_stem=False`). | The stem is a diffeomorphism that hands the model a free learned warp before the flow; the faithful setup has none. | **MEASURED (factorial, ≥5 seeds, 150 ep, NFE-capped 1500):** removing the stem does **not** hurt and slightly *helps* the NODE (circles 0.991 no-stem vs 0.955 stem; spheres 0.998 vs 0.966), at flat NFE (median ~42, peak ≤77) with **0/25 divergences**. The stem is not why the coursework NODE did/didn't fail. Code change kept regardless (data-space integration is faithful). | **RESOLVED** (code + measured). |
| A2 | Toy vector-field width **32** (`d+1→32→32→d`, App. F.1.1). | **Was:** `ode_hidden_dim=64`. **Now:** 32. | Coursework widened the field "for capacity"; a wider field lets the NODE better approximate the distorted flow, again flattering the NODE baseline. 32 matches the paper. | Narrower field → NODE slightly less able to fake the tearing → larger NODE-vs-ANODE gap. | **RESOLVED** |
| A3 | Dataset: **concentric spheres** — filled inner disk `‖x‖≤0.5` + annulus `1.0≤‖x‖≤1.5`; **1000 inner + 2000 outer** (App. F.2.1). | **Was:** two thin **circles** only. **Now:** `make_spheres` added (`data/synthetic.py`, registered as `"spheres"`); `make_circles` kept for the coursework comparison. | The paper specifies this geometry; the coursework used thin circles. | **MEASURED (factorial):** the sphere geometry is **not** harder for the NODE at these settings — spheres/no-stem 0.998 ≥ circles/no-stem 0.991, same flat NFE (~42), 0 divergences. So geometry is **not** the load-bearing factor either. (Earlier claim that geometry was load-bearing is **retracted** — see the correction note below the table.) | **RESOLVED** (implemented + measured); geometry is still the *paper-specified* dataset for D1/D2 regardless. |
| A4 | Solver **RK45 (dopri5), atol = 1e-3**, no rtol stated (App. F). | dopri5 with **atol = rtol = 1e-3**. | Coursework set both tolerances. | Adds a relative-tolerance criterion the paper does not specify; small effect on NFE/accuracy. | **OPEN** — document; optionally loosen rtol so only atol binds. |
| A5 | **50 epochs**, **20 repeats** (App. F.2.1). | **500 epochs**, **3 seeds** (→ ≥5 in Stage C). | Coursework trained longer / fewer seeds. | Longer training mainly helps the NODE saturate; 3 seeds is statistically thin (this is why report Table 3's slice loss std>mean). | **OPEN** — ≥5 seeds in Stage C; keep or reduce epochs (document). |
| A6 | Toy **lr = 1e-3** (App. F.2.1). | **lr = 3e-3**. | Coursework tuned up. | Minor; faster convergence. | **OPEN** — document or set 1e-3. |
| A7 | Toy comparison is **ResNet vs NODE vs ANODE** (Fig 5). | **NODE vs ANODE** only; no ResNet on the toy task (our Euler baseline is MNIST-only). | Scope. | Missing one baseline curve; does not affect the NODE-vs-ANODE claim. | **OPEN** — optional. |
| A8 | Missing-slice removes training points with angle in **`[0, π/13]`** (Fig 9). | Removes a sector of width **`π/5`**. | Coursework chose a wider wedge. | Wider held-out region → harder generalization; the §6 grid varies this on purpose. | **OPEN** — grid sweeps width in Stage C. |
| A9 | Augmentation appends **p zeros** to the state (Sec 5). | Same — append zeros; **now after data-space state** (was after the stem). | — | Faithful once A1 is applied. | **RESOLVED** |
| A10 | Classifier is a **single linear map** `L` applied to the terminal ODE state, `g(x)=L(φ(x))` (Sec 2). This linearity is what makes the topological argument bite. | `ODENet.fc = nn.Linear(ode_dim, num_classes)` — a single linear layer; no MLP/nonlinearity between the ODE endpoint and the logits (audited `models/networks.py`). | — | **Already faithful.** Audited because the D8 crossing demo showed a learnable readout can let a NODE cheat; here the head is a single hyperplane, which *cannot* separate topologically-nested classes, so it does **not** do the flow's work. (A diagnostic `head_hidden_dim` MLP head is available to *show* that a nonlinear head would trivialise the task — it is not used in the faithful runs; it is the "mlp" column of the factorial.) | **RESOLVED — already linear** (hypothesis "head does the work" ruled out). |

### Finding (A1/A3/A10) — and a correction

**Result of the {geometry}×{stem}×{head} factorial (`scripts/run_stem_geometry_factorial.py`,
`results/factorial/`, ≥5 seeds, 150 epochs, dopri5 tol 1e-3, solver capped at 1500 NFE):**

| geometry | stem | head | diverged | val acc | NFE median [IQR] | NFE peak-max |
|---|---|---|---|---|---|---|
| circles | no-stem | linear | 0/5 | 0.991 ± 0.013 | 44 [37,50] | 74 |
| circles | stem | linear | 0/5 | 0.955 ± 0.046 | 45 [40,68] | 77 |
| spheres | no-stem | linear | 0/5 | 0.998 ± 0.003 | 42 [37,52] | 70 |
| spheres | stem | linear | 0/5 | 0.966 ± 0.058 | 42 [36,55] | 70 |
| spheres | no-stem | **mlp** (diagnostic) | 0/5 | 0.999 ± 0.002 | 32 [30,32] | 35 |

**None of the three suspected deviations (stem, geometry, head) reproduces a NODE-failure —
because there is no NODE-failure at these faithful settings.** The data-space NODE with a
single linear head separates *both* concentric circles and Dupont's filled-disk+annulus
spheres at ~99% with modest, non-exploding NFE (median ~42, peak ≤77, well under the 1500
cap) and **0/25 divergences**. Removing the stem slightly *helps*; the sphere geometry is not
harder. The MLP-head diagnostic needs *less* NFE (32 vs 42), confirming the direction "a
nonlinear head does the flow's work" — but the linear-head flow is not working hard either.

**Correction of an earlier claim (retracted):** a previous changelog/DEVIATIONS entry said the
no-stem NODE showed "NFE explosion" and that geometry was "the load-bearing factor." That was
**wrong**: the apparent explosion was a **150-epoch GPU run contending with another user's 8 GB
GPU job** (discovered later), not solver divergence. Re-run on CPU without contention, every
cell converges at flat NFE. The claim is withdrawn.

**What this does and does NOT establish (no overclaim):**
- It **does** show the simple reading — "a faithful low-dim NODE catastrophically fails / has
  exploding NFE" — is **not reproduced** here, and that the coursework's NODE numbers
  (Table 2: 90.2% ± 8.8, NFE median 89.85 / max 307.69) are **not** reproduced by the faithful
  setup (which gives higher acc, lower NFE) — a candidate *failure-to-reproduce-the-mechanism*.
- It **does not** refute Dupont's actual claim, which is **comparative** (ANODE vs NODE: lower
  and flatter NFE, better held-out-slice generalization) and about **NFE growth over long
  training** + the **held-out-slice** gap — not NODE-only iid accuracy. This factorial ran
  NODE only, iid-val, 150 epochs. The comparative test is **D1** (with ANODE) and **D2**
  (Dupont Fig 9, held-out slices), at the paper's budget, in Stage C.
- **Caveat:** budget was 150 epochs vs the report's 500; whether NODE NFE grows at longer
  budgets is untested here (no sign of it at 150).
- **Methodological implication (aligns with plan §9):** iid-val accuracy on these datasets does
  not expose a NODE bottleneck; it is the wrong lens. D1/D2 must lead with NFE and held-out
  generalization, not iid accuracy.

## B. Dupont ANODE — images (D3 MNIST; D4 CIFAR gated)

| # | Paper specifies | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| B1 | Conv ODE field injects time as an extra channel **before every convolution** (1×1→3×3→1×1, 64 filters; App. F.1.2). | **Was:** time concatenated **once** at the input. **Now:** time channel prepended before **each** conv (independently written; state-first concat, not Dupont's `Conv2dTime`). | The single-concat version is a different vector field. | Changes the conv field's time-dependence to match the paper; changes param count slightly (conv2/conv3 gain +1 input channel). | **RESOLVED** |
| B2 | Matched-param **NODE 92 filters (84,395) vs ANODE 64 filters + aug 5 (84,816)** on MNIST; batch **256**; **5 runs** (App. F.2.2, Table 1). | Not yet built (D3 is Stage C). Current MNIST CNN is our own baseline (num_filters arg, batch 64), not the matched NODE-vs-ANODE pair. | Scope/timing. | D3 must implement the matched pair + batch 256 to target Table 1. | **OPEN** — Stage C D3. |

## C. Chen NODE — solver dynamics & memory (C1–C4)

| # | Paper specifies | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| C1 | Table 1 MNIST uses **implicit Adams (scipy)**; classification tolerance **1e-3** (Sec 3, p.8). | **dopri5** (torchdiffeq adjoint), tol 1e-3. | Modern, GPU-friendly solver; nobody reruns the scipy/TF/autograd stack. | Different solver → different NFE dynamics; **this is the basis of the C2 finding**, not an accident. | **INTENDED** (documented). |
| C2 | **Backward NFE ≈ ½ forward NFE** (Fig 3c). | We observe **backward ≈ forward** with `odeint_adjoint`. | Our adjoint solves the augmented reverse system (state + adjoint + parameter sensitivities) jointly. | This is the **promoted headline** — a genuine implementation-level divergence. Gated behind a proven-correct NFE-split test (Stage B) and a torchdiffeq-version sweep (Stage C). | **INTENDED / FINDING** |
| C3 | ODE-Net memory is **O(1) in effective depth (NFE)** via the adjoint (Table 1 Memory col). | Report Table 7 varies the **Euler-baseline depth L**, not the ODE-Net's NFE; ODE-Net is a single point. | Coursework tested the wrong axis. | Does not actually test O(1); C4 rebuilds it (fix ODE-Net, drive its NFE up, show flat peak memory vs `odeint` rising). | **OPEN** — Stage C C4. |
| C4 | Chen's ODE-Net: conv, downsample-twice + ODESolve(6 blocks), **~0.22M params, 0.42% error, ~100+ epochs**. | Our MNIST rows: MLP 204,650 / a conv baseline; 10 epochs. | We are **not** reproducing Chen's Table 1 (decision D2). | Our MNIST rows are **our own seeded baselines**, explicitly not a Chen Table-1 claim. | **RESOLVED-by-relabel** |

## D. Baseline design & general setup

| # | Paper specifies | We do | Why | Expected effect | Status |
|---|---|---|---|---|---|
| D1 | A discrete **ResNet** baseline is an independent residual network (He et al.). | Our "discrete baseline" is a **weight-tied Euler discretisation of the same ODE field** (`EulerDiscretizedODENet`), one shared `ODEFunc` across Euler steps. | Buys exact parameter parity with the ODE-Net. | "Discrete and ODE-Net reach comparable accuracy" is close to tautological (same field, two integrators). Renamed honestly; documented as a controlled baseline, not an independent architecture. | **RESOLVED-by-honest-naming** |
| D2 | Image batch size **256** (Dupont); optimiser/lr/epochs largely unstated in both papers. | MNIST batch **64**, Adam, lr 1e-3, 10 epochs. | Coursework defaults. | Minor for our own baselines; D3 (Stage C) should use batch 256 to target Dupont. | **OPEN** — Stage C for D3. |

---

## Cross-references
- Copying-provenance (not covered here): `PROVENANCE.md`.
- Scope, budget, and which items are Stage C: `REPLICATION_PLAN.md`.
- The stem-removal (A1) validation numbers and every code change: `CHANGELOG_REPLICATION.md`.
