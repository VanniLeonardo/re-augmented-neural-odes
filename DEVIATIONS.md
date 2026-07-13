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
| A1 | **ODE integrates in data space**: the toy field is an MLP acting on the raw input; the homeomorphism argument (a NODE cannot separate nested regions) requires no learned warp before the flow (Sec 3–4). | **Was:** a `Linear(2,2)+Tanh` downsampling stem before the ODE. **Now:** no stem — ODE integrates the raw 2-D input (`use_stem=False`). | The stem is a diffeomorphism that hands the model a free learned warp before the flow; the faithful setup has none. | **MEASURED (30ep, 2 seeds, thin circles): removing the stem did NOT degrade the NODE** — val acc 0.998 (no-stem) vs 0.873 (stem), within noise. So the stem is *not* why the coursework NODE reached 90%. The dominant factor is the **dataset geometry (A3)**: on thin circles a data-space NODE still succeeds; Dupont's NODE-failure needs the filled-disk+annulus geometry. Stem removal remains the faithful choice; it just isn't sufficient on its own. | **RESOLVED** (code); **finding recorded** — see `CHANGELOG_REPLICATION.md`. |
| A2 | Toy vector-field width **32** (`d+1→32→32→d`, App. F.1.1). | **Was:** `ode_hidden_dim=64`. **Now:** 32. | Coursework widened the field "for capacity"; a wider field lets the NODE better approximate the distorted flow, again flattering the NODE baseline. 32 matches the paper. | Narrower field → NODE slightly less able to fake the tearing → larger NODE-vs-ANODE gap. | **RESOLVED** |
| A3 | Dataset: **concentric spheres** — filled inner disk `‖x‖≤0.5` + annulus `1.0≤‖x‖≤1.5`; **1000 inner + 2000 outer** (App. F.2.1). | Two thin **circles** (sklearn-style, uniform angle, radius 1 & 0.5), **500/500 balanced** (`data/synthetic.py:make_circles`). | Coursework used the classic sklearn geometry. | **LOAD-BEARING (see A1 validation): this, not the stem, is the dominant reason the coursework NODE did not fail.** Thin circles are two disjoint 1-D loops — a much weaker topological bottleneck than a filled disk enclosed by an annulus, which is Dupont's actual hard case (a homeomorphism cannot tear the enclosed region out). Reproducing Dupont's NODE-failure **requires** this fix. | **OPEN — elevated priority.** Align to Dupont's filled-disk+annulus spheres early in Stage C (plan P1-15); D1's NODE-failure claim depends on it. |
| A4 | Solver **RK45 (dopri5), atol = 1e-3**, no rtol stated (App. F). | dopri5 with **atol = rtol = 1e-3**. | Coursework set both tolerances. | Adds a relative-tolerance criterion the paper does not specify; small effect on NFE/accuracy. | **OPEN** — document; optionally loosen rtol so only atol binds. |
| A5 | **50 epochs**, **20 repeats** (App. F.2.1). | **500 epochs**, **3 seeds** (→ ≥5 in Stage C). | Coursework trained longer / fewer seeds. | Longer training mainly helps the NODE saturate; 3 seeds is statistically thin (this is why report Table 3's slice loss std>mean). | **OPEN** — ≥5 seeds in Stage C; keep or reduce epochs (document). |
| A6 | Toy **lr = 1e-3** (App. F.2.1). | **lr = 3e-3**. | Coursework tuned up. | Minor; faster convergence. | **OPEN** — document or set 1e-3. |
| A7 | Toy comparison is **ResNet vs NODE vs ANODE** (Fig 5). | **NODE vs ANODE** only; no ResNet on the toy task (our Euler baseline is MNIST-only). | Scope. | Missing one baseline curve; does not affect the NODE-vs-ANODE claim. | **OPEN** — optional. |
| A8 | Missing-slice removes training points with angle in **`[0, π/13]`** (Fig 9). | Removes a sector of width **`π/5`**. | Coursework chose a wider wedge. | Wider held-out region → harder generalization; the §6 grid varies this on purpose. | **OPEN** — grid sweeps width in Stage C. |
| A9 | Augmentation appends **p zeros** to the state (Sec 5). | Same — append zeros; **now after data-space state** (was after the stem). | — | Faithful once A1 is applied. | **RESOLVED** |

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
