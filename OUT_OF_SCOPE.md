# Material excluded from the replication

The submission replicates Dupont et al. 2019 (Augmented Neural ODEs) and four solver and
memory claims of Chen et al. 2018. The Latent ODE and ODE-RNN work of Rubanova et al. 2019 is
excluded, for three reasons.

1. Rubanova's headline claims are on PhysioNet, MuJoCo Hopper and Human Activity. Running any
   of those is beyond the compute available here, so a replication of that paper is not
   possible at our scope.
2. The two-dimensional spiral that is often used as a stand-in is an experiment from Chen et
   al. 2018, not from Rubanova et al. Presenting it under a Rubanova heading would be a
   citation error.
3. A synthetic one-dimensional sine can only be compared against Rubanova's toy supplementary
   Table 2, and with substantial setup drift. That is architecture validation rather than a
   controlled reproduction.

## What this means

- No Latent ODE, ODE-RNN or time-series code is in this repository. Nothing in
  `make reproduce-all` or in the test suite depends on that material.
- Rubanova et al. 2019 is cited in the paper as related work only.

## Also out of scope

| Excluded | Why |
|---|---|
| SVHN and ImageNet from Dupont et al., Table 1 | Beyond the compute available. The two datasets we do run, MNIST and CIFAR-10, carry the same claim. |
| Chen et al., Table 1 error rates | Their ODE-Net trains for 100 or more epochs. Our MNIST rows are our own seeded baselines at 10 epochs and are not a claim about that table. See `DEVIATIONS.md` row S4. |
| A discrete residual baseline | Weight tying is the only way to match parameter counts exactly, and it makes the comparison close to tautological, since the two models are then the same vector field under two integrators. Claim C3 is instead tested on the axis the claim is about, the model's own NFE. |

The scope of each claim we do test, and where it departs from the source papers, is in
[`DEVIATIONS.md`](DEVIATIONS.md).
