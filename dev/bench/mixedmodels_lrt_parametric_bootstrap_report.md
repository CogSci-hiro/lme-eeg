# MixedModels.jl LRT Parametric Bootstrap Calibration

Date: 2026-07-22

This is a dev-only Release 2 step 2f record. It tests parametric bootstrap as
the family-wise null for random slopes after permutation and residual
permutation failed to provide both calibrated C2 and useful C3 power.

## V1/V2 Check From Step 2E

The step-2e V1 and V2 rejection rates were bit-identical, but the implementation
did not literally use the same code path in the normal, non-retry path.

- V1 used marginal residuals from the fixed-effect-reduced random-slope model:
  `y ~ 1 + (1 + cond | subject) + (1 | item)`.
- V2 used marginal residuals from the fixed-effect-reduced crossed-intercept
  model:
  `y ~ 1 + (1 | subject) + (1 | item)`.

Evidence that V2 did drop the random slope: V1/V2 diagnostics differed in
runtime and singular fractions, even though reject decisions were identical for
the fixed seeds. For C2, observed singular was `0.001` in V1 versus `0.002` in
V2; mean seconds per sim was `31.759` versus `29.916`. For C1/C3, V2 was much
faster because its residual-source model was simpler.

One latent bug was found while checking this: the residual-setup retry branch
referenced `random_slope` inside Julia instead of `test_random_slope`. That
branch only runs after a negative observed LR in residual setup; the step-2e
evidence had no unresolved negative LR maps and no observed residual-setup retry
rate, so the recorded V1/V2 normal-path results were not produced by the same
code path. The dev harness now fixes this retry typo.

## Scheme

For each simulated dataset and each feature:

1. Fit the fixed-effect-reduced null model by ML. For random-slope scenarios,
   this drops fixed `cond` but keeps the full random structure:
   `y ~ 1 + (1 + cond | subject) + (1 | item)`.
2. Fit the full model by ML and compute the observed unsigned LR statistic:
   `2 * (logLik_full - logLik_reduced)`.
3. For each bootstrap replicate, call MixedModels.jl `simulate(rng, reduced)`
   on the fitted reduced model to generate a new response from the fitted null.
4. Refit full and reduced ML models to the simulated response and compute the
   same LR statistic.
5. Use the existing one-sided maxstat, cluster, and TFCE summaries over the
   bootstrap LR maps.

The same tight-convergence LRT retry path from the earlier MixedModels.jl work
is used. No Satterthwaite/KR df are used inside the family-wise null.

Parametric bootstrap does not require exchangeability of labels or residuals,
but it does assume the fitted null model is distributionally correct. In this
harness that means Gaussian random effects and Gaussian residuals with the
fitted covariance structure. Because the data generator is also Gaussian, this
calibration is partly self-fulfilling: it tests the plumbing and finite-sample
behavior under correct specification, not robustness to non-Gaussian EEG errors.

## Guard

The random-slope H0 guard passed before bootstrap calibration:

```text
LMEEG_PROGRESS_EVERY=25 RPY2_CFFI_MODE=ABI \
R_LIBS_USER=/Users/hiro/Projects/active/lme-eeg/.venv/R/library \
.venv/bin/python -m pytest \
  tests/backends/lmm/test_real_lmm_null_calibration_slow.py::\
test_real_lmm_slope_h0_guard_passes_before_refit_calibration \
  -q -s -m slow

1 passed in 276.73 s
```

## Results

Settings for all cells:

- 6 subjects x 5 items
- four features
- `n_sims=300`
- `n_boot=500`
- MixedModels.jl full and reduced fits by ML
- unsigned LR maps
- maxstat, cluster, and TFCE summaries

Rates are `maxstat / cluster / TFCE`.

| Scenario | Rates | MC SEs | Mean sec/sim | Verdict |
| --- | --- | --- | --- | --- |
| C1 crossed-intercept H0 | 0.050 / 0.060 / 0.057 | 0.013 / 0.014 / 0.013 | 2.272 | nominal |
| C2 random-slope H0 | 0.047 / 0.093 / 0.087 | 0.012 / 0.017 / 0.016 | 5.357 | mixed |
| C3 fixed effect | 0.340 / 0.613 / 0.590 | 0.027 / 0.028 / 0.028 | 5.372 | powered |

No unresolved negative LR maps occurred. Singular fractions were low:

- C1 observed/permuted singular: `0.002 / 0.008`
- C2 observed/permuted singular: `0.001 / 0.004`
- C3 observed/permuted singular: `0.000 / 0.002`

## Verdict

Parametric bootstrap is the best-performing random-slope family-wise null tested
so far, but it is not a clean production GO at 6x5 for all three correction
backends.

It fixes the main residual-permutation failure mode:

- C1 is nominal.
- C3 power is usable: `0.340 / 0.613 / 0.590`.
- C2 maxstat is nominal at `0.047`.

The blocker is C2 cluster/TFCE: `0.093 / 0.087`. With `n_sims=300`, a three-SE
nominal bound is approximately `0.088`, so TFCE is right at the boundary and
cluster is just above it. This is far better than design permutation and
marginal residual permutation, but it does not satisfy the requested "must be
nominal" criterion across all backends.

Do not build production from this slice alone. The next useful check is either a
larger-design bootstrap cell or a higher-precision 6x5 rerun focused on
cluster/TFCE to separate a small-sample edge effect from residual inflation.
