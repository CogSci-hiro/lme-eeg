# MixedModels.jl LRT Reduced-Residual Permutation Report

Date: 2026-07-21

This is a dev-only Release 2 Route A step 2d record. It tests whether the C2
inflation seen under within-subject design permutation is caused by
non-exchangeability of the condition design when the data-generating and fitted
models include a subject random slope.

## Method

The scheme used reduced-model conditional residual permutation:

1. Fit the reduced ML mixed model once per feature on the observed data.
2. Drop only the fixed condition effect, while retaining the random-effects
   structure. For the random-slope model the reduced formula is
   `y ~ 1 + (1 + cond | subject) + (1 | item)`.
3. Store `fitted(reduced)` and `residuals(reduced)`.
4. Keep condition labels fixed. For each permutation, shuffle the conditional
   residuals within subject and reconstruct
   `y_perm = fitted_reduced + residual_perm`.
5. Refit full and reduced ML models on `y_perm`, compute the unsigned LR
   statistic `2 * (logLik_full - logLik_reduced)`, and summarize the resulting
   LR maps with maxstat, cluster, and TFCE.

Conditional residuals were chosen because they are the closest available
estimate of observation-level error after the reduced mixed model has accounted
for subject/item random effects. Keeping condition labels fixed avoids the
random-slope design scramble that likely made design permutation nonexchangeable.
The limitation is unchanged from the earlier fast path: the residual permutation
respects subject blocks but does not impose a crossed item exchangeability
constraint.

The correction helpers use one-sided LR maps: max over nonnegative LR maps for
maxstat and `tail=1` in MNE cluster/TFCE. P-values use `(b + 1) / (m + 1)`.
Negative LR values below `-1e-8` are convergence failures and trigger the tight
LRT retry path. These runs had zero unresolved negative LR maps.

## Guard

The random-slope H0 guard passed before FWER calibration:

```text
LMEEG_PROGRESS_EVERY=25 RPY2_CFFI_MODE=ABI \
R_LIBS_USER=/Users/hiro/Projects/active/lme-eeg/.venv/R/library \
.venv/bin/python -m pytest \
  tests/backends/lmm/test_real_lmm_null_calibration_slow.py::\
test_real_lmm_slope_h0_guard_passes_before_refit_calibration \
  -q -s -m slow

1 passed in 465.47 s
```

## Results

Settings for all cells:

- `permutation_method=reduced_residual`
- 6 subjects x 5 items
- four features
- `n_sims=300`
- `n_permutations=500`
- full and reduced MixedModels.jl fits by ML
- LR statistic, no Satterthwaite/KR inside the permutation loop

| Scenario | Backend | Rate | MC SE | Observed singular | Permuted singular | Permuted refit-trigger | Mean sec/sim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C2 random slope H0 | maxstat | 0.003 | 0.003 | 0.001 | 0.002 | 0.000007 | 21.994 |
| C2 random slope H0 | cluster | 0.017 | 0.007 | 0.001 | 0.002 | 0.000007 | 21.994 |
| C2 random slope H0 | TFCE | 0.007 | 0.005 | 0.001 | 0.002 | 0.000007 | 21.994 |
| C1 crossed intercept H0 | maxstat | 0.047 | 0.012 | 0.002 | 0.004 | 0.000000 | 15.132 |
| C1 crossed intercept H0 | cluster | 0.057 | 0.013 | 0.002 | 0.004 | 0.000000 | 15.132 |
| C1 crossed intercept H0 | TFCE | 0.050 | 0.013 | 0.002 | 0.004 | 0.000000 | 15.132 |
| C3 fixed effect present | maxstat | 0.000 | 0.000 | 0.000 | 0.001 | 0.000000 | 8.275 |
| C3 fixed effect present | cluster | 0.027 | 0.009 | 0.000 | 0.001 | 0.000000 | 8.275 |
| C3 fixed effect present | TFCE | 0.013 | 0.007 | 0.000 | 0.001 | 0.000000 | 8.275 |

Design-permutation comparison for the decisive C2 6x5 cell:

| Scheme | maxstat | cluster | TFCE |
| --- | --- | --- | --- |
| Design permutation, tight LRT (`9c9af5b`) | 0.123 | 0.127 | 0.167 |
| Wald design permutation (`9fb2aab`) | 0.130 | 0.133 | 0.173 |
| Reduced-residual permutation, tight LRT | 0.003 | 0.017 | 0.007 |

## Verdict

The C2 inflation is reconciled as a permutation-scheme failure for design
permutation under random slopes: reduced-residual permutation drops the C2 6x5
FWER from roughly 0.12-0.17 to conservative rates, and C1 remains nominal.

This is not a production GO. The C3 power sanity check fails at the same 6x5
calibration scale, with rates below alpha for all three correction summaries.
That means the tested reduced-residual scheme is probably too conservative for
the effect-present alternative in this harness, despite fixing the C2 null.

Next step should be a focused power diagnostic or a revised residual-resampling
scheme before any production backend work. Do not promote this scheme solely on
the C2 null result.

## Reproduction Commands

Decisive C2 cell:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_mixedmodels_lrt_refit.py \
  --skip-guard --print-contract --permutation-method reduced_residual \
  --n-sims 300 --n-permutations 500 --workers 4 \
  --scenarios C2 --sizes 6x5 \
  --output-dir dev/bench/mixedmodels_lrt_residual_c2_6x5 \
  --progress-every 25
```

C1 and C3 used the same command with `--scenarios C1` and `--scenarios C3`,
writing to the corresponding `dev/bench/mixedmodels_lrt_residual_*_6x5`
directories.
