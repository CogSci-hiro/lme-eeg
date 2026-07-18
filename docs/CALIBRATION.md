# lmeEEG Calibration Record

Date: 2026-07-18

This file is the source of truth for the current fast marginal-OLS calibration
envelope. Do not use pre-3f33618 calibration numbers.

## Provenance

Calibration was run after these commits:

- `3f33618` fixed the calibration null generator.
- `07006fc` added the random-slope null guard and C2 sweep harness.

Runtime stack:

- R 4.5.3
- lme4 2.0.1
- lmerTest 3.2.1
- pymer4 0.9.2
- rpy2 3.6.7

Run settings unless noted:

- Permutation scheme: `within_subject`
- Simulations per cell: `n_sims=300`
- Permutations per simulation: `n_perms=1000`
- Monte Carlo SE: `sqrt(FWER * (1 - FWER) / n_sims)`

## Void Results

All calibration numbers produced before `3f33618` are VOID.

The pre-fix null generator was contaminated: the condition effect leaked through
location/time structure even when `fixed_effect=0`. Specifically, the generated
condition coefficient was effectively `fixed_effect + 0.03 * location - 0.02 *
time`, so three of four features were not null under nominal H0. Do not
resurrect the old C1 `0.80` size-sweep result or the old C2 `0.67` result.

## C1: Crossed Random Intercepts

Scenario: crossed subject/item random intercepts, H0 true at every feature.
Formula fitted by lme4: `y ~ cond + (1 | subject) + (1 | item)`.

FWER +/- MC SE:

| Size | maxstat | cluster | tfce |
| --- | ---: | ---: | ---: |
| 6x5 | 0.063 +/- 0.014 | 0.067 +/- 0.014 | 0.063 +/- 0.014 |
| 12x10 | 0.037 +/- 0.011 | 0.037 +/- 0.011 | 0.027 +/- 0.009 |
| 24x20 | 0.043 +/- 0.012 | 0.037 +/- 0.011 | 0.043 +/- 0.012 |
| 36x30 | 0.053 +/- 0.013 | 0.053 +/- 0.013 | 0.037 +/- 0.011 |

Verdict: calibrated to nominal. The C1 fast marginal-OLS path is flat to mildly
conservative across n under `within_subject` permutation.

## C2: Random Slope Present

Scenario: subject-varying random condition slopes with zero population mean
effect. H0 on the mean is true at every feature. The random slope is present in
the data-generating model and in the fitted lme4 model.

Formula fitted by lme4: `y ~ cond + (1 + cond | subject) + (1 | item)`.

FWER +/- MC SE:

| Size | maxstat | cluster | tfce |
| --- | ---: | ---: | ---: |
| 6x5 | 0.663 +/- 0.027 | 0.680 +/- 0.027 | 0.733 +/- 0.026 |
| 12x10 | 0.700 +/- 0.026 | 0.713 +/- 0.026 | 0.780 +/- 0.024 |
| 24x20 | 0.773 +/- 0.024 | 0.790 +/- 0.024 | 0.830 +/- 0.022 |
| 36x30 | 0.800 +/- 0.023 | 0.813 +/- 0.022 | 0.857 +/- 0.020 |

Verdict: NOT calibrated. Inflation grows with n, and no usable design size was
found. The correction backends agree, so this is not a maxstat/cluster/TFCE
scheme-specific artifact. The cause is the statistic: OLS t-values on
marginalized data ignore subject-varying slope variance.

## C3: Power Sanity

Scenario: crossed subject/item random intercepts with a true fixed condition
effect present on the corrected generator. Formula fitted by lme4:
`y ~ cond + (1 | subject) + (1 | item)`.

Detection rate +/- MC SE:

| Size | maxstat | cluster | tfce |
| --- | ---: | ---: | ---: |
| 6x5 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |

Verdict: power sanity passes on the corrected generator; the calibrated C1
pipeline is not passing only by never rejecting.

## Validated Envelope

Release 1 validates the fast marginal-OLS path for crossed random intercepts
under `within_subject` permutation. Random slopes are INVALID on the fast path
and are refused pending the real-LMM path, which is not yet built.

## Reproduce

Run from the repository root with the project virtual environment.

Generator guards:

```bash
LMEEG_PROGRESS_EVERY=25 \
  ./.venv/bin/python -m pytest \
  tests/backends/lmm/test_pymer4_calibration_slow.py::test_pymer4_calibration_generator_h0_is_null_at_every_feature \
  tests/backends/lmm/test_pymer4_calibration_slow.py::test_pymer4_calibration_slope_generator_h0_is_null_at_every_feature \
  -q -s
```

C1 crossed-intercept size sweep:

```bash
LMEEG_D3_N_SIMS=300 LMEEG_D3_N_PERMUTATIONS=1000 \
LMEEG_D3_PERMUTATION_SCHEME=within_subject LMEEG_PROGRESS_EVERY=25 \
  ./.venv/bin/python -m pytest \
  tests/backends/lmm/test_pymer4_calibration_slow.py::test_pymer4_d3_c1_size_sweep \
  -q -m slow -s
```

C1 Release 1 regression guard. This test is marked `slow` because it fits real
pymer4/lme4 models, but it must pass before any Release 1 tag:

```bash
LMEEG_RELEASE_C1_N_SIMS=200 LMEEG_RELEASE_C1_N_PERMUTATIONS=500 \
LMEEG_PROGRESS_EVERY=25 \
  ./.venv/bin/python -m pytest \
  tests/backends/lmm/test_pymer4_calibration_slow.py::test_pymer4_release_c1_maxstat_regression_guard \
  -q -m slow -s
```

C2 random-slope size sweep:

```bash
LMEEG_C2_N_SIMS=300 LMEEG_C2_N_PERMUTATIONS=1000 \
LMEEG_C2_PERMUTATION_SCHEME=within_subject LMEEG_PROGRESS_EVERY=25 \
  ./.venv/bin/python -m pytest \
  tests/backends/lmm/test_pymer4_calibration_slow.py::test_pymer4_c2_random_slope_size_sweep \
  -q -m slow -s
```

C3 power sanity:

```bash
LMEEG_CALIBRATION_N_SIMS=300 LMEEG_CALIBRATION_N_PERMUTATIONS=1000 \
LMEEG_PROGRESS_EVERY=25 \
  ./.venv/bin/python -m pytest \
  tests/backends/lmm/test_pymer4_calibration_slow.py::test_pymer4_calibration_c3_power_sanity \
  -q -m slow -s
```
