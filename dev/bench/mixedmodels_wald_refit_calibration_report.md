# MixedModels.jl Wald-Refit Permutation Calibration

Date: 2026-07-21

Purpose: Release 2 Route A step 2. Calibrate the full-refit engine-only
permutation path for random slopes at calibration scale, using the engine
contract from `f992ad4`:

- MixedModels.jl full refit for every observed and permuted feature.
- In-loop statistic is the default Wald t/z-style statistic:
  `fixef(model)[cond] / stderror(model)[cond]`.
- No Satterthwaite or Kenward-Roger inside the permutation loop.
- Within-subject condition permutations only.
- Existing Python maxstat/cluster/TFCE correction summaries unchanged.

This is a dev/calibration record only. No production backend was built or wired
into the real pipeline.

## Guard

The random-slope H0 null guard passed before any FWER number was trusted:

```bash
LMEEG_PROGRESS_EVERY=25 \
RPY2_CFFI_MODE=ABI \
R_LIBS_USER=/Users/hiro/Projects/active/lme-eeg/.venv/R/library \
.venv/bin/python -m pytest \
  tests/backends/lmm/test_real_lmm_null_calibration_slow.py::test_real_lmm_slope_h0_guard_passes_before_refit_calibration \
  -q -s -m slow
```

Result:

```text
1 passed, 800 warnings in 374.31s (0:06:14)
```

## Calibration Command

```bash
MPLCONFIGDIR=/tmp/mplconfig \
LMEEG_JULIA_WALD_WORKERS=8 \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_mixedmodels_wald_refit.py \
  --skip-guard \
  --n-sims 300 \
  --n-permutations 500 \
  --workers 8 \
  --output-dir dev/bench/mixedmodels_wald_refit_calibration \
  --progress-every 25
```

The first sandboxed attempt failed before running simulations because
`ProcessPoolExecutor` could not query multiprocessing semaphore limits in the
sandbox. The same command was rerun with sandbox escalation so Python
multiprocessing could work.

Checkpoint output:

- Raw simulation rows:
  `dev/bench/mixedmodels_wald_refit_calibration/sim_results.jsonl`
- Partial-stop summary:
  `dev/bench/mixedmodels_wald_refit_calibration/summary_partial_stop.json`

The checkpoint write window was 2026-07-21 19:04:13 to 20:05:42 local time.
The run was intentionally interrupted after C2 6x5 completed and satisfied the
stop condition.

## Results

FWER is reported as `rate +/- Monte Carlo SE`, where
`SE = sqrt(rate * (1 - rate) / n_sims)`. Singular fractions are fit fractions:
observed denominator is `n_sims * 4` feature fits; permuted denominator is
`n_sims * 500 * 4` feature fits.

### C1: Crossed-Intercept H0 Control

All C1 cells completed with `n_sims=300`, `n_permutations=500`.

| Size | Backend | FWER +/- MC SE | Observed singular | Permuted singular | Mean s/sim |
| --- | --- | ---: | ---: | ---: | ---: |
| 6x5 | maxstat | 0.073 +/- 0.015 | 0.002 | 0.002 | 9.7 |
| 6x5 | cluster | 0.070 +/- 0.015 | 0.002 | 0.002 | 9.7 |
| 6x5 | tfce | 0.043 +/- 0.012 | 0.002 | 0.002 | 9.7 |
| 12x10 | maxstat | 0.050 +/- 0.013 | 0.000 | 0.000 | 33.0 |
| 12x10 | cluster | 0.050 +/- 0.013 | 0.000 | 0.000 | 33.0 |
| 12x10 | tfce | 0.040 +/- 0.011 | 0.000 | 0.000 | 33.0 |
| 24x20 | maxstat | 0.043 +/- 0.012 | 0.000 | 0.000 | 15.8 |
| 24x20 | cluster | 0.047 +/- 0.012 | 0.000 | 0.000 | 15.8 |
| 24x20 | tfce | 0.043 +/- 0.012 | 0.000 | 0.000 | 15.8 |
| 36x30 | maxstat | 0.033 +/- 0.010 | 0.000 | 0.000 | 16.7 |
| 36x30 | cluster | 0.037 +/- 0.011 | 0.000 | 0.000 | 16.7 |
| 36x30 | tfce | 0.057 +/- 0.013 | 0.000 | 0.000 | 16.7 |

C1 verdict: control passes. Crossed-intercept H0 remains nominal under the
Julia Wald full-refit path.

### C2: Random-Slope H0, Zero Mean

The first completed C2 cell, 6x5, hit the stop condition. It used
`n_sims=300`, `n_permutations=500`.

| Size | Backend | FWER +/- MC SE | Observed singular | Permuted singular | Mean s/sim |
| --- | --- | ---: | ---: | ---: | ---: |
| 6x5 | maxstat | 0.130 +/- 0.019 | 0.047 | 0.721 | 14.1 |
| 6x5 | cluster | 0.133 +/- 0.020 | 0.047 | 0.721 | 14.1 |
| 6x5 | tfce | 0.173 +/- 0.022 | 0.047 | 0.721 | 14.1 |

The run was interrupted after this completed C2 cell. A partial 12x10 C2 read
had only `n_sims=39` and is not used for the verdict:

| Size | Backend | FWER +/- MC SE | Observed singular | Permuted singular | Mean s/sim |
| --- | --- | ---: | ---: | ---: | ---: |
| 12x10 | maxstat | 0.103 +/- 0.049 | 0.000 | 0.652 | 55.4 |
| 12x10 | cluster | 0.103 +/- 0.049 | 0.000 | 0.652 | 55.4 |
| 12x10 | tfce | 0.128 +/- 0.054 | 0.000 | 0.652 | 55.4 |

C2 verdict: FAIL. Full-refit Wald permutation is inflated in the random-slope
case at the first completed size. The failure coincides with a very high
permuted singular-fit fraction, while C1 cells with near-zero singular fractions
remain nominal. This isolates Wald boundary behavior under permuted
random-slope fits as the likely culprit.

### C3: Effect Present

C3 was not run. Per the stop condition, once C2 was inflated the Wald refit
family should not be promoted to production; running C3 power cannot rescue a
non-calibrated null.

## Verdict

NO-GO for the full-refit Wald permutation-TFCE slope path.

The method passes crossed-intercept controls, but it does not calibrate the
random-slope H0. The inflation appears specifically in the singular-fit regime:
C2 6x5 has permuted singular fraction `0.721` and inflated FWER across all three
correction summaries, with TFCE worst at `0.173 +/- 0.022`.

Do not build a production Wald full-refit backend for random slopes from this
route. The next Route A candidate is LRT-in-loop, because it avoids relying on
the boundary-sensitive Wald standard error. If LRT is too slow or also
inflated, the remaining honest path is Route B / residual resampling.
