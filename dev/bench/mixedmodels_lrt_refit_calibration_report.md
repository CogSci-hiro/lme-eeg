# MixedModels.jl LRT-Refit Permutation Calibration

Date: 2026-07-21

Purpose: Release 2 Route A step 2b. Test whether replacing the Wald in-loop
statistic with a likelihood-ratio statistic fixes the random-slope C2
inflation seen in `9fb2aab`.

This is a dev/calibration record only. No production backend was built or wired
into the real pipeline.

## Engine Contract

The LRT runner fits full and reduced MixedModels.jl models by ML for every
observed and within-subject-permuted feature:

```text
full random-slope:    y ~ 1 + cond + (1 + cond | subject) + (1 | item)
reduced random-slope: y ~ 1        + (1 + cond | subject) + (1 | item)
full crossed-int:     y ~ 1 + cond + (1 | subject) + (1 | item)
reduced crossed-int:  y ~ 1        + (1 | subject) + (1 | item)
REML: false
```

The reduced random-slope model drops only the tested fixed condition effect and
keeps the full random-effects structure, including the subject random slope.

The LR statistic is unsigned:

```text
2 * (logLik(full ML) - logLik(reduced ML))
```

Correction uses one-sided/non-negative maps, matching the existing nested-block
F correction pattern rather than the signed-t absolute-value path:

- maxstat: max raw LR over the map, no absolute value.
- cluster: MNE `_find_clusters(..., tail=1)`, threshold `4.0`.
- TFCE: MNE `_find_clusters(..., tail=1)`, threshold
  `{"start": 0.0, "step": 0.2, "h_power": 2.0, "e_power": 0.5}`.

Singular status is recorded for every full and reduced fit. Fits are not
dropped or zeroed.

## Guard

The random-slope H0 null guard passed before any LRT FWER result was trusted:

```text
1 passed, 800 warnings in 374.71s (0:06:14)
```

## Head-to-Head C2 6x5 Run

Command:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
LMEEG_JULIA_LRT_WORKERS=8 \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_mixedmodels_lrt_refit.py \
  --skip-guard \
  --print-contract \
  --n-sims 300 \
  --n-permutations 500 \
  --workers 8 \
  --scenarios C2 \
  --sizes 6x5 \
  --output-dir dev/bench/mixedmodels_lrt_refit_c2_6x5 \
  --progress-every 25
```

The 8-worker run stalled after 16 completed simulations in boundary-heavy ML
fits. A 4-worker resume advanced only to 20 rows before the same tail behavior.
A single-worker resume made steady progress to 42 completed rows, then failed
loudly on a non-negligible negative LR:

```text
RuntimeError: Negative LR statistic -0.00024349955464231243 at feature 0,1.
```

Because the full and reduced models are nested and both are ML fits, the LR
statistic should be non-negative. Tiny floating-point negatives are tolerated
up to `1e-6`; `-2.4e-4` is too large to treat as roundoff. Clipping it to zero
would silently change the statistic in the exact boundary regime under test, so
the calibration was stopped.

Partial C2 6x5 checkpoint, not a final FWER estimate:

| Backend | n_sims | FWER +/- MC SE | Observed singular | Permuted singular | Mean s/sim |
| --- | ---: | ---: | ---: | ---: | ---: |
| maxstat | 42 | 0.071 +/- 0.040 | 0.068 | 0.754 | 15.3 |
| cluster | 42 | 0.071 +/- 0.040 | 0.068 | 0.754 | 15.3 |
| tfce | 42 | 0.095 +/- 0.045 | 0.068 | 0.754 | 15.3 |

Wald head-to-head from `9fb2aab`, completed at n=300:

| Backend | Wald C2 6x5 FWER +/- MC SE | Wald observed singular | Wald permuted singular |
| --- | ---: | ---: | ---: |
| maxstat | 0.130 +/- 0.019 | 0.047 | 0.721 |
| cluster | 0.133 +/- 0.020 | 0.047 | 0.721 |
| tfce | 0.173 +/- 0.022 | 0.047 | 0.721 |

The partial LRT rates look lower than Wald, but the run did not reach an
interpretable n and cannot be accepted because the engine failed to produce a
valid non-negative LR statistic.

## Verdict

NO-GO for LRT-in-loop as implemented through MixedModels.jl full refits.

The failure mode differs from Wald:

- Wald completed but inflated C2 in the high-singular regime.
- LRT avoided the Wald SE but could not reliably produce valid nested ML LR
  statistics in that same high-singular regime; full/reduced optimizer behavior
  produced `logLik(full) < logLik(reduced)` by more than numerical tolerance.

The refit-family candidate is therefore exhausted for this engine footprint.
The remaining honest options are Route B / residual resampling, or constraining
random-slope TFCE to a much smaller ROI/window with a separately validated,
more robust optimizer strategy.
