# MixedModels.jl Tight-LRT Refit Calibration

Date: 2026-07-21

Purpose: Release 2 Route A step 2c. Retry LRT-in-loop after addressing the
negative-LR numerical problem from `6ac7957`.

This is a dev/calibration record only. No production backend was built or wired
into the real pipeline.

## Convergence Fix

The LRT runner now fits both full and reduced nested models by ML with stricter
MixedModels.jl/NLopt settings before computing:

```text
LR = 2 * (logLik(full ML) - logLik(reduced ML))
```

Primary fit settings:

- optimizer/backend: `LN_BOBYQA` / `:nlopt`
- `REML=false` for both full and reduced fits
- `ftol_rel = 1e-14`
- `ftol_abs = 1e-10`
- `xtol_rel = 0`
- `xtol_abs = 1e-12`
- `maxfeval = 10000`
- `xtol_zero_abs = 1e-8`
- `ftol_zero_abs = 1e-10`

If an LR is below `-1e-8`, the runner records a convergence-refit trigger and
refits that feature pair using alternate optimizer/start combinations:

- `LN_BOBYQA`, initial scale `0.5`
- `LN_NEWUOA`, initial scale `1.0`
- `LN_NELDERMEAD`, initial scale `1.0`
- `LN_NELDERMEAD`, initial scale `0.5`
- `LN_COBYLA`, initial scale `1.0`

Meaningful negative LR after all retries remains a hard failure. Nothing is
silently clipped. The final decisive run did not trigger any convergence refits.

## Guard

The random-slope H0 null guard passed before the C2 FWER run:

```text
1 passed, 800 warnings in 374.70s (0:06:14)
```

## Engine Contract

The runtime contract was printed before the decisive cell:

```text
full random-slope:    y ~ 1 + cond + :((1 + cond) | subject) + :(1 | item)
reduced random-slope: y ~ 1 + :((1 + cond) | subject) + :(1 | item)
full crossed-int:     y ~ 1 + cond + :(1 | subject) + :(1 | item)
reduced crossed-int:  y ~ 1 + :(1 | subject) + :(1 | item)
reml = false
```

The reduced random-slope model drops only the tested fixed condition effect and
keeps the full random-effects structure.

## Decisive C2 6x5 Cell

Command:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_mixedmodels_lrt_refit.py \
  --skip-guard \
  --print-contract \
  --n-sims 300 \
  --n-permutations 500 \
  --workers 4 \
  --scenarios C2 \
  --sizes 6x5 \
  --output-dir dev/bench/mixedmodels_lrt_refit_c2_6x5_tight \
  --progress-every 25
```

Final result, `n_sims=300`, `n_permutations=500`:

| Backend | FWER +/- MC SE | Observed singular | Permuted singular | Observed refit-trigger | Permuted refit-trigger | Unresolved negative maps | Mean s/sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxstat | 0.123 +/- 0.019 | 0.001 | 0.026 | 0.000 | 0.000 | 0 | 14.5 |
| cluster | 0.127 +/- 0.019 | 0.001 | 0.026 | 0.000 | 0.000 | 0 | 14.5 |
| tfce | 0.167 +/- 0.022 | 0.001 | 0.026 | 0.000 | 0.000 | 0 | 14.5 |

The run completed without negative-LR failures. The tighter optimizer also
reduced the high permuted-singular rate seen in step 2b from about `0.75` to
`0.026`.

## Comparison

Same C2 6x5 stress cell:

| Statistic | maxstat | cluster | tfce | Permuted singular | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Wald, `9fb2aab` | 0.130 +/- 0.019 | 0.133 +/- 0.020 | 0.173 +/- 0.022 | 0.721 | completed, inflated |
| LRT partial, `6ac7957` | 0.071 +/- 0.040 | 0.071 +/- 0.040 | 0.095 +/- 0.045 | 0.754 | stopped at n=42 after negative LR |
| Tight LRT, this run | 0.123 +/- 0.019 | 0.127 +/- 0.019 | 0.167 +/- 0.022 | 0.026 | completed, inflated |

The 2b partial rates were a small-n mirage. Once the numerical issue was fixed
and the cell completed at final precision, LRT landed essentially on the Wald
inflation.

## Verdict

NO-GO for full-refit LRT-in-loop on the random-slope C2 path.

This run addresses the optimizer-convergence concern from step 2b:

- no unresolved negative LR maps;
- no convergence-refit triggers in the final cell;
- low singular fractions under tight fitting.

Despite that, the final C2 6x5 FWER is clearly inflated for all three
correction summaries, and TFCE is worst at `0.167 +/- 0.022`. This is above the
requested PARTIAL band (`~0.08-0.10`) and not within Monte Carlo error of
nominal.

The refit-family path is therefore genuinely exhausted for this engine footprint.
The remaining options are Route B / residual resampling or an explicitly scoped
ROI/window analysis with a separate validation target.
