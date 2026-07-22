# MixedModels.jl LRT Residual-Permutation Power Frontier

Date: 2026-07-21

This is a dev-only Release 2 Route A step 2e record. It tunes the
residual-permutation scheme from step 2d for power while checking whether the
C2 random-slope null remains calibrated. No production backend was built.

## Guard

The random-slope H0 guard passed on the current generator/code before these
calibration cells were trusted:

```text
LMEEG_PROGRESS_EVERY=25 RPY2_CFFI_MODE=ABI \
R_LIBS_USER=/Users/hiro/Projects/active/lme-eeg/.venv/R/library \
.venv/bin/python -m pytest \
  tests/backends/lmm/test_real_lmm_null_calibration_slow.py::\
test_real_lmm_slope_h0_guard_passes_before_refit_calibration \
  -q -s -m slow

1 passed in 638.34 s
```

## Variants

All variants keep condition labels fixed and use the same full-refit LRT
statistic for observed and permuted maps. Full and reduced models are fit by ML
with the tight-convergence retry path from the LRT design-permutation work.

V1: marginal same-slope residuals.

- Residual source:
  `y ~ 1 + (1 + cond | subject) + (1 | item)`.
- Residuals:
  `y - X beta_reduced`, leaving random effects in the residual.
- Permutation:
  within-subject shuffle of marginal residuals.
- Exchangeability argument:
  this is the canonical Freedman-Lane residual idea adapted to the mixed model;
  condition stays fixed and observation-level residual material moves only
  within subject. The crossed-item limitation remains.

V2: marginal no-slope residuals.

- Residual source:
  `y ~ 1 + (1 | subject) + (1 | item)`.
- Test model:
  unchanged full random-slope LRT,
  `y ~ 1 + cond + (1 + cond | subject) + (1 | item)` versus the matching
  fixed-effect-reduced model.
- Residuals:
  `y - X beta_reduced`, leaving random effects in the residual.
- Permutation:
  within-subject shuffle of marginal residuals.
- Exchangeability argument:
  the residual source no longer lets the reduced random-slope term absorb
  condition-like variation before permutation; the full random-slope model is
  still used for the test statistic.

V3: subject sign-flip of marginal same-slope residuals.

- Residual source:
  same as V1.
- Residuals:
  `y - X beta_reduced`.
- Permutation:
  one random sign per subject applied to that subject's residual vector.
- Exchangeability argument:
  if the residualized within-subject contrast errors are symmetric around zero,
  subject-level sign flips preserve the paired/within-subject structure more
  directly than label shuffling. This is a stronger symmetry assumption and is
  checked by the C1/C2 controls.

## Decision Rule

The hard null constraint is all-backend C2 and C1 nominal behavior. With
`n_sims=300`, a conservative three-SE nominal bound is approximately:

```text
0.05 + 3 * sqrt(0.05 * 0.95 / 300) = 0.088
```

For a production GO at 6x5, a variant must keep C2 and C1 at or below this
bound for maxstat, cluster, and TFCE, and must recover C3 power clearly above
alpha. No variant met all three requirements.

## Results

Settings for all cells:

- 6 subjects x 5 items
- four features
- `n_sims=300`
- `n_permutations=500`
- MixedModels.jl full-refit LRT by ML
- maxstat, cluster, and TFCE summaries from the shared LR maps

Rates are `maxstat / cluster / TFCE`; MC SEs are in the same order.

| Variant | Scenario | Rates | MC SEs | Verdict |
| --- | --- | --- | --- | --- |
| 2d conditional same-slope | C1 H0 | 0.047 / 0.057 / 0.050 | 0.012 / 0.013 / 0.013 | nominal |
| 2d conditional same-slope | C2 H0 | 0.003 / 0.017 / 0.007 | 0.003 / 0.007 / 0.005 | conservative |
| 2d conditional same-slope | C3 effect | 0.000 / 0.027 / 0.013 | 0.000 / 0.009 / 0.007 | underpowered |
| V1 marginal same-slope | C1 H0 | 0.047 / 0.057 / 0.053 | 0.012 / 0.013 / 0.013 | nominal |
| V1 marginal same-slope | C2 H0 | 0.113 / 0.113 / 0.150 | 0.018 / 0.018 / 0.021 | inflated |
| V1 marginal same-slope | C3 effect | 0.673 / 0.683 / 0.757 | 0.027 / 0.027 / 0.025 | powered |
| V2 marginal no-slope | C1 H0 | 0.047 / 0.057 / 0.053 | 0.012 / 0.013 / 0.013 | nominal |
| V2 marginal no-slope | C2 H0 | 0.113 / 0.113 / 0.150 | 0.018 / 0.018 / 0.021 | inflated |
| V2 marginal no-slope | C3 effect | 0.673 / 0.683 / 0.757 | 0.027 / 0.027 / 0.025 | powered |
| V3 subject sign-flip | C1 H0 | 0.377 / 0.153 / 0.400 | 0.028 / 0.021 / 0.028 | inflated |
| V3 subject sign-flip | C2 H0 | 0.060 / 0.097 / 0.107 | 0.014 / 0.017 / 0.018 | partly inflated |
| V3 subject sign-flip | C3 effect | 0.410 / 0.620 / 0.610 | 0.028 / 0.028 / 0.028 | powered |

No unresolved negative LR maps occurred in any cell. Per-fit singular fractions
were variant- and scenario-dependent; the most salient pattern is that
less-conservative marginal residual variants had substantially higher permuted
singular fractions than the conditional baseline, especially for C1.

## Power Ranking Among Null-Valid Variants

No tuned variant is null-valid across C1 and C2 for all three backends, so there
is no admissible power ranking.

For the frontier:

1. Conditional same-slope residuals are null-valid but underpowered
   (`C3 = 0.000 / 0.027 / 0.013`).
2. V1 and V2 recover strong C3 power
   (`0.673 / 0.683 / 0.757`) and keep C1 nominal, but C2 inflates to
   `0.113 / 0.113 / 0.150`.
3. V3 has moderate-to-strong C3 power (`0.410 / 0.620 / 0.610`) and improves C2
   relative to V1/V2, but still fails C2 for cluster/TFCE and badly breaks C1.

## Verdict

No tested residual-permutation variant achieves both calibrated null behavior
and usable power at the 6x5 random-slope calibration cell. The result is a real
small-design frontier:

- conditional residuals protect the null but are too conservative,
- marginal residuals recover power but reintroduce C2 inflation,
- subject sign-flip recovers some power but is not a valid C1/C2 null scheme.

Do not build a production random-slope permutation backend from these variants.
The honest next options are:

1. test larger minimum design sizes to see whether the frontier closes with more
   subjects/items,
2. restrict random-slope permutation inference to explicit ROI/window scale with
   a design-specific calibration requirement, or
3. use the step-2A parametric per-feature LMM statistic with a non-permutation
   multiple-comparison correction.

## Reproduction Pattern

Each variant used the same command shape, changing `--permutation-method`,
`--scenarios`, and `--output-dir`:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_mixedmodels_lrt_refit.py \
  --skip-guard --permutation-method marginal_same_slope \
  --n-sims 300 --n-permutations 500 --workers 4 \
  --scenarios C2 --sizes 6x5 \
  --output-dir dev/bench/mixedmodels_lrt_residual_v1_marginal_same_slope_c2_6x5 \
  --progress-every 25
```

The tested `--permutation-method` values were `marginal_same_slope`,
`marginal_no_slope`, and `subject_signflip`.
