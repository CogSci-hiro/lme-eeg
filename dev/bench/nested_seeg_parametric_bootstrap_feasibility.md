# Nested sEEG Parametric Bootstrap Calibration Feasibility

Date: 2026-07-22

This is a Release 2 step 3 dev-only record. It introduces a nested
contacts-in-subjects calibration harness for the real sEEG design, but the full
requested `n_sims=300`, `n_boot=500`, `n_features=128` grid was not launched:
timing probes show it would be a multi-week local job.

No production backend was built.

## Harness

Script:

```text
dev/bench/calibrate_nested_seeg_parametric_bootstrap.py
```

Simulated design:

- subjects: configurable; smoke/timing used 6 and 15
- contacts: 5 per subject, nested and globally labeled as `subject:contact`
- trials: 40 per subject, balanced A/B within subject
- observations: subject x contact x trial
- features: 1D time map, default 128 timepoints
- C1: fixed condition effect zero, random slope variances zero
- C2: fixed condition effect zero, random subject slope present; random contact
  slope present only for R1
- C3: fixed condition effect present with the same slope structure as C2

Random-effect variants:

- R1: `y ~ 1 + cond + (1 + cond | subject) + (1 + cond | contact)`
- R2: `y ~ 1 + cond + (1 + cond | subject) + (1 | contact)`

Reduced models drop only fixed `cond` and keep the corresponding random-effects
structure. Fits use ML with the tight-convergence MixedModels.jl LRT settings
from `9c9af5b`.

Bootstrap scheme:

1. Fit the reduced null model per feature.
2. Simulate a response from the fitted reduced model with MixedModels.jl
   `simulate(MersenneTwister(seed), reduced)`.
3. Refit full and reduced ML models to the simulated response.
4. Compute unsigned LR maps.
5. Summarize maxstat, cluster, and TFCE. Maxstat is the primary decision metric.

## Guard And Smoke

The harness guard checks within-subject condition balance and fits a four-feature
C2 bootstrap setup for each requested size/variant before calibration. The smoke
used n=6/R1/C2 with four features and two bootstraps:

```text
CONTRACT (
  R1_full = "y ~ 1 + cond + :((1 + cond) | subject) + :((1 + cond) | contact)",
  R1_reduced = "y ~ 1 + :((1 + cond) | subject) + :((1 + cond) | contact)",
  R2_full = "y ~ 1 + cond + :((1 + cond) | subject) + :(1 | contact)",
  R2_reduced = "y ~ 1 + :((1 + cond) | subject) + :(1 | contact)",
  reml = false,
)
GUARD n_subjects=6 re_variant=R1 features=4 max_lr=2.546 singular=0.000
```

The smoke completed without non-finite maps or unresolved negative LR maps.

## Timing Probe

The timing grid used one C2 simulation per size/variant, 16 features, and
10 bootstraps:

```text
n_sims=1
n_boot=10
n_features=16
approx_fits_per_sim=352
```

Raw timing evidence is in:

```text
dev/bench/nested_seeg_parametric_bootstrap_timing_grid/
```

| Subjects | Variant | Probe sec/sim | Projected full sec/sim | Projected hours/cell, 4 workers |
| --- | --- | ---: | ---: | ---: |
| 6 | R1 | 9.879 | 3599 | 75.0 |
| 6 | R2 | 0.881 | 321 | 6.7 |
| 15 | R1 | 4.272 | 1557 | 32.4 |
| 15 | R2 | 1.392 | 507 | 10.6 |

Projection factor:

```text
(128 features * 501 maps * 2 fits) / (16 features * 11 maps * 2 fits)
= 364.36
```

The requested full grid has 12 cells:

```text
2 subject sizes * 2 RE variants * 3 scenarios
```

Summing the projected cell times gives roughly 374 worker-wall hours on a
4-worker local run, about 15-16 days, before accounting for long-tail optimizer
fits. The n=6/R1 cells dominate.

## Stop Decision

The full `n_sims>=300`, `n_boot>=500`, `n_features=128` table was not run in
this Codex session. Launching it would tie up the machine for a multi-week job
and still not produce a useful answer in-turn.

This is not a method failure and not a calibration result. It is a feasibility
stop with a runnable, seeded harness and timing evidence.

## Full Command

Exact full-grid command:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_nested_seeg_parametric_bootstrap.py \
  --n-subjects 6 15 \
  --re-variants R1 R2 \
  --scenarios C1 C2 C3 \
  --n-sims 300 \
  --n-boot 500 \
  --n-features 128 \
  --contacts-per-subject 5 \
  --trials-per-subject 40 \
  --workers 4 \
  --output-dir dev/bench/nested_seeg_parametric_bootstrap_full \
  --progress-every 1
```

Recommended next move: run a narrower decisive slice first, for example
R2/maxstat-focused C2 at n=6 and n=15, or run the full grid on a batch system
with cell-level parallelization.
