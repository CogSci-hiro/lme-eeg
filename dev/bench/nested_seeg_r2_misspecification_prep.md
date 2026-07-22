# Nested sEEG R2 Misspecification Calibration Prep

Date: 2026-07-22

This is a Release 2 step 3b prep record. The nested sEEG harness now supports
the requested misspecification test:

- generator: subject condition-slope variance plus contact condition-slope
  variance;
- fitted model: R2,
  `y ~ 1 + cond + (1 + cond | subject) + (1 | contact)`;
- reduced bootstrap model: fixed `cond` dropped, R2 random structure retained.

The full calibration was not launched because the required real-data
contact-slope SD was not supplied in the prompt and is not recorded in the repo.
The value must not be invented: it defines the misspecification magnitude.

## Guard

The harness now requires a positive contact-slope SD for R2 misspecification
tests. A zero value fails before any calibration can run:

```text
RuntimeError: R2 misspecification guard failed: --contact-slope-sd must be > 0
so the generator includes omitted contact-level condition slopes.
```

This protects against accidentally re-running the correctly specified R2
generator from step 3.

## Smoke

A tiny mechanics-only smoke with `--contact-slope-sd 0.25`, four features, and
two bootstraps passed. This was not treated as the observed-magnitude
calibration.

Contract:

```text
R2_full = "y ~ 1 + cond + :((1 + cond) | subject) + :(1 | contact)"
R2_reduced = "y ~ 1 + :((1 + cond) | subject) + :(1 | contact)"
reml = false
```

Smoke guard:

```text
GUARD n_subjects=6 re_variant=R2 subject_slope_sd=0.400
contact_slope_sd=0.250 features=4 max_lr=3.332 singular=0.000
```

## Commands Once Contact-Slope SD Is Supplied

Set the measured real-data contact-slope SD:

```bash
CONTACT_SLOPE_SD=<real-data-contact-slope-sd>
```

Decisive observed-magnitude C2 cell:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_nested_seeg_parametric_bootstrap.py \
  --n-subjects 6 \
  --re-variants R2 \
  --scenarios C2 \
  --n-sims 150 \
  --n-boot 500 \
  --n-features 128 \
  --contacts-per-subject 5 \
  --trials-per-subject 40 \
  --subject-slope-sd 0.40 \
  --contact-slope-sd "${CONTACT_SLOPE_SD}" \
  --workers 4 \
  --output-dir dev/bench/nested_seeg_r2_misspec_c2_observed \
  --progress-every 1
```

Margin check at 2x contact-slope SD:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_nested_seeg_parametric_bootstrap.py \
  --n-subjects 6 \
  --re-variants R2 \
  --scenarios C2 \
  --n-sims 150 \
  --n-boot 500 \
  --n-features 128 \
  --contacts-per-subject 5 \
  --trials-per-subject 40 \
  --subject-slope-sd 0.40 \
  --contact-slope-sd "$(python -c 'import os; print(2 * float(os.environ[\"CONTACT_SLOPE_SD\"]))')" \
  --workers 4 \
  --output-dir dev/bench/nested_seeg_r2_misspec_c2_2x \
  --progress-every 1
```

If observed-magnitude C2 maxstat is nominal, run C1/C3 with the same observed
contact-slope SD:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
.venv/bin/python dev/bench/calibrate_nested_seeg_parametric_bootstrap.py \
  --n-subjects 6 \
  --re-variants R2 \
  --scenarios C1 C3 \
  --n-sims 150 \
  --n-boot 500 \
  --n-features 128 \
  --contacts-per-subject 5 \
  --trials-per-subject 40 \
  --subject-slope-sd 0.40 \
  --contact-slope-sd "${CONTACT_SLOPE_SD}" \
  --workers 4 \
  --output-dir dev/bench/nested_seeg_r2_misspec_c1_c3_observed \
  --progress-every 1
```
