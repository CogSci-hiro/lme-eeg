# MixedModels.jl Engine-Only Benchmark

Date: 2026-07-21

Purpose: Release 2 Route A step 1 go/no-go gate for using Julia
MixedModels.jl through juliacall as a per-permutation LMM fitting engine. This
is engine-only: Julia fits one dataset and returns the condition fixed-effect
t statistic. Python remains responsible for simulation, permutation, TFCE, and
correction logic.

## Phase 0 Provisioning

Phase 0 is provisionable and reproducible.

Python side:

- `pyproject.toml` declares optional extra `julia = ["juliacall==0.9.35"]`.
- `juliacall` uses JuliaPkg for Julia/PythonCall provisioning.

Julia side:

- Managed Julia executable provisioned by JuliaPkg:
  `.venv/julia_env/pyjuliapkg/install/bin/julia`
- Julia version: 1.11.9
- Committed benchmark project:
  `dev/julia/mixedmodels_bench/Project.toml`
- Committed benchmark manifest:
  `dev/julia/mixedmodels_bench/Manifest.toml`

Pinned direct Julia dependencies:

- DataFrames 1.8.2
- MixedModels 5.7.1
- MixedModelsSmallSample 0.1.0
- PythonCall 0.9.35
- StatsModels 0.7.10
- Tables 1.13.0

MixedModels.jl itself reports Wald z-style fixed-effect inference by default,
not lmerTest-style Satterthwaite inference. The benchmark therefore uses
MixedModelsSmallSample.jl `small_sample_adjust(..., Satterthwaite())` so the
Julia engine returns the same statistic family as pymer4/lmerTest.

## Phase 1 Correctness Oracle

Command:

```bash
LMEEG_JULIA_BENCH_REPEATS=3 \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
RPY2_CFFI_MODE=ABI \
R_LIBS_USER=/Users/hiro/Projects/active/lme-eeg/.venv/R/library \
.venv/bin/python dev/bench/benchmark_mixedmodels_julia.py
```

Oracle agreement compares MixedModels.jl + MixedModelsSmallSample Satterthwaite
condition t against pymer4/lmerTest condition t on the same generated data and
same REML model.

| Model | Max abs t-difference |
| --- | ---: |
| Crossed intercepts | 2.54470213e-06 |
| Random slope | 1.61079073e-05 |

Phase 1 verdict: PASS. The engines agree on the tested statistic to the
benchmark tolerance (`1e-4`).

## Phase 2 Throughput

One-time juliacall/JIT load in the measured process:

```text
JULIA_LOAD_SECONDS 7.242
```

Steady-state timings include Python-to-Julia marshalling plus one MixedModels.jl
fit and one Satterthwaite adjustment. They exclude the one-time JuliaPkg
download and package precompilation.

| Model | Size | Repeats | Warm fit s | Mean fit s | Median fit s | Min fit s | Max fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crossed intercepts | 6x5 | 3 | 0.204615 | 0.003779 | 0.002761 | 0.002211 | 0.006367 |
| Random slope | 6x5 | 3 | 0.015324 | 0.005654 | 0.005605 | 0.004899 | 0.006457 |
| Crossed intercepts | 12x10 | 3 | 0.022777 | 0.020877 | 0.020583 | 0.020516 | 0.021533 |
| Random slope | 12x10 | 3 | 0.053121 | 0.100321 | 0.053274 | 0.051755 | 0.195935 |
| Crossed intercepts | 24x20 | 3 | 1.287662 | 1.099535 | 1.098530 | 1.093115 | 1.106958 |
| Random slope | 24x20 | 3 | 2.781795 | 2.814534 | 2.853397 | 2.733122 | 2.857084 |
| Crossed intercepts | 36x30 | 3 | 11.599089 | 11.413638 | 11.314621 | 11.185418 | 11.740876 |
| Random slope | 36x30 | 3 | 30.052800 | 31.338185 | 31.501263 | 30.747953 | 31.765338 |

Projection using the 36x30 random-slope mean (`31.338185 s/fit`):

- Full calibration grid:
  `C1/C2/C3 * 4 sizes * 300 sims * 500 permutations * 4 features`
  = 7,200,000 fits = 2,611.52 days serial.
- Real-analysis example:
  `64 channels * 500 time points * 1000 permutations`
  = 32,000,000 fits = 11,606.74 days serial.

The 6x5 and 12x10 timings show that juliacall marshalling is not the dominant
cost at small sizes; MixedModels.jl fit/Satterthwaite cost dominates as the
number of observations and random effects grows.

## Verdict

NO-GO for the engine-only architecture.

Provisioning works and the statistic agrees with pymer4/lmerTest, but one
Julia call per feature per permutation is not fast enough at the validated
design sizes. The 36x30 random-slope fit takes about 31 seconds steady-state,
making both the full calibration grid and a realistic single EEG permutation
analysis impractical.

This does not rule out Julia as an engine if the footprint is widened. The next
route would have to push more work into Julia, such as batching many features
and/or running the permutation loop inside Julia, so model setup and
Satterthwaite machinery are amortized. That is outside this benchmark gate and
requires a separate design decision.
