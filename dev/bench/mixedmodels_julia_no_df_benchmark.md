# MixedModels.jl No-DF Refit Benchmark

Date: 2026-07-21

Purpose: Release 2 Route A step 1b. Re-benchmark the full-refit
MixedModels.jl engine without per-permutation Satterthwaite/Kenward-Roger
degrees-of-freedom machinery. This is still a dev/calibration artifact only;
no production backend or pipeline wiring was added.

The tested in-loop statistics are permutation statistics, not analytic
p-values:

- Wald/z-style signed statistic: `fixef(model)[cond] / stderror(model)[cond]`
  from the default REML MixedModels fit.
- Likelihood-ratio statistic: `2 * (logLik(full ML) - logLik(reduced ML))`.
  This requires two ML fits per feature/permutation and is unsigned.

The permutation null would provide calibration, so neither statistic applies
Satterthwaite or Kenward-Roger inside the loop.

## Command

```bash
LMEEG_JULIA_BENCH_REPEATS=10 \
LMEEG_JULIA_BENCH_STATS=wald,lrt \
PYTHON_JULIACALL_EXE=/Users/hiro/Projects/active/lme-eeg/.venv/julia_env/pyjuliapkg/install/bin/julia \
PYTHON_JULIACALL_PROJECT=/Users/hiro/Projects/active/lme-eeg/dev/julia/mixedmodels_bench \
RPY2_CFFI_MODE=ABI \
R_LIBS_USER=/Users/hiro/Projects/active/lme-eeg/.venv/R/library \
.venv/bin/python dev/bench/benchmark_mixedmodels_julia.py
```

One-time Julia/juliacall load in the measured process:

```text
JULIA_LOAD_SECONDS 11.586
```

## Statistic Correctness

The script keeps the previous pymer4/lmerTest oracle check for the
Satterthwaite statistic, then separately checks the no-df Wald surface.

Satterthwaite oracle agreement:

| Model | Max abs t-difference vs pymer4/lmerTest |
| --- | ---: |
| Crossed intercepts | 2.54470213e-06 |
| Random slope | 1.61079073e-05 |

Wald sanity checks:

| Model | Max abs `fixef/stderror - coeftable z` | Spearman rank of abs Wald vs abs Satterthwaite | Max abs Wald-Satterthwaite |
| --- | ---: | ---: | ---: |
| Crossed intercepts | 0 | 1.0 | 0 |
| Random slope | 0 | 1.0 | 0 |

In this MixedModels.jl surface the default coefficient-table `z` is exactly
`fixef / stderror`, and the observed-data Wald t ordering is identical to the
Satterthwaite t ordering on the checked features. The small-sample adjustment
changes the analytic df/p-value path, not the tested t value in this benchmark.

## Throughput

Timings below include Python-to-Julia marshalling plus MixedModels fitting and
statistic extraction. They exclude JuliaPkg download/precompilation and separate
the first warm fit from steady-state repeats.

### Wald

| Model | Size | Repeats | Warm fit s | Mean fit s | Median fit s | Min fit s | Max fit s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crossed intercepts | 6x5 | 10 | 0.521876 | 0.001260 | 0.001142 | 0.000817 | 0.002172 |
| Random slope | 6x5 | 10 | 0.017606 | 0.002615 | 0.002374 | 0.001776 | 0.003813 |
| Crossed intercepts | 12x10 | 10 | 0.002666 | 0.001802 | 0.001876 | 0.000936 | 0.002940 |
| Random slope | 12x10 | 10 | 0.010549 | 0.011179 | 0.008010 | 0.006046 | 0.025937 |
| Crossed intercepts | 24x20 | 10 | 0.535579 | 0.008121 | 0.007432 | 0.005436 | 0.014829 |
| Random slope | 24x20 | 10 | 0.028047 | 0.006233 | 0.006000 | 0.004869 | 0.008645 |
| Crossed intercepts | 36x30 | 10 | 0.280360 | 0.004169 | 0.004125 | 0.002661 | 0.006114 |
| Random slope | 36x30 | 10 | 0.026066 | 0.017918 | 0.010632 | 0.008010 | 0.087365 |

### LRT

LRT fits full and reduced ML models, so each reported statistic performs two
MixedModels fits. On these small generated designs its wall time was still
comparable to Wald, but the doubled fit count is inherent.

| Model | Size | Repeats | Warm fit s | Mean statistic s | Median statistic s | Min statistic s | Max statistic s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crossed intercepts | 6x5 | 10 | 1.210406 | 0.001792 | 0.001267 | 0.000894 | 0.006620 |
| Random slope | 6x5 | 10 | 0.120830 | 0.004708 | 0.004244 | 0.003538 | 0.006746 |
| Crossed intercepts | 12x10 | 10 | 0.042944 | 0.003000 | 0.003000 | 0.001987 | 0.004474 |
| Random slope | 12x10 | 10 | 0.011071 | 0.006632 | 0.005604 | 0.004392 | 0.012263 |
| Crossed intercepts | 24x20 | 10 | 0.055155 | 0.028471 | 0.007921 | 0.005354 | 0.212731 |
| Random slope | 24x20 | 10 | 0.012746 | 0.007193 | 0.006595 | 0.005305 | 0.010640 |
| Crossed intercepts | 36x30 | 10 | 0.046296 | 0.006739 | 0.006745 | 0.006105 | 0.007891 |
| Random slope | 36x30 | 10 | 0.012728 | 0.014575 | 0.014255 | 0.012446 | 0.019494 |

## Projection

Using the 36x30 random-slope mean:

| Statistic | 36x30 random-slope mean s/stat | Calibration grid | Real EEG example |
| --- | ---: | ---: | ---: |
| Wald | 0.017918 | 7,200,000 feature-statistics = 1.49 serial days | 32,000,000 feature-statistics = 6.64 serial days |
| LRT | 0.014575 | 7,200,000 feature-statistics = 1.21 serial days | 32,000,000 feature-statistics = 5.40 serial days |
| Prior Satterthwaite | 31.338185 | 7,200,000 feature-statistics = 2,611.52 serial days | 32,000,000 feature-statistics = 11,606.74 serial days |

Relative to the prior Satterthwaite step-1 timing, the no-df Wald path is about
`31.338185 / 0.017918 = 1,749x` faster by the 36x30 random-slope mean. The LRT
path is about `2,150x` faster by the measured mean, despite requiring two fits;
that surprising result should not be over-interpreted beyond these small
calibration designs.

## Verdict

GO for the engine-only Wald refit architecture at calibration scale.

The step-1 NO-GO was driven by per-fit Satterthwaite machinery, not by Julia
fit or juliacall marshalling cost. Without small-sample df in the permutation
loop, the full calibration grid is projected at roughly 1.5 serial days before
parallelism, which is tolerable on a workstation or modest parallel runner.

For real EEG, 32,000 features x 1,000 permutations is still about 6.6 serial
days for Wald at the measured 36x30 random-slope mean. That is not instant, but
it is practical with feature/permutation parallelism or an ROI/time-window
scope. Dense whole-map refits remain expensive enough that production work
should still expose fit-count/runtime estimates before running.

Recommended in-loop statistic for Route A is Wald t:

- It is signed, directly tied to the condition coefficient, and cheap.
- It is computed identically for observed and permuted data.
- Its observed feature ordering matched the Satterthwaite t in the oracle
  check.

LRT remains a viable secondary statistic if an unsigned omnibus condition test
is desired, but it doubles the conceptual fit count and changes the statistic.
