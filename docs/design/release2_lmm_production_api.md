# Release 2 Full-LMM Production API Design

Date: 2026-07-22

Status: Phase 0 proposal for review. Do not implement Phase 1 until this API
shape is reviewed.

## Goal

Release 2 should ship a generic, reusable full-LMM inference path alongside the
existing fast marginal-OLS path. This is a method implementation and validation
tooling release, not a claim that every mixed model or study design is already
calibrated. Calibration remains design-dependent.

The implementation target is:

- MixedModels.jl through `juliacall`, engine-only from Python.
- ML full/reduced mixed-model fits for an unsigned likelihood-ratio statistic.
- Parametric bootstrap from the fitted reduced null model.
- Max-stat correction as the default family-wise correction.
- Cluster and TFCE available as explicitly unvalidated secondary options for
  random-slope analyses.

The Release 1 fast path remains the default and remains unchanged for existing
users.

## User Selection

Add a `null_method` selector to `FitConfig`:

```python
FitConfig(
    null_method="ols",  # default: existing fast marginal-OLS path
)

FitConfig(
    null_method="lmm",  # full-LMM refit + parametric bootstrap path
)
```

Semantics:

- `null_method="ols"` uses the existing LMM-marginalisation followed by mass OLS
  and the existing permutation/correction machinery. It is the backward-
  compatible default.
- `null_method="lmm"` uses a full mixed-model statistic and a parametric
  bootstrap family-wise null. It is opt-in and additive.

The existing public functions remain reachable:

- `fit_lmm_mass_univariate(...)` still constructs and validates the design.
- `permute_fixed_effect(...)` should grow into a more general inference entry
  point without breaking current calls. The name can stay for compatibility, but
  the implementation should dispatch based on `fit_result.backend_metadata` or
  an explicit inference config.

Open naming point for review:

- Keep the public function name `permute_fixed_effect` even when the LMM path is
  bootstrap-based, and document that it means "family-wise inference" for
  backward compatibility.
- Or add `infer_fixed_effect(...)` as the clearer new API while keeping
  `permute_fixed_effect(...)` as a thin compatibility wrapper for
  `null_method="ols"`.

Recommendation: add `infer_fixed_effect(...)` and keep
`permute_fixed_effect(...)` as a compatibility alias/wrapper. The LMM path is not
permutation inference, and the name should not mislead users.

## Configuration Surface

Extend `FitConfig` with method-level knobs:

```python
@dataclass(slots=True)
class FitConfig:
    null_method: Literal["ols", "lmm"] = "ols"
    lmm_engine: Literal["mixedmodels_jl"] = "mixedmodels_jl"
    lmm_statistic: Literal["lrt"] = "lrt"
    lmm_n_bootstrap: int = 1000
    lmm_seed: int = 0
    lmm_n_jobs: int = 1
    lmm_shard_index: int | None = None
    lmm_n_shards: int | None = None
    lmm_output_dir: Path | str | None = None
    lmm_store_null_maps: bool = False
    lmm_fail_on_singular: bool = False
```

Rationale:

- `lmm_n_bootstrap` and `lmm_seed` make the bootstrap null reproducible.
- `lmm_n_jobs` is process-level parallelism only. Julia and R bridges should not
  be treated as thread-safe.
- `lmm_shard_index`, `lmm_n_shards`, and `lmm_output_dir` make long ROI/feature
  jobs Snakemake-friendly and resumable.
- `lmm_store_null_maps` defaults to `False` because full null maps can be large.
- `lmm_fail_on_singular=False` records singular fits as diagnostics without
  dropping them. Singular fits are common and informative.

Extend inference configuration with correction-specific knobs:

```python
@dataclass(slots=True)
class InferenceConfig:
    correction: Literal["maxstat", "cluster", "tfce"] = "maxstat"
    threshold: float | dict[str, float] | None = None
    adjacency: Any | None = None
    tail: int = 1
    verbose: bool | str | int | None = "info"
```

For `null_method="lmm"`, `correction="maxstat"` is the default. `cluster` and
`tfce` remain available, but random-slope use must emit a warning:

```text
Cluster/TFCE full-LMM bootstrap correction is not validated for random slopes.
In the 6x5 crossed C2 calibration, maxstat was nominal (0.047) while cluster
and TFCE were elevated (0.093 and 0.087). Use maxstat for validated inference or
run the calibration harness for this design.
```

## Formula And Random Effects

The package should not lower arbitrary random-effect structures into internal
group-code abstractions for the LMM path. Instead:

- `core.formulas.parse_mixed_formula` keeps `original_formula` as the source of
  truth.
- The MixedModels backend receives the full formula string and passes it through
  to StatsModels.jl/MixedModels.jl.
- Fixed-effect column names remain available from Patsy for the fast OLS path
  and for user-facing effect validation.
- A resolver maps the tested Python/Patsy fixed-effect name to the MixedModels
  fixed-effect term, failing loudly on genuine mismatches.

Supported random-effect structures are therefore whatever MixedModels.jl accepts
for the formula, including crossed effects, nested effects represented by
globally labeled factors or explicit interaction factors, and random slopes.

Design note for nested terms:

- Prefer the user providing an explicit nested grouping column, such as
  `subject_contact`, and writing `(1 | subject_contact)`.
- If the user writes an lme4-style interaction such as `(1 | subject:contact)`,
  the backend should pass it through only if the Julia formula surface accepts
  it for the provided data. The package should not silently synthesize nested
  groups unless the user explicitly asks for that helper.

## Full-LMM Backend Contract

Add a backend module, proposed path:

```text
src/lmeeeg/backends/lmm/mixedmodels_backend.py
```

It should expose a reusable engine for one fixed-effect test:

```python
class MixedModelsLMMBackend:
    def bootstrap_fixed_effect(
        self,
        eeg: np.ndarray,
        metadata: pd.DataFrame,
        formula: str,
        effect: str,
        *,
        n_bootstrap: int,
        seed: int,
        correction: str = "maxstat",
        threshold: float | dict[str, float] | None = None,
        adjacency: Any | None = None,
        n_jobs: int = 1,
        shard_index: int | None = None,
        n_shards: int | None = None,
        output_dir: Path | None = None,
    ) -> InferenceResult:
        ...
```

The backend must:

- Fit full and reduced models with ML, never REML, for LRT on fixed effects.
- Build the reduced formula by dropping only the tested fixed effect while
  retaining the full random-effects structure exactly.
- Use the same unsigned LRT statistic for observed and bootstrap maps:
  `2 * (logLik_full - logLik_reduced)`.
- Fit the reduced model once per feature and call MixedModels.jl `simulate` from
  that fitted null model for each bootstrap replicate.
- Refit full and reduced models to each simulated response.
- Summarize the shared observed/null statistic maps with maxstat, cluster, or
  TFCE.

Correctness assertions:

- The Julia fit call must set `REML=false`.
- The reduced formula must retain the random-effects terms byte-for-byte or via
  a structured formula representation proven equivalent.
- The tested fixed effect must be absent from the reduced fixed formula and
  present in the full fixed formula.
- The full and reduced models must use the same response, metadata rows, and
  random-effects formula.

## Diagnostics

The LMM path must surface a diagnostics table, not just corrected p-values.

Required diagnostics:

- observed singular-fit fraction;
- bootstrap singular-fit fraction;
- observed convergence-refit trigger count/fraction;
- bootstrap convergence-refit trigger count/fraction;
- unresolved negative-LR count;
- per-feature observed LR statistic;
- per-feature reduced/full log-likelihoods when feasible;
- per-feature fit error messages;
- total fit count;
- elapsed time and fits/second;
- seed and bootstrap count;
- MixedModels.jl, Julia, and juliacall versions;
- formula, reduced formula, tested effect, and correction.

Negative LR policy:

- A negative LR below numerical epsilon is an optimizer/convergence problem for
  nested ML models.
- The backend retries that fit with the validated alternate optimizer/start
  sequence from `9c9af5b`.
- Do not silently clip meaningful negative LR values.
- If unresolved negative LR remains after retries, raise an error by default and
  include the feature/bootstrap identity in the exception. A future
  `errors="continue"` mode may write NaNs and continue, but it must be explicit.

Singular fits:

- Never drop singular fits.
- Never zero their statistics.
- Record singular/boundary status for observed and bootstrap fits.
- Warn when singular fractions are high enough that inference should be treated
  as fragile.

## Sharding And Parallelism

The workload is embarrassingly parallel over ROIs, spatial chunks, and features.
The backend should support process-level parallelism and file-based sharding.

Proposed behavior:

- `n_jobs > 1` uses multiprocessing/process pools, not threads.
- Each worker initializes Julia independently.
- `shard_index`/`n_shards` split the feature list deterministically.
- `output_dir` stores shard outputs as JSONL/NPZ/Parquet artifacts with enough
  metadata to resume and combine.
- A combine helper merges shard outputs into one `InferenceResult`.
- If `output_dir` exists, completed feature/bootstrap records are skipped only
  when their config hash matches the requested config.

Snakemake-friendly command proposal:

```bash
python -m lmeeeg.calibrate_lmm \
  --formula 'y ~ cond + (1 + cond | subject) + (1 | item)' \
  --effect 'cond[T.B]' \
  --n-bootstrap 1000 \
  --seed 123 \
  --shard-index 0 \
  --n-shards 32 \
  --output-dir results/lmm_bootstrap/{sample}
```

## Guardrail Flip

Current Release 1 behavior hard-refuses random slopes on the fast path. Release
2 should make that conditional:

- `null_method="ols"`: keep the hard refusal unchanged.
- `null_method="lmm"`: allow random slopes and route to the MixedModels
  full-LMM bootstrap path.

The `null_method="lmm"` route should warn:

```text
Full-LMM bootstrap inference for random slopes is design-dependent. The shipped
method is implemented generically, but only specific calibration cells are
validated. Run the calibration harness for this design before relying on FWER.
```

Tests:

- Random-slope formula with `null_method="ols"` raises the Release 1 error.
- The same formula with `null_method="lmm"` reaches the MixedModels backend.
- Random-intercept-only formulas remain valid on both paths.

## User-Facing Calibration Harness

Promote the dev calibration harnesses into a supported entry point:

```bash
python -m lmeeeg.calibrate_lmm_design \
  --formula 'y ~ cond + (1 + cond | subject) + (1 | item)' \
  --effect 'cond[T.B]' \
  --design-json design.json \
  --scenarios C1 C2 C3 \
  --n-sims 300 \
  --n-bootstrap 1000 \
  --correction maxstat \
  --seed 123 \
  --output-dir calibration/lmm_design
```

Supported harness inputs:

- formula, including arbitrary random-effects terms;
- tested effect;
- subject/group counts;
- nested/crossed grouping structure;
- trial counts and within-subject condition balance;
- feature count and optional adjacency;
- random-effect variance magnitudes;
- fixed-effect magnitude for C3;
- correction choices;
- seed, simulation count, bootstrap count, and sharding options.

The harness must include an H0 generator guard:

- For C1/C2, repeated generated datasets must have fixed-effect estimates
  centered at zero.
- Per-feature parametric p-values or bootstrap sanity checks must not show
  obvious non-null contamination.
- If the guard fails, the FWER run must stop. This is non-negotiable: our own
  pre-`3f33618` calibration was invalid because the null generator leaked a
  feature-specific condition effect.

Documentation should present this harness as the recommended pre-registration
step for any slope analysis.

## Validation Envelope To Document

User-facing docs must be precise.

Validated:

- Crossed random intercepts on the fast marginal-OLS path under
  `within_subject` permutation.
- Crossed random intercepts on the full-LMM path as a method sanity control.
- C1 crossed random-intercept calibration is nominal across tested sizes.

Slope evidence for full-LMM bootstrap:

| Scenario | maxstat | cluster | TFCE |
| --- | ---: | ---: | ---: |
| C1 crossed-intercept H0, 6x5 | 0.050 +/- 0.013 | 0.060 +/- 0.014 | 0.057 +/- 0.013 |
| C2 random-slope H0, 6x5 | 0.047 +/- 0.012 | 0.093 +/- 0.017 | 0.087 +/- 0.016 |
| C3 fixed effect, 6x5 | 0.340 +/- 0.027 | 0.613 +/- 0.028 | 0.590 +/- 0.028 |

Interpretation:

- Parametric bootstrap + maxstat was nominal for the crossed 6x5 C2 slope null.
- Cluster and TFCE were elevated or borderline in that same cell and are not the
  default for random-slope inference.
- Power was usable in C3.

Not validated:

- Nested random-effects structures as production inference.
- The real sEEG nested-contact design.
- Random-slope models outside the tested crossed 6x5 cell.
- Non-Gaussian random effects or residuals.
- Arbitrary cluster/TFCE use with slopes.

Negative results to include:

- Design permutation failed under random slopes for LMM-t, Wald, and LRT
  statistics, with FWER around `0.12-0.17` in the decisive cell.
- Fixed-theta permutation failed under random slopes.
- Residual permutation exposed a null/power tradeoff: conditional residuals were
  conservative and underpowered; marginal residuals recovered power but reopened
  C2 inflation.
- Parametric bootstrap is the surviving candidate, but it assumes Gaussian
  random effects and residuals. Permutation assumed exchangeability instead.

Compute expectations:

- Real analyses can require millions of full/reduced fits.
- Users must plan ROI/window scope or sharding before launching dense maps.
- The backend should print or return fit-count and runtime estimates before
  starting large jobs.

## Validation Required Before Phase 1 Is Done

Implementation cannot be considered complete until:

- Existing suite is green and Release 1 behavior is unchanged.
- `null_method="ols"` still refuses random slopes.
- `null_method="lmm"` reaches the MixedModels path for random slopes.
- The production API reproduces the recorded crossed 6x5 C2 maxstat result at
  small scale, using the production backend rather than dev harness internals.
- Determinism holds: same seed produces identical bootstrap nulls, statistic
  maps, corrected p-values, and diagnostics.
- `import lmeeeg` succeeds without Julia, R, pymer4, or juliacall installed.
- Attempting to use the LMM path without Julia/juliacall raises a clear
  install/provisioning message.
- `ruff` and `mypy` are installed in the development environment and run cleanly.

## Review Questions

1. Should the public inference function be renamed to `infer_fixed_effect`, or
   should `permute_fixed_effect` remain the only entry point despite bootstrap
   semantics?
2. Should `null_method` live on `FitConfig`, an inference config, or both? This
   proposal puts it on `FitConfig` so the random-slope guardrail can route early.
3. Should cluster/TFCE with random slopes warn or require an explicit
   `allow_unvalidated=True`?
4. Should unresolved negative LR always raise, or should an explicit
   `errors="continue"` mode be part of the first production build?
5. What on-disk shard format should be considered stable for Snakemake users:
   JSONL plus NPZ, Parquet, or zarr?
