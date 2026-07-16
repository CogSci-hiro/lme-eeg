# Pymer4 LMM Backend Verified Surface

Verified locally on 2026-07-16.

- Python: project `.venv`
- R: 4.5.3 at `/opt/homebrew/Cellar/r/4.5.3/lib/R`
- R packages: `lme4` 2.0.1, `lmerTest` 3.2.1, `emmeans` 2.0.3, `report` 0.6.4
- Python packages: `pymer4` 0.9.2, `rpy2` 3.6.7

Resolved pymer4 0.9.2 surface:

- Model class is `pymer4.models.lmer`, not `Lmer`.
- Input data path is Polars; the backend converts feature-level Pandas data with `polars.from_pandas`.
- Fitting call is `model.fit(summary=False, verbose=False)`.
- Fixed-effect table is `model.result_fit`, a Polars DataFrame with columns:
  `term`, `estimate`, `std_error`, `conf_low`, `conf_high`, `t_stat`, `df`, `p_value`.
- Fixed-effect term names are lme4-style, e.g. `(Intercept)`, `condlow`, `condhigh`.
- Fixed-only predictions use `model.predict(data, use_rfx=False)`.
- Full predictions use `model.predict(data, use_rfx=True)`.
- Random-effect variance diagnostics use `model.ranef_var`; rows contain standard deviations in `estimate`.
- Residual standard deviation is `model.result_fit_stats["sigma"]`.
- Design matrix is present as `model.design_matrix`.
- Convergence/singularity messages are surfaced through `model.r_console` and `model.convergence_status`.

Local runtime notes:

- `RPY2_CFFI_MODE=ABI` avoids a broken API-mode probe against a missing framework `libRblas.dylib`.
- The project-local R library `.venv/R/library` contains `report` and related easystats dependencies.
