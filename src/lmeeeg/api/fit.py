from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lmeeeg.backends.lmm.statsmodels_backend import StatsModelsLMMBackend
from lmeeeg.backends.ols.numpy_backend import NumPyOLSBackend
from lmeeeg.core.design import build_design_spec
from lmeeeg.core.marginal import compute_marginal_eeg
from lmeeeg.core.results import ConvergenceSummary, FitResult


@dataclass(slots=True)
class FitConfig:
    """Configuration for the public fit API.

    Parameters
    ----------
    lmm_backend_name : str
        Name of the LMM backend.
    ols_backend_name : str
        Name of the OLS backend.
    show_progress : bool
        Whether to show a progress bar while fitting the per-feature mixed models.
    store_fitted_random_effects : bool
        Whether to keep the full observation × channel × time random-effects array
        in memory and in the returned result.
    store_marginal_eeg : bool
        Whether to keep the marginalized EEG cube in memory and in the returned result.
    output_dtype : np.dtype | None
        Output dtype for large returned EEG-like arrays. Defaults to the dtype of `eeg`
        when it is already floating-point.
    """

    lmm_backend_name: str = "statsmodels"
    ols_backend_name: str = "numpy"
    show_progress: bool = True
    store_fitted_random_effects: bool = False
    store_marginal_eeg: bool = True
    output_dtype: np.dtype | None = None


# ==============================
# Public fit entry point
# ==============================

def fit_lmm_mass_univariate(
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    formula: str,
    variable_types: dict[str, str],
    fit_intercept: bool = True,
    config: FitConfig | None = None,
) -> FitResult:
    """Fit the minimal lmeEEG pipeline.

    Parameters
    ----------
    eeg : np.ndarray
        EEG data with shape `(n_observations, n_channels, n_times)`.
    metadata : pd.DataFrame
        Observation-level metadata. One row per EEG observation.
    formula : str
        Mixed-model style formula, e.g. ``"y ~ condition + latency + (1|subject)"``.
        The response variable must be `y` and is treated as symbolic only.
    variable_types : dict[str, str]
        Explicit variable typing map. Allowed values are ``categorical``,
        ``numeric``, and ``group``.
    fit_intercept : bool
        Whether to include a fixed intercept.
    config : FitConfig | None
        Backend configuration.

    Returns
    -------
    FitResult
        Result object containing design information, convergence diagnostics,
        marginal EEG, and OLS summary statistics.
    """
    config = config or FitConfig()
    if not config.store_fitted_random_effects and not config.store_marginal_eeg:
        raise ValueError(
            "At least one of `store_fitted_random_effects` or `store_marginal_eeg` must be True."
        )
    design_spec = build_design_spec(
        metadata=metadata,
        formula=formula,
        variable_types=variable_types,
        fit_intercept=fit_intercept,
    )

    if config.lmm_backend_name != "statsmodels":
        raise ValueError(f"Unsupported LMM backend: {config.lmm_backend_name}")
    if config.ols_backend_name != "numpy":
        raise ValueError(f"Unsupported OLS backend: {config.ols_backend_name}")

    lmm_backend = StatsModelsLMMBackend()
    lmm_result = lmm_backend.fit_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        design_spec=design_spec,
        show_progress=config.show_progress,
        store_fitted_random_effects=config.store_fitted_random_effects,
        store_marginal_eeg=config.store_marginal_eeg,
        output_dtype=config.output_dtype,
    )

    if lmm_result.marginal_eeg is not None:
        marginal_eeg = lmm_result.marginal_eeg
    elif lmm_result.fitted_random_effects is not None:
        marginal_eeg = compute_marginal_eeg(eeg=eeg, fitted_random_effects=lmm_result.fitted_random_effects)
    else:  # pragma: no cover - guarded by config validation above
        raise RuntimeError("LMM backend returned neither marginalized EEG nor fitted random effects.")

    ols_backend = NumPyOLSBackend()
    ols_result = ols_backend.fit_mass_univariate(
        eeg=marginal_eeg,
        design_matrix=design_spec.fixed_design_matrix,
        column_names=design_spec.fixed_column_names,
    )

    convergence_summary = ConvergenceSummary.from_feature_table(lmm_result.feature_diagnostics)

    return FitResult(
        formula=formula,
        variable_types=variable_types,
        design_spec=design_spec,
        fixed_effects_maps=lmm_result.fixed_effects_maps,
        random_effect_variance_map=lmm_result.random_effect_variance_map,
        residual_variance_map=lmm_result.residual_variance_map,
        fitted_random_effects=lmm_result.fitted_random_effects,
        feature_diagnostics=lmm_result.feature_diagnostics,
        convergence_summary=convergence_summary,
        marginal_eeg=lmm_result.marginal_eeg,
        ols_betas=ols_result.beta_maps,
        ols_t_values=ols_result.t_value_maps,
        ols_residual_variance=ols_result.residual_variance_map,
        backend_metadata={
            "lmm_backend": config.lmm_backend_name,
            "ols_backend": config.ols_backend_name,
            "store_fitted_random_effects": config.store_fitted_random_effects,
            "store_marginal_eeg": config.store_marginal_eeg,
            "output_dtype": None if config.output_dtype is None else str(np.dtype(config.output_dtype)),
        },
    )
