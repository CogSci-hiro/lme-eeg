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
    """

    lmm_backend_name: str = "statsmodels"
    ols_backend_name: str = "numpy"


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

    Usage example
    -------------
        fit_result = fit_lmm_mass_univariate(
            eeg=eeg,
            metadata=metadata,
            formula="y ~ condition + latency + (1|subject)",
            variable_types={
                "condition": "categorical",
                "latency": "numeric",
                "subject": "group",
            },
        )
    """
    config = config or FitConfig()
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
    )

    marginal_eeg = compute_marginal_eeg(eeg=eeg, fitted_random_effects=lmm_result.fitted_random_effects)

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
        marginal_eeg=marginal_eeg,
        ols_betas=ols_result.beta_maps,
        ols_t_values=ols_result.t_value_maps,
        ols_residual_variance=ols_result.residual_variance_map,
        backend_metadata={
            "lmm_backend": config.lmm_backend_name,
            "ols_backend": config.ols_backend_name,
        },
    )
