import inspect
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from lmeeeg.backends.lmm.pymer4_backend import Pymer4LMMBackend
from lmeeeg.backends.lmm.statsmodels_backend import StatsModelsLMMBackend
from lmeeeg.backends.ols.numpy_backend import NumPyOLSBackend
from lmeeeg.core.design import build_design_spec
from lmeeeg.core.marginal import compute_marginal_eeg
from lmeeeg.core.results import ConvergenceSummary, FitResult
from lmeeeg.core.space import DTypePolicy, SpaceInfo, SpaceKind, resolve_output_dtype
from lmeeeg.utils.checks import validate_eeg_and_metadata


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
    space : {"sensor", "source", "generic"}
        Meaning of the second data axis. This is metadata only; the core model
        treats sensors, sources, parcels, and vertices as generic locations.
    location_names : Sequence[str] | None
        Optional names for the second data axis.
    source_names : Sequence[str] | None
        Source-space alias for `location_names`.
    feature_names : Sequence[str] | None
        Generic alias for `location_names`.
    dtype : {"preserve", "float32", "float64"}
        Dtype policy for large returned signal arrays. Statistical algebra uses
        float64 regardless of this setting.
    spatial_chunk_size : int | None
        Optional chunk size over the second data axis for OLS.
    time_chunk_size : int | None
        Optional chunk size over the time axis for OLS.
    compute_fixed_effect_t : bool
        Whether supported LMM backends should return fixed-effect standard-error
        and t-statistic maps.
    """

    lmm_backend_name: str = "statsmodels"
    ols_backend_name: str = "numpy"
    show_progress: bool = True
    store_fitted_random_effects: bool = False
    store_marginal_eeg: bool = True
    output_dtype: np.dtype | None = None
    space: SpaceKind = "sensor"
    location_names: Sequence[str] | None = None
    source_names: Sequence[str] | None = None
    feature_names: Sequence[str] | None = None
    dtype: DTypePolicy = "preserve"
    spatial_chunk_size: int | None = None
    time_chunk_size: int | None = None
    compute_fixed_effect_t: bool = False


def _resolve_location_names(config: FitConfig) -> Sequence[str] | None:
    provided = [
        names
        for names in (config.location_names, config.source_names, config.feature_names)
        if names is not None
    ]
    if len(provided) > 1:
        first = list(provided[0])
        if any(list(names) != first for names in provided[1:]):
            raise ValueError(
                "`location_names`, `source_names`, and `feature_names` are aliases; "
                "provide only one value or identical values."
            )
    return provided[0] if provided else None


def _fit_ols_mass_univariate(
    ols_backend: Any,
    marginal_eeg: np.ndarray,
    design_matrix: np.ndarray,
    column_names: list[str],
    spatial_chunk_size: int | None,
    time_chunk_size: int | None,
):
    parameters = inspect.signature(ols_backend.fit_mass_univariate).parameters
    if "spatial_chunk_size" not in parameters and (spatial_chunk_size is not None or time_chunk_size is not None):
        raise ValueError("Selected OLS backend does not support chunked fitting.")
    if "spatial_chunk_size" not in parameters:
        return ols_backend.fit_mass_univariate(
            eeg=marginal_eeg,
            design_matrix=design_matrix,
            column_names=column_names,
        )
    return ols_backend.fit_mass_univariate(
        eeg=marginal_eeg,
        design_matrix=design_matrix,
        column_names=column_names,
        spatial_chunk_size=spatial_chunk_size,
        time_chunk_size=time_chunk_size,
    )


def _fit_lmm_backend(
    lmm_backend: Any,
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    design_spec,
    config: FitConfig,
    output_dtype: np.dtype | None,
):
    parameters = inspect.signature(lmm_backend.fit_mass_univariate).parameters
    kwargs = {
        "eeg": eeg,
        "metadata": metadata,
        "design_spec": design_spec,
        "show_progress": config.show_progress,
        "store_fitted_random_effects": config.store_fitted_random_effects,
        "store_marginal_eeg": config.store_marginal_eeg,
        "output_dtype": output_dtype,
    }
    if "compute_fixed_effect_t" in parameters:
        kwargs["compute_fixed_effect_t"] = config.compute_fixed_effect_t
    return lmm_backend.fit_mass_univariate(**kwargs)


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
    eeg = np.asanyarray(eeg)
    space_info = SpaceInfo(kind=config.space, names=_resolve_location_names(config))
    validate_eeg_and_metadata(eeg=eeg, metadata=metadata, space_info=space_info)
    output_dtype = config.output_dtype
    if output_dtype is None:
        output_dtype = resolve_output_dtype(eeg.dtype, config.dtype)
    design_spec = build_design_spec(
        metadata=metadata,
        formula=formula,
        variable_types=variable_types,
        fit_intercept=fit_intercept,
    )

    if config.lmm_backend_name == "statsmodels":
        lmm_backend = StatsModelsLMMBackend()
    elif config.lmm_backend_name == "pymer4":
        lmm_backend = Pymer4LMMBackend()
    else:
        raise ValueError(f"Unsupported LMM backend: {config.lmm_backend_name}")
    if config.ols_backend_name != "numpy":
        raise ValueError(f"Unsupported OLS backend: {config.ols_backend_name}")

    lmm_result = _fit_lmm_backend(
        lmm_backend=lmm_backend,
        eeg=eeg,
        metadata=metadata,
        design_spec=design_spec,
        config=config,
        output_dtype=output_dtype,
    )

    if lmm_result.marginal_eeg is not None:
        marginal_eeg = lmm_result.marginal_eeg
    elif lmm_result.fitted_random_effects is not None:
        marginal_eeg = compute_marginal_eeg(eeg=eeg, fitted_random_effects=lmm_result.fitted_random_effects)
    else:  # pragma: no cover - guarded by config validation above
        raise RuntimeError("LMM backend returned neither marginalized EEG nor fitted random effects.")

    ols_backend = NumPyOLSBackend()
    ols_result = _fit_ols_mass_univariate(
        ols_backend=ols_backend,
        marginal_eeg=marginal_eeg,
        design_matrix=design_spec.fixed_design_matrix,
        column_names=design_spec.fixed_column_names,
        spatial_chunk_size=config.spatial_chunk_size,
        time_chunk_size=config.time_chunk_size,
    )

    convergence_summary = ConvergenceSummary.from_feature_table(lmm_result.feature_diagnostics)
    n_observations, n_locations, n_times = eeg.shape

    return FitResult(
        formula=formula,
        variable_types=variable_types,
        design_spec=design_spec,
        space_info=space_info,
        n_observations=n_observations,
        n_locations=n_locations,
        n_times=n_times,
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
        fixed_effects_t_maps=getattr(lmm_result, "fixed_effects_t_maps", None),
        fixed_effects_se_maps=getattr(lmm_result, "fixed_effects_se_maps", None),
        backend_metadata={
            "lmm_backend": config.lmm_backend_name,
            "ols_backend": config.ols_backend_name,
            "store_fitted_random_effects": config.store_fitted_random_effects,
            "store_marginal_eeg": config.store_marginal_eeg,
            "output_dtype": None if output_dtype is None else str(np.dtype(output_dtype)),
            "space": space_info.kind,
            "location_names": None if space_info.names is None else list(space_info.names),
            "spatial_chunk_size": config.spatial_chunk_size,
            "time_chunk_size": config.time_chunk_size,
            "dtype": config.dtype,
            "compute_fixed_effect_t": config.compute_fixed_effect_t,
            "lmm_fixed_effect_map_key_names": "patsy",
        },
    )
