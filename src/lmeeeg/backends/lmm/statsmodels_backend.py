from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from lmeeeg.backends.lmm.base import BaseLMMBackend, LMMBackendResult
from lmeeeg.core.design import DesignSpec


# ==============================
# statsmodels MixedLM backend
# ==============================

class StatsModelsLMMBackend(BaseLMMBackend):
    """Random-intercept LMM backend using statsmodels MixedLM.

    Notes
    -----
    This backend prioritizes clarity and minimal dependencies over speed.
    It is suitable for MVP-scale analyses and small simulations.
    """

    def fit_mass_univariate(
        self,
        eeg: np.ndarray,
        metadata: pd.DataFrame,
        design_spec: DesignSpec,
    ) -> LMMBackendResult:
        """Fit one random-intercept LMM per feature.

        Parameters
        ----------
        eeg : np.ndarray
            EEG data with shape `(n_observations, n_channels, n_times)`.
        metadata : pd.DataFrame
            Observation-level metadata.
        design_spec : DesignSpec
            Design specification.

        Returns
        -------
        LMMBackendResult
            Backend result object.
        """
        n_observations, n_channels, n_times = eeg.shape
        fixed_maps = {
            column_name: np.full((n_channels, n_times), np.nan, dtype=float)
            for column_name in design_spec.fixed_column_names
        }
        fitted_random_effects = np.full_like(eeg, np.nan, dtype=float)
        random_effect_variance_map = np.full((n_channels, n_times), np.nan, dtype=float)
        residual_variance_map = np.full((n_channels, n_times), np.nan, dtype=float)

        diagnostics_rows: list[dict[str, object]] = []

        mixed_formula = design_spec.parsed_formula.fixed_formula
        group_variable = design_spec.group_variable

        for channel_index in range(n_channels):
            for time_index in range(n_times):
                feature_vector = eeg[:, channel_index, time_index]
                feature_data = metadata.copy()
                feature_data["y"] = feature_vector
                converged = False
                boundary_warning = False
                message = ""

                try:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always")
                        mixed_model = smf.mixedlm(
                            mixed_formula,
                            data=feature_data,
                            groups=feature_data[group_variable],
                            re_formula="1",
                        )
                        mixed_result = mixed_model.fit(reml=True, method="lbfgs", disp=False)

                    converged = bool(getattr(mixed_result, "converged", False))
                    warning_messages = [str(warning.message) for warning in caught_warnings]
                    boundary_warning = any("boundary" in item.lower() for item in warning_messages)
                    message = " | ".join(warning_messages)

                    for column_name in design_spec.fixed_column_names:
                        if column_name in mixed_result.fe_params.index:
                            fixed_maps[column_name][channel_index, time_index] = float(mixed_result.fe_params[column_name])

                    random_effect_variance_map[channel_index, time_index] = float(np.squeeze(mixed_result.cov_re.to_numpy()))
                    residual_variance_map[channel_index, time_index] = float(mixed_result.scale)

                    group_random_effects: dict[object, float] = {}
                    for group_label, random_effect_series in mixed_result.random_effects.items():
                        if hasattr(random_effect_series, "iloc"):
                            random_value = float(random_effect_series.iloc[0])
                        else:
                            random_value = float(np.ravel(random_effect_series)[0])
                        group_random_effects[group_label] = random_value

                    fitted_random_effects[:, channel_index, time_index] = np.array(
                        [group_random_effects[group_label] for group_label in feature_data[group_variable]],
                        dtype=float,
                    )
                except Exception as error:  # pragma: no cover - error path is hard to force deterministically
                    message = str(error)

                diagnostics_rows.append(
                    {
                        "channel": channel_index,
                        "time": time_index,
                        "converged": converged,
                        "boundary_warning": boundary_warning,
                        "message": message,
                    }
                )

        diagnostics_table = pd.DataFrame(diagnostics_rows)
        return LMMBackendResult(
            fixed_effects_maps=fixed_maps,
            fitted_random_effects=fitted_random_effects,
            random_effect_variance_map=random_effect_variance_map,
            residual_variance_map=residual_variance_map,
            feature_diagnostics=diagnostics_table,
        )
