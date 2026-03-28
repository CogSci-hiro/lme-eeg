from __future__ import annotations

from contextlib import nullcontext
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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
        show_progress: bool = True,
        store_fitted_random_effects: bool = False,
        store_marginal_eeg: bool = True,
        output_dtype: np.dtype | None = None,
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
        n_features = n_channels * n_times
        if output_dtype is None:
            output_dtype = eeg.dtype if np.issubdtype(eeg.dtype, np.floating) else np.float64
        fixed_maps = {
            column_name: np.full((n_channels, n_times), np.nan, dtype=float)
            for column_name in design_spec.fixed_column_names
        }
        fitted_random_effects = (
            np.full(eeg.shape, np.nan, dtype=output_dtype) if store_fitted_random_effects else None
        )
        marginal_eeg = np.full(eeg.shape, np.nan, dtype=output_dtype) if store_marginal_eeg else None
        random_effect_variance_map = np.full((n_channels, n_times), np.nan, dtype=float)
        residual_variance_map = np.full((n_channels, n_times), np.nan, dtype=float)

        diagnostics_rows: list[dict[str, object]] = []

        mixed_formula = design_spec.parsed_formula.fixed_formula
        group_variable = design_spec.group_variable

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
        else:
            progress = None

        with progress or nullcontext() as active_progress:
            task_id = None
            if active_progress is not None:
                task_id = active_progress.add_task("Fitting mixed models", total=n_features)

            for channel_index in range(n_channels):
                for time_index in range(n_times):
                    feature_vector = eeg[:, channel_index, time_index]
                    feature_data = metadata.copy(deep=False)
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

                        fitted_random_feature = np.array(
                            [group_random_effects[group_label] for group_label in feature_data[group_variable]],
                            dtype=output_dtype,
                        )
                        if fitted_random_effects is not None:
                            fitted_random_effects[:, channel_index, time_index] = fitted_random_feature
                        if marginal_eeg is not None:
                            marginal_eeg[:, channel_index, time_index] = feature_vector.astype(
                                output_dtype,
                                copy=False,
                            ) - fitted_random_feature
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

                    if active_progress is not None and task_id is not None:
                        active_progress.advance(task_id)

        diagnostics_table = pd.DataFrame(diagnostics_rows)
        return LMMBackendResult(
            fixed_effects_maps=fixed_maps,
            fitted_random_effects=fitted_random_effects,
            marginal_eeg=marginal_eeg,
            random_effect_variance_map=random_effect_variance_map,
            residual_variance_map=residual_variance_map,
            feature_diagnostics=diagnostics_table,
        )
