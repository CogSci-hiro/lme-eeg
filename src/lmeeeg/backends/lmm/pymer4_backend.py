from __future__ import annotations

from contextlib import nullcontext
import inspect
import re
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from lmeeeg.backends.lmm.base import BaseLMMBackend, LMMBackendResult
from lmeeeg.core.design import DesignSpec


class Pymer4LMMBackend(BaseLMMBackend):
    """LMM backend using pymer4/lme4.

    Supports lme4 random-effect formulas such as
    ``y ~ cond + (1 + cond | subject) + (1 | item)``.

    Public fixed-effect maps are keyed by Patsy column names so they line up with
    ``DesignSpec.fixed_column_names`` and downstream inference validation. Those
    keys are resolved to lme4 coefficient-table rows per fit.
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
        compute_fixed_effect_t: bool = False,
    ) -> LMMBackendResult:
        try:
            from pymer4.models import Lmer
        except Exception as import_error:  # pragma: no cover - depends on optional R stack
            raise ImportError(
                "Pymer4LMMBackend requires pymer4 plus an R install with lme4/lmerTest. "
                "Install the R dependencies, then install pymer4 in this environment."
            ) from import_error

        n_observations, n_locations, n_times = eeg.shape
        n_features = n_locations * n_times
        if len(metadata) != n_observations:
            raise ValueError("metadata must have one row per EEG observation.")
        if output_dtype is None:
            output_dtype = eeg.dtype if np.issubdtype(eeg.dtype, np.floating) else np.float64

        lme4_formula = design_spec.parsed_formula.original_formula
        fixed_names = design_spec.fixed_column_names
        fixed_maps = {
            column_name: np.full((n_locations, n_times), np.nan, dtype=float)
            for column_name in fixed_names
        }
        t_maps = (
            {
                column_name: np.full((n_locations, n_times), np.nan, dtype=float)
                for column_name in fixed_names
            }
            if compute_fixed_effect_t
            else None
        )
        se_maps = (
            {
                column_name: np.full((n_locations, n_times), np.nan, dtype=float)
                for column_name in fixed_names
            }
            if compute_fixed_effect_t
            else None
        )
        fitted_random_effects = (
            np.full(eeg.shape, np.nan, dtype=output_dtype) if store_fitted_random_effects else None
        )
        marginal_eeg = np.full(eeg.shape, np.nan, dtype=output_dtype) if store_marginal_eeg else None
        random_effect_variance_map = np.full((n_locations, n_times), np.nan, dtype=float)
        residual_variance_map = np.full((n_locations, n_times), np.nan, dtype=float)
        diagnostics_rows: list[dict[str, object]] = []

        progress = _make_progress() if show_progress else None
        with progress or nullcontext() as active_progress:
            task_id = (
                active_progress.add_task("Fitting lme4 models", total=n_features)
                if active_progress is not None
                else None
            )

            for location_index in range(n_locations):
                for time_index in range(n_times):
                    feature_vector = eeg[:, location_index, time_index]
                    feature_data = metadata.copy(deep=False)
                    feature_data["y"] = feature_vector

                    converged = False
                    boundary_warning = False
                    message = ""

                    try:
                        model = Lmer(lme4_formula, data=feature_data)
                        _fit_model(model)

                        coefs = _coefficient_table(model)
                        for column_name in fixed_names:
                            term = _resolve_term(column_name, coefs.index)
                            if term is None:
                                raise KeyError(
                                    f"Could not resolve Patsy fixed-effect term '{column_name}' "
                                    f"against lme4 coefficient terms {list(coefs.index)!r}."
                                )
                            fixed_maps[column_name][location_index, time_index] = float(
                                coefs.loc[term, "Estimate"]
                            )
                            if compute_fixed_effect_t and t_maps is not None and se_maps is not None:
                                se_maps[column_name][location_index, time_index] = float(
                                    coefs.loc[term, _first_existing_column(coefs, ("SE", "Std. Error"))]
                                )
                                t_maps[column_name][location_index, time_index] = float(
                                    coefs.loc[term, _first_existing_column(coefs, ("T-stat", "t value", "T.value"))]
                                )

                        random_effect_variance_map[location_index, time_index] = _total_re_variance(model)
                        residual_variance_map[location_index, time_index] = _residual_variance(model)

                        full_fitted = _predict(model, feature_data, use_rfx=True)
                        fixed_only = _predict(model, feature_data, use_rfx=False)
                        random_contrib = (full_fitted - fixed_only).astype(output_dtype, copy=False)

                        if fitted_random_effects is not None:
                            fitted_random_effects[:, location_index, time_index] = random_contrib
                        if marginal_eeg is not None:
                            marginal_eeg[:, location_index, time_index] = (
                                feature_vector.astype(output_dtype, copy=False) - random_contrib
                            )

                        converged = _converged(model)
                        boundary_warning = _has_boundary_warning(model)
                    except Exception as error:  # pragma: no cover - exercised by optional pymer4 stack
                        message = str(error)

                    diagnostics_rows.append(
                        {
                            "location": location_index,
                            "channel": location_index,
                            "time": time_index,
                            "converged": converged,
                            "boundary_warning": boundary_warning,
                            "message": message,
                        }
                    )
                    if active_progress is not None and task_id is not None:
                        active_progress.advance(task_id)

        return LMMBackendResult(
            fixed_effects_maps=fixed_maps,
            fitted_random_effects=fitted_random_effects,
            marginal_eeg=marginal_eeg,
            random_effect_variance_map=random_effect_variance_map,
            residual_variance_map=residual_variance_map,
            feature_diagnostics=pd.DataFrame(diagnostics_rows),
            fixed_effects_t_maps=t_maps,
            fixed_effects_se_maps=se_maps,
        )


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _fit_model(model: Any) -> None:
    try:
        model.fit(summarize=False)
    except TypeError:
        model.fit()


def _coefficient_table(model: Any) -> pd.DataFrame:
    coefs = getattr(model, "coefs", None)
    if coefs is None:
        coefs = getattr(model, "coef_table", None)
    if coefs is None:
        raise AttributeError("Could not find pymer4 coefficient table on model.coefs or model.coef_table.")
    if not isinstance(coefs, pd.DataFrame):
        coefs = pd.DataFrame(coefs)
    missing_columns = {"Estimate"} - set(coefs.columns)
    if missing_columns:
        raise AttributeError(f"pymer4 coefficient table is missing columns: {sorted(missing_columns)}")
    return coefs


def _resolve_term(fixed_name: str, available_terms) -> str | None:
    """Map a Patsy fixed-effect name to an lme4 coefficient-table key."""
    terms = [str(term) for term in list(available_terms)]
    candidates = _term_candidates(fixed_name)
    for candidate in candidates:
        if candidate in terms:
            return candidate
    return None


def _term_candidates(term: str) -> list[str]:
    if term in {"Intercept", "(Intercept)"}:
        return [term, "(Intercept)", "Intercept"]

    pieces = term.split(":")
    converted_pieces = [_convert_patsy_piece(piece) for piece in pieces]
    converted = ":".join(converted_pieces)
    candidates = [term, converted]
    if converted != term:
        candidates.append(converted.replace("[T.", "").replace("]", ""))
    return list(dict.fromkeys(candidates))


def _convert_patsy_piece(piece: str) -> str:
    match = re.fullmatch(r"(.+)\[T\.(.+)\]", piece)
    if match is None:
        return piece
    variable, level = match.groups()
    return f"{variable}{level}"


def _predict(model: Any, data: pd.DataFrame, use_rfx: bool) -> np.ndarray:
    """Return predictions aligned to ``data`` with or without random effects."""
    predict = model.predict
    kwargs = _prediction_kwargs(predict, use_rfx=use_rfx)
    try:
        predicted = predict(data, **kwargs)
    except TypeError:
        predicted = predict(data=data, **kwargs)
    values = np.asarray(predicted, dtype=float)
    if values.ndim > 1:
        values = np.ravel(values)
    if len(values) != len(data):
        raise ValueError(f"pymer4 returned {len(values)} predictions for {len(data)} rows.")
    return values


def _prediction_kwargs(predict: Any, use_rfx: bool) -> dict[str, Any]:
    parameters = inspect.signature(predict).parameters
    kwargs: dict[str, Any] = {}
    if "use_rfx" in parameters:
        kwargs["use_rfx"] = use_rfx
    elif "use_re" in parameters:
        kwargs["use_re"] = use_rfx
    elif "re_formula" in parameters:
        kwargs["re_formula"] = None if use_rfx else "NA"
    elif "re_form" in parameters:
        kwargs["re_form"] = None if use_rfx else "NA"
    else:
        raise AttributeError("pymer4 predict() does not expose a random-effect inclusion argument.")

    if "verify_predictions" in parameters:
        kwargs["verify_predictions"] = False
    if "skip_data_checks" in parameters:
        kwargs["skip_data_checks"] = True
    return kwargs


def _total_re_variance(model: Any) -> float:
    """Return the sum of random-effect variances as a best-effort diagnostic."""
    for attribute in ("ranef_var", "ranef_var_", "random_effects_var"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            total = _sum_numeric_variance(value, exclude_residual=True)
            if np.isfinite(total):
                return total
    return np.nan


def _residual_variance(model: Any) -> float:
    """Return residual variance as a best-effort diagnostic."""
    for attribute in ("sigma", "resid_sd", "residual_sd"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            try:
                return float(value) ** 2
            except (TypeError, ValueError):
                pass

    for attribute in ("ranef_var", "ranef_var_", "fit_stats"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            residual = _extract_residual_variance(value)
            if np.isfinite(residual):
                return residual
    return np.nan


def _sum_numeric_variance(value: Any, exclude_residual: bool) -> float:
    if isinstance(value, pd.DataFrame):
        table = value.copy()
        if exclude_residual:
            mask = ~table.index.astype(str).str.contains("resid", case=False, regex=True)
            table = table.loc[mask]
        numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        return float(np.nansum(numeric)) if numeric.size else np.nan
    if isinstance(value, pd.Series):
        series = value.copy()
        if exclude_residual:
            series = series.loc[~series.index.astype(str).str.contains("resid", case=False, regex=True)]
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return float(np.nansum(numeric)) if numeric.size else np.nan
    if isinstance(value, dict):
        total = 0.0
        seen = False
        for key, item in value.items():
            if exclude_residual and "resid" in str(key).lower():
                continue
            partial = _sum_numeric_variance(item, exclude_residual=False)
            if np.isfinite(partial):
                total += partial
                seen = True
        return total if seen else np.nan
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric_value


def _extract_residual_variance(value: Any) -> float:
    if isinstance(value, pd.DataFrame):
        residual_rows = value.index.astype(str).str.contains("resid", case=False, regex=True)
        if residual_rows.any():
            numeric = value.loc[residual_rows].select_dtypes(include=[np.number]).to_numpy(dtype=float)
            return float(np.ravel(numeric)[0]) if numeric.size else np.nan
    if isinstance(value, pd.Series):
        residual_rows = value.index.astype(str).str.contains("resid", case=False, regex=True)
        if residual_rows.any():
            numeric = pd.to_numeric(value.loc[residual_rows], errors="coerce").to_numpy(dtype=float)
            return float(np.ravel(numeric)[0]) if numeric.size else np.nan
    if isinstance(value, dict):
        for key, item in value.items():
            if "sigma" in str(key).lower() or "resid" in str(key).lower():
                try:
                    numeric = float(item)
                    return numeric**2 if "sigma" in str(key).lower() else numeric
                except (TypeError, ValueError):
                    residual = _extract_residual_variance(item)
                    if np.isfinite(residual):
                        return residual
    return np.nan


def _converged(model: Any) -> bool:
    for attribute in ("converged", "convergence_status"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"converged", "ok", "true", "success"}
    warnings_text = " ".join(_model_warnings(model)).lower()
    return "failed to converge" not in warnings_text and "convergence" not in warnings_text


def _has_boundary_warning(model: Any) -> bool:
    return any("boundary" in warning.lower() for warning in _model_warnings(model))


def _model_warnings(model: Any) -> list[str]:
    warnings: list[str] = []
    for attribute in ("warnings", "fit_warnings", "ranef_warnings"):
        value = getattr(model, attribute, None)
        if value is None:
            continue
        if isinstance(value, str):
            warnings.append(value)
        else:
            warnings.extend(str(item) for item in value)
    return warnings


def _first_existing_column(table: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in table.columns:
            return name
    raise AttributeError(f"pymer4 coefficient table is missing all columns: {list(names)}")
