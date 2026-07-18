from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
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

    Calibration envelope and real-EEG go/no-go status are documented in
    ``docs/CALIBRATION.md``.
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
        _prepare_pymer4_environment()
        try:
            import polars as pl
            from pymer4.models import lmer
        except Exception as import_error:  # pragma: no cover - depends on optional R stack
            raise ImportError(
                "Pymer4LMMBackend requires pymer4 plus its Python dependencies and an R install "
                "with lme4/lmerTest. Install R packages lme4, lmerTest, emmeans, report, then "
                "install pymer4, rpy2, polars, pyarrow, great-tables, scikit-learn, and formulae."
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
                        model_data = pl.from_pandas(feature_data)
                        model = lmer(lme4_formula, data=model_data)
                        factors = _factor_levels(feature_data)
                        if factors:
                            model.set_factors(factors)
                        _fit_model(model)

                        coefs = _coefficient_table(model)
                        for column_name in fixed_names:
                            term = _resolve_term(column_name, coefs["term"].to_list())
                            if term is None:
                                raise KeyError(
                                    f"Could not resolve Patsy fixed-effect term '{column_name}' "
                                    f"against lme4 coefficient terms {coefs['term'].to_list()!r}."
                                )
                            row = coefs.filter(pl.col("term") == term).row(0, named=True)
                            fixed_maps[column_name][location_index, time_index] = float(row["estimate"])
                            if compute_fixed_effect_t and t_maps is not None and se_maps is not None:
                                se_maps[column_name][location_index, time_index] = float(row["std_error"])
                                t_maps[column_name][location_index, time_index] = float(row["t_stat"])

                        random_effect_variance_map[location_index, time_index] = _total_re_variance(model)
                        residual_variance_map[location_index, time_index] = _residual_variance(model)

                        full_fitted = _predict(model, model_data, use_rfx=True)
                        fixed_only = _predict(model, model_data, use_rfx=False)
                        random_contrib = (full_fitted - fixed_only).astype(output_dtype, copy=False)

                        if fitted_random_effects is not None:
                            fitted_random_effects[:, location_index, time_index] = random_contrib
                        if marginal_eeg is not None:
                            marginal_eeg[:, location_index, time_index] = (
                                feature_vector.astype(output_dtype, copy=False) - random_contrib
                            )

                        converged = _converged(model)
                        boundary_warning = _has_boundary_warning(model)
                        message = " | ".join(_model_warnings(model))
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


def _prepare_pymer4_environment() -> None:
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
    local_r_library = Path(__file__).resolve().parents[4] / ".venv" / "R" / "library"
    if local_r_library.exists():
        current = os.environ.get("R_LIBS_USER")
        local = str(local_r_library)
        if current:
            if local not in current.split(os.pathsep):
                os.environ["R_LIBS_USER"] = os.pathsep.join([local, current])
        else:
            os.environ["R_LIBS_USER"] = local


def _fit_model(model: Any) -> None:
    model.fit(summary=False, verbose=False)


def _coefficient_table(model: Any):
    coefs = model.result_fit
    missing_columns = {"term", "estimate", "std_error", "t_stat"} - set(coefs.columns)
    if missing_columns:
        raise AttributeError(f"pymer4 coefficient table is missing columns: {sorted(missing_columns)}")
    return coefs


def _factor_levels(data: pd.DataFrame) -> dict[str, list[Any]]:
    factors: dict[str, list[Any]] = {}
    for column_name in data.columns:
        if column_name == "y":
            continue
        series = data[column_name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            factors[column_name] = list(series.cat.categories)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            factors[column_name] = sorted(series.dropna().unique().tolist())
    return factors


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


def _predict(model: Any, data: Any, use_rfx: bool) -> np.ndarray:
    """Return predictions aligned to ``data`` with or without random effects."""
    predicted = model.predict(data, use_rfx=use_rfx)
    values = np.asarray(predicted, dtype=float)
    if values.ndim > 1:
        values = np.ravel(values)
    if len(values) != len(data):
        raise ValueError(f"pymer4 returned {len(values)} predictions for {len(data)} rows.")
    return values


def _total_re_variance(model: Any) -> float:
    """Return the sum of random-effect variances as a diagnostic."""
    ranef_var = model.ranef_var
    table = ranef_var.filter(~ranef_var["group"].str.contains("Residual"))
    return float((table["estimate"] ** 2).sum()) if table.height else np.nan


def _residual_variance(model: Any) -> float:
    """Return residual variance as a diagnostic."""
    return float(model.result_fit_stats["sigma"].item() ** 2)


def _converged(model: Any) -> bool:
    warnings_text = " ".join(_model_warnings(model)).lower()
    return "failed to converge" not in warnings_text and "convergence" not in warnings_text


def _has_boundary_warning(model: Any) -> bool:
    warnings_text = " ".join(_model_warnings(model)).lower()
    return "boundary" in warnings_text or "singular" in warnings_text


def _model_warnings(model: Any) -> list[str]:
    warnings = [str(item) for item in getattr(model, "r_console", [])]
    convergence_status = getattr(model, "convergence_status", "")
    if convergence_status:
        warnings.append(str(convergence_status))
    return warnings
