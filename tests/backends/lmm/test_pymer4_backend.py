import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.backends.lmm.pymer4_backend import Pymer4LMMBackend, _prepare_pymer4_environment, _resolve_term
from lmeeeg.core.design import build_design_spec


def _require_pymer4():
    _prepare_pymer4_environment()
    try:
        pymer4 = importlib.import_module("pymer4")
        importlib.import_module("pymer4.models")
    except Exception as error:
        pytest.skip(f"pymer4/R stack unavailable: {error}")
    return pymer4


CONDITION_LEVELS = ["mid", "low", "high"]


def _simulate_crossed_dataset(
    random_slope: bool,
    crossed_item: bool = True,
    n_channels: int = 2,
    n_times: int = 2,
    seed: int = 41,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = [f"s{i}" for i in range(8)]
    items = [f"i{i}" for i in range(6)]
    rows = []
    for subject in subjects:
        for item in items:
            for condition in CONDITION_LEVELS:
                rows.append({"subject": subject, "item": item, "cond": condition})
    metadata = pd.DataFrame(rows)
    metadata["cond"] = pd.Categorical(metadata["cond"], categories=CONDITION_LEVELS, ordered=False)
    low = (metadata["cond"] == "low").to_numpy(dtype=float)
    high = (metadata["cond"] == "high").to_numpy(dtype=float)
    subject_intercepts = dict(zip(subjects, rng.normal(0.0, 0.55, size=len(subjects)), strict=True))
    subject_low_slopes = dict(zip(subjects, rng.normal(0.0, 0.2, size=len(subjects)), strict=True))
    subject_high_slopes = dict(zip(subjects, rng.normal(0.0, 0.25, size=len(subjects)), strict=True))
    item_intercepts = dict(zip(items, rng.normal(0.0, 0.45, size=len(items)), strict=True))
    random_part = np.array(
        [
            subject_intercepts[row.subject]
            + (
                subject_low_slopes[row.subject] * low_value
                + subject_high_slopes[row.subject] * high_value
                if random_slope
                else 0.0
            )
            + (item_intercepts[row.item] if crossed_item else 0.0)
            for row, low_value, high_value in zip(metadata.itertuples(index=False), low, high, strict=True)
        ]
    )

    eeg = np.empty((len(metadata), n_channels, n_times), dtype=float)
    for location in range(n_channels):
        for time in range(n_times):
            intercept = 1.0 + 0.2 * location - 0.1 * time
            low_effect = 0.45 + 0.05 * location + 0.03 * time
            high_effect = 0.9 + 0.04 * location - 0.02 * time
            noise = rng.normal(0.0, 0.015, size=len(metadata))
            eeg[:, location, time] = intercept + low_effect * low + high_effect * high + random_part + noise
    return eeg, metadata


def _variable_types() -> dict[str, str]:
    return {"cond": "categorical", "subject": "group", "item": "group"}


def _assert_marginal_ols_matches_lmer(eeg: np.ndarray, metadata: pd.DataFrame, formula: str) -> None:
    fit_result = fit_lmm_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        formula=formula,
        variable_types=_variable_types(),
        config=FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    )
    assert fit_result.backend_metadata["lmm_fixed_effect_map_key_names"] == "patsy"
    for effect, beta_map in fit_result.ols_betas.items():
        np.testing.assert_allclose(beta_map, fit_result.fixed_effects_maps[effect], atol=1e-3, rtol=1e-3)


def _dummy_coefficients(y: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    fixed = pd.get_dummies(metadata["cond"], drop_first=True, dtype=float)
    subject = pd.get_dummies(metadata["subject"], prefix="subject", drop_first=True, dtype=float)
    item = pd.get_dummies(metadata["item"], prefix="item", drop_first=True, dtype=float)
    design = pd.concat([fixed, subject, item], axis=1)
    x = np.column_stack([np.ones(len(metadata)), design.to_numpy(dtype=float)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    group_start = 1 + fixed.shape[1]
    return beta[group_start:]


def _direct_lmer_fixed_table(y: np.ndarray, metadata: pd.DataFrame, formula: str):
    _prepare_pymer4_environment()
    import polars as pl
    from pymer4.models import lmer

    data = metadata.copy()
    data["y"] = y
    model = lmer(formula, data=pl.from_pandas(data))
    model.set_factors(
        {
            column: list(data[column].cat.categories)
            for column in data.columns
            if isinstance(data[column].dtype, pd.CategoricalDtype)
        }
    )
    model.fit(summary=False, verbose=False)
    return model.result_fit


def test_import_lmeeeg_succeeds_without_pymer4(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pymer4", None)
    import lmeeeg

    assert hasattr(lmeeeg, "fit_lmm_mass_univariate")


def test_resolve_term_maps_patsy_names_to_lme4_names() -> None:
    available = ["(Intercept)", "condb", "latency", "condb:latency"]
    assert _resolve_term("Intercept", available) == "(Intercept)"
    assert _resolve_term("cond[T.b]", available) == "condb"
    assert _resolve_term("latency", available) == "latency"
    assert _resolve_term("cond[T.b]:latency", available) == "condb:latency"
    assert _resolve_term("missing[T.x]", available) is None


def test_pymer4_crossed_intercepts_marginal_ols_matches_lme4_fixed_effects() -> None:
    pymer4 = _require_pymer4()
    assert getattr(pymer4, "__version__", "unknown")
    eeg, metadata = _simulate_crossed_dataset(random_slope=False, seed=41)
    _assert_marginal_ols_matches_lmer(eeg, metadata, "y ~ cond + (1 | subject) + (1 | item)")


def test_pymer4_random_slope_marginal_ols_matches_lme4_fixed_effects() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=True, crossed_item=False, seed=42)
    _assert_marginal_ols_matches_lmer(eeg, metadata, "y ~ cond + (1 + cond | subject)")


def test_pymer4_crossed_random_slope_marginal_ols_matches_lme4_fixed_effects() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=True, crossed_item=True, seed=43)
    _assert_marginal_ols_matches_lmer(eeg, metadata, "y ~ cond + (1 + cond | subject) + (1 | item)")


def test_pymer4_fixed_effect_t_and_se_maps_are_finite() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=False, n_channels=1, n_times=1, seed=44)
    design_spec = build_design_spec(
        metadata=metadata,
        formula="y ~ cond + (1 | subject) + (1 | item)",
        variable_types=_variable_types(),
    )
    result = Pymer4LMMBackend().fit_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        design_spec=design_spec,
        show_progress=False,
        compute_fixed_effect_t=True,
    )
    direct_table = _direct_lmer_fixed_table(eeg[:, 0, 0], metadata, design_spec.parsed_formula.original_formula)
    assert result.fixed_effects_t_maps is not None
    assert result.fixed_effects_se_maps is not None
    for effect in design_spec.fixed_column_names:
        term = _resolve_term(effect, direct_table["term"].to_list())
        assert term is not None
        direct_row = direct_table.filter(direct_table["term"] == term).row(0, named=True)
        assert np.isfinite(result.fixed_effects_t_maps[effect]).all()
        assert np.isfinite(result.fixed_effects_se_maps[effect]).all()
        assert effect in result.fixed_effects_maps
        np.testing.assert_allclose(result.fixed_effects_t_maps[effect][0, 0], direct_row["t_stat"], rtol=1e-6)
        np.testing.assert_allclose(result.fixed_effects_se_maps[effect][0, 0], direct_row["std_error"], rtol=1e-6)


def test_pymer4_marginalisation_removes_crossed_random_effects() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=False, n_channels=1, n_times=1, seed=45)
    fit_result = fit_lmm_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        formula="y ~ cond + (1 | subject) + (1 | item)",
        variable_types=_variable_types(),
        config=FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    )
    raw_group_coefficients = _dummy_coefficients(eeg[:, 0, 0], metadata)
    marginal_group_coefficients = _dummy_coefficients(fit_result.marginal_eeg[:, 0, 0], metadata)
    assert np.max(np.abs(marginal_group_coefficients)) < 0.05
    assert np.var(fit_result.marginal_eeg[:, 0, 0]) < np.var(eeg[:, 0, 0])
    assert np.max(np.abs(raw_group_coefficients)) > 5 * np.max(np.abs(marginal_group_coefficients))
