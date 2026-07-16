import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.backends.lmm.pymer4_backend import Pymer4LMMBackend, _resolve_term
from lmeeeg.core.design import build_design_spec


def _require_pymer4():
    try:
        pymer4 = importlib.import_module("pymer4")
        importlib.import_module("pymer4.models")
    except Exception as error:
        pytest.skip(f"pymer4/R stack unavailable: {error}")
    return pymer4


def _simulate_crossed_dataset(random_slope: bool) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(42 if random_slope else 41)
    subjects = [f"s{i}" for i in range(6)]
    items = [f"i{i}" for i in range(6)]
    rows = []
    for subject_index, subject in enumerate(subjects):
        for item_index, item in enumerate(items):
            rows.append(
                {
                    "subject": subject,
                    "item": item,
                    "cond": "b" if (subject_index + item_index) % 2 else "a",
                }
            )
    metadata = pd.DataFrame(rows)
    cond_numeric = (metadata["cond"] == "b").to_numpy(dtype=float)
    subject_intercepts = dict(zip(subjects, rng.normal(0.0, 0.35, size=len(subjects)), strict=True))
    subject_slopes = dict(zip(subjects, rng.normal(0.0, 0.15, size=len(subjects)), strict=True))
    item_intercepts = dict(zip(items, rng.normal(0.0, 0.25, size=len(items)), strict=True))
    random_part = np.array(
        [
            subject_intercepts[row.subject]
            + (subject_slopes[row.subject] * cond if random_slope else 0.0)
            + item_intercepts[row.item]
            for row, cond in zip(metadata.itertuples(index=False), cond_numeric, strict=True)
        ]
    )

    eeg = np.empty((len(metadata), 2, 2), dtype=float)
    for location in range(2):
        for time in range(2):
            intercept = 1.0 + 0.2 * location - 0.1 * time
            cond_effect = 0.75 + 0.05 * location + 0.03 * time
            noise = rng.normal(0.0, 0.03, size=len(metadata))
            eeg[:, location, time] = intercept + cond_effect * cond_numeric + random_part + noise
    return eeg, metadata


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


def test_pymer4_crossed_marginal_ols_matches_lme4_fixed_effects() -> None:
    pymer4 = _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=False)
    fit_result = fit_lmm_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        formula="y ~ cond + (1 | subject) + (1 | item)",
        variable_types={"cond": "categorical", "subject": "group", "item": "group"},
        config=FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    )
    assert getattr(pymer4, "__version__", "unknown")
    for effect, beta_map in fit_result.ols_betas.items():
        np.testing.assert_allclose(beta_map, fit_result.fixed_effects_maps[effect], atol=1e-6, rtol=1e-6)


def test_pymer4_random_slope_marginal_ols_matches_lme4_fixed_effects() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=True)
    fit_result = fit_lmm_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        formula="y ~ cond + (1 + cond | subject) + (1 | item)",
        variable_types={"cond": "categorical", "subject": "group", "item": "group"},
        config=FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    )
    for effect, beta_map in fit_result.ols_betas.items():
        np.testing.assert_allclose(beta_map, fit_result.fixed_effects_maps[effect], atol=1e-6, rtol=1e-6)


def test_pymer4_fixed_effect_t_and_se_maps_are_finite() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=False)
    design_spec = build_design_spec(
        metadata=metadata,
        formula="y ~ cond + (1 | subject) + (1 | item)",
        variable_types={"cond": "categorical", "subject": "group", "item": "group"},
    )
    result = Pymer4LMMBackend().fit_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        design_spec=design_spec,
        show_progress=False,
        compute_fixed_effect_t=True,
    )
    assert result.fixed_effects_t_maps is not None
    assert result.fixed_effects_se_maps is not None
    for effect in design_spec.fixed_column_names:
        assert np.isfinite(result.fixed_effects_t_maps[effect]).all()
        assert np.isfinite(result.fixed_effects_se_maps[effect]).all()
