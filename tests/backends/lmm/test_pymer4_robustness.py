import importlib

import numpy as np
import pandas as pd

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect
from lmeeeg.backends.lmm.pymer4_backend import Pymer4LMMBackend, _prepare_pymer4_environment
from lmeeeg.core.design import build_design_spec
from lmeeeg.simulation.generator import simulate_random_intercept_dataset

from tests.backends.lmm.test_pymer4_backend import _require_pymer4, _simulate_crossed_dataset, _variable_types


def test_pymer4_singular_random_slope_surfaces_boundary_warning() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=False, n_channels=1, n_times=1, seed=50)
    design_spec = build_design_spec(
        metadata=metadata,
        formula="y ~ cond + (1 + cond | subject) + (1 | item)",
        variable_types=_variable_types(),
    )
    result = Pymer4LMMBackend().fit_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        design_spec=design_spec,
        show_progress=False,
    )
    assert result.feature_diagnostics["boundary_warning"].any()
    assert result.feature_diagnostics["message"].str.contains("singular|boundary", case=False).any()


def test_pymer4_fit_failure_records_nan_maps_and_message(monkeypatch) -> None:
    _prepare_pymer4_environment()
    pymer4_models = importlib.import_module("pymer4.models")

    class FailingLmer:
        def __init__(self, *args, **kwargs):
            pass

        def set_factors(self, factors):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("forced pymer4 failure")

    monkeypatch.setattr(pymer4_models, "lmer", FailingLmer)
    eeg, metadata = _simulate_crossed_dataset(random_slope=False, n_channels=1, n_times=1, seed=51)
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
    )
    assert not result.feature_diagnostics["converged"].iloc[0]
    assert "forced pymer4 failure" in result.feature_diagnostics["message"].iloc[0]
    for fixed_map in result.fixed_effects_maps.values():
        assert np.isnan(fixed_map).all()
    assert result.marginal_eeg is not None
    assert np.isnan(result.marginal_eeg).all()


def test_pymer4_fit_and_permutation_are_deterministic() -> None:
    _require_pymer4()
    eeg, metadata = _simulate_crossed_dataset(random_slope=False, n_channels=1, n_times=1, seed=52)
    kwargs = {
        "eeg": eeg,
        "metadata": metadata,
        "formula": "y ~ cond + (1 | subject) + (1 | item)",
        "variable_types": _variable_types(),
        "config": FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    }
    fit_a = fit_lmm_mass_univariate(**kwargs)
    fit_b = fit_lmm_mass_univariate(**kwargs)
    for effect in fit_a.fixed_effects_maps:
        np.testing.assert_allclose(fit_a.fixed_effects_maps[effect], fit_b.fixed_effects_maps[effect])
        np.testing.assert_allclose(fit_a.ols_betas[effect], fit_b.ols_betas[effect])
    effect = "cond[T.low]"
    inference_a = permute_fixed_effect(fit_a, effect=effect, correction="maxstat", n_permutations=10, seed=123)
    inference_b = permute_fixed_effect(fit_a, effect=effect, correction="maxstat", n_permutations=10, seed=123)
    np.testing.assert_allclose(inference_a.null_distribution, inference_b.null_distribution)
    np.testing.assert_allclose(inference_a.corrected_p_values, inference_b.corrected_p_values)


def test_pymer4_matches_statsmodels_on_single_random_intercept() -> None:
    _require_pymer4()
    simulated = simulate_random_intercept_dataset(
        n_subjects=6,
        n_trials_per_subject=8,
        n_channels=1,
        n_times=2,
        seed=53,
    )
    common = {
        "eeg": simulated.eeg,
        "metadata": simulated.metadata,
        "formula": "y ~ condition + latency + (1|subject)",
        "variable_types": {
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
    }
    pymer4_result = fit_lmm_mass_univariate(
        **common,
        config=FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    )
    statsmodels_result = fit_lmm_mass_univariate(
        **common,
        config=FitConfig(show_progress=False, lmm_backend_name="statsmodels"),
    )
    for effect in statsmodels_result.fixed_effects_maps:
        np.testing.assert_allclose(
            pymer4_result.fixed_effects_maps[effect],
            statsmodels_result.fixed_effects_maps[effect],
            rtol=0.15,
            atol=0.15,
        )
