import numpy as np
import pandas as pd

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_fit_lmm_mass_univariate_smoke() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=6,
        n_channels=2,
        n_times=4,
        seed=1,
    )
    fit_result = fit_lmm_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
    )
    assert fit_result.marginal_eeg.shape == simulated.eeg.shape
    assert fit_result.fitted_random_effects is None
    assert "condition[T.B]" in fit_result.ols_t_values


def test_fit_lmm_mass_univariate_passes_progress_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyLMMBackend:
        def fit_mass_univariate(
            self,
            eeg,
            metadata,
            design_spec,
            show_progress=True,
            store_fitted_random_effects=False,
            store_marginal_eeg=True,
            output_dtype=None,
        ):
            captured["show_progress"] = show_progress
            captured["store_fitted_random_effects"] = store_fitted_random_effects
            captured["store_marginal_eeg"] = store_marginal_eeg
            captured["output_dtype"] = output_dtype
            return type(
                "DummyLMMResult",
                (),
                {
                    "fixed_effects_maps": {
                        column_name: np.zeros(eeg.shape[1:], dtype=float)
                        for column_name in design_spec.fixed_column_names
                    },
                    "fitted_random_effects": None,
                    "marginal_eeg": np.zeros_like(eeg, dtype=float),
                    "random_effect_variance_map": np.zeros(eeg.shape[1:], dtype=float),
                    "residual_variance_map": np.zeros(eeg.shape[1:], dtype=float),
                    "feature_diagnostics": pd.DataFrame(
                        [
                            {
                                "channel": 0,
                                "time": 0,
                                "converged": True,
                                "boundary_warning": False,
                                "message": "",
                            }
                        ]
                    ),
                },
            )()

    class DummyOLSBackend:
        def fit_mass_univariate(self, eeg, design_matrix, column_names):
            return type(
                "DummyOLSResult",
                (),
                {
                    "beta_maps": {column_name: np.zeros(eeg.shape[1:], dtype=float) for column_name in column_names},
                    "t_value_maps": {column_name: np.zeros(eeg.shape[1:], dtype=float) for column_name in column_names},
                    "residual_variance_map": np.zeros(eeg.shape[1:], dtype=float),
                },
            )()

    monkeypatch.setattr("lmeeeg.api.fit.StatsModelsLMMBackend", DummyLMMBackend)
    monkeypatch.setattr("lmeeeg.api.fit.NumPyOLSBackend", DummyOLSBackend)

    simulated = simulate_random_intercept_dataset(
        n_subjects=2,
        n_trials_per_subject=2,
        n_channels=1,
        n_times=1,
        seed=2,
    )
    fit_lmm_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
        config=FitConfig(show_progress=False),
    )

    assert captured["show_progress"] is False
    assert captured["store_fitted_random_effects"] is False
    assert captured["store_marginal_eeg"] is True
    assert captured["output_dtype"] is None


def test_fit_lmm_mass_univariate_can_store_random_effects_when_requested() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_channels=1,
        n_times=2,
        seed=3,
    )
    fit_result = fit_lmm_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
        config=FitConfig(show_progress=False, store_fitted_random_effects=True),
    )

    assert fit_result.fitted_random_effects is not None
    assert fit_result.fitted_random_effects.shape == simulated.eeg.shape
