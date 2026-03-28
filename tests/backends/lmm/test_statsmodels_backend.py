from lmeeeg.backends.lmm.statsmodels_backend import StatsModelsLMMBackend
from lmeeeg.core.design import build_design_spec
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_statsmodels_backend_smoke() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=3, n_trials_per_subject=4, n_channels=2, n_times=3, seed=4)
    design_spec = build_design_spec(
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
    )
    backend = StatsModelsLMMBackend()
    result = backend.fit_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        design_spec=design_spec,
        show_progress=False,
    )
    assert result.fitted_random_effects is None
    assert result.marginal_eeg is not None
    assert result.marginal_eeg.shape == simulated.eeg.shape


def test_statsmodels_backend_can_store_random_effects() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=3, n_trials_per_subject=4, n_channels=2, n_times=3, seed=5)
    design_spec = build_design_spec(
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
    )
    backend = StatsModelsLMMBackend()
    result = backend.fit_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        design_spec=design_spec,
        show_progress=False,
        store_fitted_random_effects=True,
    )
    assert result.fitted_random_effects is not None
    assert result.fitted_random_effects.shape == simulated.eeg.shape
