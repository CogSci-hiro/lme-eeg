from lmeeeg.backends.ols.numpy_backend import NumPyOLSBackend
from lmeeeg.simulation.generator import simulate_random_intercept_dataset
from lmeeeg.core.design import build_design_spec


def test_numpy_backend_smoke() -> None:
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
    backend = NumPyOLSBackend()
    result = backend.fit_mass_univariate(
        eeg=simulated.eeg,
        design_matrix=design_spec.fixed_design_matrix,
        column_names=design_spec.fixed_column_names,
    )
    assert result.residual_variance_map.shape == (2, 3)
