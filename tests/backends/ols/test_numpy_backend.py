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


def test_numpy_backend_chunked_matches_full_fit() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=3,
        n_trials_per_subject=4,
        n_channels=5,
        n_times=4,
        seed=15,
    )
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
    full = backend.fit_mass_univariate(
        eeg=simulated.eeg.astype("float32"),
        design_matrix=design_spec.fixed_design_matrix,
        column_names=design_spec.fixed_column_names,
    )
    chunked = backend.fit_mass_univariate(
        eeg=simulated.eeg.astype("float32"),
        design_matrix=design_spec.fixed_design_matrix,
        column_names=design_spec.fixed_column_names,
        spatial_chunk_size=2,
        time_chunk_size=3,
    )

    assert chunked.residual_variance_map.shape == (5, 4)
    for column_name in design_spec.fixed_column_names:
        assert (abs(full.beta_maps[column_name] - chunked.beta_maps[column_name]) < 1e-10).all()
        assert (abs(full.t_value_maps[column_name] - chunked.t_value_maps[column_name]) < 1e-10).all()
