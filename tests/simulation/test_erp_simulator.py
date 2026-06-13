import numpy as np

from lmeeeg.simulation.generator import ERPComponentSpec, simulate_erp_random_intercept_dataset


def test_erp_simulator_is_reproducible_with_fixed_seed() -> None:
    first = simulate_erp_random_intercept_dataset(seed=11)
    second = simulate_erp_random_intercept_dataset(seed=11)

    assert np.allclose(first.eeg, second.eeg)
    assert first.metadata.equals(second.metadata)
    assert np.allclose(first.time_ms, second.time_ms)
    assert np.allclose(first.ground_truth_beta_condition, second.ground_truth_beta_condition)


def test_erp_simulator_metadata_contains_required_columns() -> None:
    simulated = simulate_erp_random_intercept_dataset(n_subjects=2, n_trials_per_subject=3, seed=5)

    required_columns = {"subject", "condition", "latency", "trial_index", "observation_index"}
    assert required_columns.issubset(simulated.metadata.columns)


def test_condition_beta_is_nonzero_when_condition_effect_is_enabled() -> None:
    simulated = simulate_erp_random_intercept_dataset(seed=3)

    assert np.max(np.abs(simulated.ground_truth_beta_condition)) > 0.1


def test_condition_beta_is_near_zero_when_condition_effect_is_disabled() -> None:
    component_specs = (
        ERPComponentSpec(
            name="P100",
            peak_latency_ms=100.0,
            width_ms=20.0,
            polarity=1.0,
            amplitude_intercept=1.5,
            amplitude_condition_effect=0.0,
        ),
        ERPComponentSpec(
            name="N200",
            peak_latency_ms=200.0,
            width_ms=30.0,
            polarity=-1.0,
            amplitude_intercept=1.2,
            amplitude_condition_effect=0.0,
        ),
    )

    simulated = simulate_erp_random_intercept_dataset(component_specs=component_specs, seed=13)

    assert np.max(np.abs(simulated.ground_truth_beta_condition)) < 1e-10


def test_latency_jitter_changes_component_peak_timing_across_trials() -> None:
    simulated = simulate_erp_random_intercept_dataset(n_subjects=4, n_trials_per_subject=5, seed=17)

    peak_latencies_ms = simulated.simulation_metadata.trial_component_peak_latencies_ms
    assert np.std(peak_latencies_ms[:, 0]) > 1.0
    assert np.std(peak_latencies_ms[:, 1]) > 1.0


def test_subject_random_intercepts_create_between_subject_variability() -> None:
    simulated = simulate_erp_random_intercept_dataset(n_subjects=6, n_trials_per_subject=4, seed=23)

    offsets = simulated.simulation_metadata.subject_component_amplitude_offsets
    assert offsets.shape == (6, 2)
    assert np.std(offsets[:, 0]) > 0.05
    assert np.std(offsets[:, 1]) > 0.05
