import numpy as np

from lmeeeg.simulation.generator import simulate_erp_random_intercept_dataset, simulate_random_intercept_dataset


def test_simulate_random_intercept_dataset_shapes() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=3, n_trials_per_subject=5, n_channels=2, n_times=4, seed=8)
    assert simulated.eeg.shape == (15, 2, 4)
    assert simulated.ground_truth_effect.shape == (2, 4)


def test_simulate_erp_random_intercept_dataset_shapes() -> None:
    simulated = simulate_erp_random_intercept_dataset(
        n_subjects=3,
        n_trials_per_subject=4,
        n_channels=5,
        sampling_rate_hz=200.0,
        tmin_s=-0.1,
        tmax_s=0.2,
        seed=8,
    )
    assert simulated.eeg.shape == (12, 5, 61)
    assert simulated.time_ms.shape == (61,)
    assert simulated.ground_truth_beta_condition.shape == (5, 61)
    assert set(simulated.ground_truth_component_maps) == {"P100", "N200"}
    assert simulated.simulation_metadata.trial_component_amplitudes.shape == (12, 2)
    assert simulated.simulation_metadata.trial_component_peak_latencies_ms.shape == (12, 2)
    assert np.allclose(
        simulated.ground_truth_beta_condition,
        simulated.ground_truth_component_maps["P100"] + simulated.ground_truth_component_maps["N200"],
    )
