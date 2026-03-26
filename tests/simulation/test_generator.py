from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_simulate_random_intercept_dataset_shapes() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=3, n_trials_per_subject=5, n_channels=2, n_times=4, seed=8)
    assert simulated.eeg.shape == (15, 2, 4)
    assert simulated.ground_truth_effect.shape == (2, 4)
