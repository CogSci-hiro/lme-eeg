import numpy as np

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_condition_effect_recovery_is_positive_in_ground_truth_region() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=10,
        n_trials_per_subject=10,
        n_channels=4,
        n_times=12,
        effect_channels=[1],
        effect_times=range(4, 8),
        beta=1.0,
        noise_sd=0.5,
        seed=9,
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
    estimated = fit_result.ols_betas["condition[T.B]"]
    truth_mask = simulated.ground_truth_effect > 0
    assert np.nanmean(estimated[truth_mask]) > 0.3
