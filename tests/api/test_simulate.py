import numpy as np

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.simulate import simulate_erp_random_intercept_dataset


def test_public_simulate_api_returns_erp_result_and_runs_fit() -> None:
    simulated = simulate_erp_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=6,
        n_channels=4,
        sampling_rate_hz=200.0,
        tmin_s=-0.1,
        tmax_s=0.25,
        seed=19,
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
    assert "condition[T.B]" in fit_result.ols_t_values
    assert simulated.ground_truth_component_maps["N200"].shape == simulated.ground_truth_beta_condition.shape
    assert np.any(np.abs(simulated.ground_truth_beta_condition) > 0.0)
