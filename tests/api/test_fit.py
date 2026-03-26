import numpy as np

from lmeeeg.api.fit import fit_lmm_mass_univariate
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
    assert "condition[T.B]" in fit_result.ols_t_values
