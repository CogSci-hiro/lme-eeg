from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_permute_fixed_effect_maxstat_smoke() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=6,
        n_channels=2,
        n_times=4,
        seed=2,
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
    inference = permute_fixed_effect(
        fit_result=fit_result,
        effect="condition[T.B]",
        correction="maxstat",
        n_permutations=20,
        seed=2,
    )
    assert inference.corrected_p_values.shape == (2, 4)
