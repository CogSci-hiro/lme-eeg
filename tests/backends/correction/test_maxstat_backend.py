from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.backends.correction.maxstat_backend import MaxStatCorrectionBackend
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_maxstat_backend_smoke() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=4, n_trials_per_subject=4, n_channels=2, n_times=3, seed=6)
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
    backend = MaxStatCorrectionBackend()
    inference = backend.run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=10,
        seed=6,
        tail=0,
        threshold=None,
        adjacency=None,
    )
    assert inference.corrected_p_values.shape == (2, 3)
