import numpy as np

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.backends.correction._regression import max_permutation_p_values
from lmeeeg.backends.correction.maxstat_backend import MaxStatCorrectionBackend
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_max_permutation_p_values_use_plus_one_formula() -> None:
    null_distribution = np.array([0.5, 1.0, 2.0, 3.0])
    observed = np.array([[1.0, 2.5]])
    p_values = max_permutation_p_values(null_distribution, observed)
    expected = np.array([[(1 + 3) / (4 + 1), (1 + 1) / (4 + 1)]])
    np.testing.assert_allclose(p_values, expected)


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
    assert inference.backend_metadata["permutation_scheme"] == "within_subject"


def test_maxstat_backend_chunked_matches_full_fit() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        n_channels=3,
        n_times=4,
        seed=16,
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
    backend = MaxStatCorrectionBackend()
    full = backend.run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=6,
        seed=16,
        tail=0,
        threshold=None,
        adjacency=None,
    )
    chunked = backend.run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=6,
        seed=16,
        tail=0,
        threshold=None,
        adjacency=None,
        spatial_chunk_size=2,
        time_chunk_size=2,
    )

    np.testing.assert_allclose(full.null_distribution, chunked.null_distribution)
    np.testing.assert_allclose(full.corrected_p_values, chunked.corrected_p_values)


def test_maxstat_backend_supports_within_subject_scheme() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        n_channels=2,
        n_times=2,
        seed=26,
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
    inference = MaxStatCorrectionBackend().run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=6,
        seed=26,
        tail=0,
        threshold=None,
        adjacency=None,
        permutation_scheme="within_subject",
    )
    assert inference.corrected_p_values.shape == (2, 2)
    assert inference.backend_metadata["permutation_scheme"] == "within_subject"
