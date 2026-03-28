import numpy as np
import pytest
from scipy import sparse

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.backends.correction._regression import (
    compute_effect_t_statistics,
    prepare_effect_regression,
)
from lmeeeg.backends.correction.mne_cluster_backend import MNEClusterCorrectionBackend
from lmeeeg.backends.correction.mne_tfce_backend import MNETFCorrectionBackend
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mne_backends_smoke() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=4, n_trials_per_subject=4, n_channels=2, n_times=3, seed=7)
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
    cluster_backend = MNEClusterCorrectionBackend()
    tfce_backend = MNETFCorrectionBackend()
    cluster_result = cluster_backend.run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=10,
        seed=7,
        tail=0,
        threshold=1.5,
        adjacency=None,
    )
    tfce_result = tfce_backend.run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=10,
        seed=7,
        tail=0,
        threshold={"start": 0.0, "step": 0.2},
        adjacency=None,
    )
    assert cluster_result.corrected_p_values.shape == (2, 3)
    assert tfce_result.corrected_p_values.shape == (2, 3)
    assert cluster_result.backend_metadata["permutation_scheme"] == "within_group_row_shuffle"
    assert tfce_result.backend_metadata["permutation_scheme"] == "within_group_row_shuffle"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("backend_cls", "threshold", "adjacency"),
    [
        (
            MNEClusterCorrectionBackend,
            1.5,
            sparse.csr_matrix([[0, 1], [1, 0]]),
        ),
        (
            MNETFCorrectionBackend,
            {"start": 0.0, "step": 0.2},
            sparse.eye(6, format="csr"),
        ),
    ],
)
def test_mne_backends_accept_sparse_adjacency(backend_cls, threshold, adjacency) -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        n_channels=2,
        n_times=3,
        seed=7,
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

    result = backend_cls().run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=10,
        seed=7,
        tail=0,
        threshold=threshold,
        adjacency=adjacency,
    )

    assert result.corrected_p_values.shape == (2, 3)


def test_effect_regression_matches_ols_t_map() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=6,
        n_channels=2,
        n_times=4,
        seed=7,
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
    prepared = prepare_effect_regression(fit_result=fit_result, effect="condition[T.B]")
    t_values = compute_effect_t_statistics(
        y_residualized=prepared["y_residualized"],
        effect_residualized=prepared["effect_residualized"],
        effect_sum_squares=float(prepared["effect_sum_squares"]),
        degrees_of_freedom=int(prepared["degrees_of_freedom"]),
    ).reshape(2, 4)

    assert np.allclose(t_values, fit_result.ols_t_values["condition[T.B]"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tfce_backend_detects_nonzero_effect_statistic() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=10,
        n_trials_per_subject=12,
        n_channels=4,
        n_times=12,
        effect_channels=[1, 2],
        effect_times=range(4, 8),
        beta=0.8,
        seed=13,
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

    cluster_result = MNEClusterCorrectionBackend().run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=20,
        seed=13,
        tail=0,
        threshold=2.0,
        adjacency=None,
        verbose=False,
    )
    tfce_result = MNETFCorrectionBackend().run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=20,
        seed=13,
        tail=0,
        threshold={"start": 0.0, "step": 0.2},
        adjacency=None,
        verbose=False,
    )

    assert np.allclose(cluster_result.observed_statistic, fit_result.ols_t_values["condition[T.B]"])
    assert np.nanmax(np.abs(tfce_result.observed_statistic)) > 0.0
    assert np.nanmin(tfce_result.corrected_p_values) < 1.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tfce_backend_logs_at_info_by_default(capsys) -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        n_channels=2,
        n_times=3,
        seed=7,
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

    MNETFCorrectionBackend().run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=5,
        seed=7,
        tail=0,
        threshold={"start": 0.0, "step": 0.2},
        adjacency=None,
    )
    captured = capsys.readouterr()
    assert "Running TFCE correction" in captured.out

    MNETFCorrectionBackend().run(
        fit_result=fit_result,
        effect="condition[T.B]",
        n_permutations=5,
        seed=7,
        tail=0,
        threshold={"start": 0.0, "step": 0.2},
        adjacency=None,
        verbose=False,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
