import pytest

from lmeeeg.api.fit import fit_lmm_mass_univariate
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
