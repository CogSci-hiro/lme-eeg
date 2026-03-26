import numpy as np

from lmeeeg.core.marginal import compute_marginal_eeg


def test_compute_marginal_eeg() -> None:
    eeg = np.ones((3, 2, 2))
    random_effects = 0.25 * np.ones((3, 2, 2))
    marginal = compute_marginal_eeg(eeg=eeg, fitted_random_effects=random_effects)
    assert np.allclose(marginal, 0.75)
