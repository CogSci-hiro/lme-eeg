from __future__ import annotations

import numpy as np


# ==============================
# Marginal EEG computation
# ==============================

def compute_marginal_eeg(eeg: np.ndarray, fitted_random_effects: np.ndarray) -> np.ndarray:
    """Subtract fitted random effects from EEG.

    Parameters
    ----------
    eeg : np.ndarray
        Original EEG data with shape `(n_observations, n_channels, n_times)`.
    fitted_random_effects : np.ndarray
        Fitted random contribution with the same shape as `eeg`.

    Returns
    -------
    np.ndarray
        Marginal EEG.
    """
    if eeg.shape != fitted_random_effects.shape:
        raise ValueError(
            f"Shape mismatch between eeg {eeg.shape} and fitted_random_effects {fitted_random_effects.shape}."
        )
    return eeg - fitted_random_effects
