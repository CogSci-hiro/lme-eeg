from __future__ import annotations

import numpy as np


def flatten_features(eeg: np.ndarray) -> np.ndarray:
    """Flatten channel × time features into a single trailing dimension."""
    return eeg.reshape(eeg.shape[0], -1)
