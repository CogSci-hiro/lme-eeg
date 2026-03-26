from __future__ import annotations

import numpy as np


def stack_effect_for_mne(stat_map: np.ndarray) -> np.ndarray:
    """Convert a `(n_channels, n_times)` map to MNE-friendly `(n_times, n_channels)` layout."""
    return stat_map.T
