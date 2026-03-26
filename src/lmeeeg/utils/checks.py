from __future__ import annotations

import numpy as np
import pandas as pd


def validate_eeg_and_metadata(eeg: np.ndarray, metadata: pd.DataFrame) -> None:
    """Validate the basic relationship between EEG data and metadata."""
    if eeg.ndim != 3:
        raise ValueError(f"EEG must have 3 dimensions, got {eeg.ndim}.")
    if eeg.shape[0] != len(metadata):
        raise ValueError(
            f"Number of EEG observations ({eeg.shape[0]}) must match metadata rows ({len(metadata)})."
        )
