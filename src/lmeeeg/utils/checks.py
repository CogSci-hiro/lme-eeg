import numpy as np
import pandas as pd

from lmeeeg.core.space import SpaceInfo, validate_space_info


def validate_eeg_and_metadata(
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    space_info: SpaceInfo | None = None,
) -> None:
    """Validate the basic relationship between data, metadata, and spatial metadata."""
    if eeg.ndim != 3:
        raise ValueError(
            "EEG/source data must have shape "
            "`(n_observations, n_channels/n_sources/n_locations, n_times)`, "
            f"got {eeg.ndim} dimensions."
        )
    if eeg.shape[0] != len(metadata):
        raise ValueError(
            f"Number of EEG observations ({eeg.shape[0]}) must match metadata rows ({len(metadata)})."
        )
    if space_info is not None:
        validate_space_info(space_info=space_info, n_locations=eeg.shape[1])
