from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


SpaceKind = Literal["sensor", "source", "generic"]
DTypePolicy = Literal["preserve", "float32", "float64"]


@dataclass(frozen=True, slots=True)
class SpaceInfo:
    """Metadata for the spatial feature axis.

    The core package treats sensors, sources, parcels, and vertices as the same
    second array axis. Optional adjacency is stored as opaque metadata so MNE
    helpers can use it without making MNE a core dependency.
    """

    kind: SpaceKind = "sensor"
    names: Sequence[str] | None = None
    adjacency: Any | None = None


def validate_space_info(space_info: SpaceInfo, n_locations: int) -> None:
    """Validate spatial-axis metadata against an EEG/source data cube."""
    if space_info.kind not in {"sensor", "source", "generic"}:
        raise ValueError(
            "`space` must be one of 'sensor', 'source', or 'generic', "
            f"got {space_info.kind!r}."
        )
    if space_info.names is not None and len(space_info.names) != n_locations:
        raise ValueError(
            "Number of location names must match the second data axis "
            f"({len(space_info.names)} names for {n_locations} channels/sources/locations)."
        )


def resolve_output_dtype(input_dtype: np.dtype, policy: DTypePolicy) -> np.dtype | None:
    """Resolve the signal-array dtype policy for large returned arrays."""
    if policy == "preserve":
        return None
    if policy == "float32":
        return np.dtype(np.float32)
    if policy == "float64":
        return np.dtype(np.float64)
    raise ValueError("`dtype` must be one of 'preserve', 'float32', or 'float64'.")


def iter_spatiotemporal_chunks(
    n_locations: int,
    n_times: int,
    spatial_chunk_size: int | None = None,
    time_chunk_size: int | None = None,
):
    """Yield slices over the location × time grid."""
    if spatial_chunk_size is None:
        spatial_chunk_size = n_locations
    if time_chunk_size is None:
        time_chunk_size = n_times
    if spatial_chunk_size <= 0:
        raise ValueError("`spatial_chunk_size` must be positive when provided.")
    if time_chunk_size <= 0:
        raise ValueError("`time_chunk_size` must be positive when provided.")

    for location_start in range(0, n_locations, spatial_chunk_size):
        location_stop = min(location_start + spatial_chunk_size, n_locations)
        for time_start in range(0, n_times, time_chunk_size):
            time_stop = min(time_start + time_chunk_size, n_times)
            yield slice(location_start, location_stop), slice(time_start, time_stop)
