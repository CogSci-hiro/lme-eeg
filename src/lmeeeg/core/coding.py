from __future__ import annotations

import pandas as pd


ALLOWED_VARIABLE_TYPES = {"categorical", "numeric", "group"}


# ==============================
# Variable typing checks
# ==============================

def validate_variable_types(metadata: pd.DataFrame, variable_types: dict[str, str]) -> None:
    """Validate explicit variable typing declarations.

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation-level metadata.
    variable_types : dict[str, str]
        Mapping from variable names to variable type labels.
    """
    missing_columns = [column for column in variable_types if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Unknown columns in variable_types: {missing_columns}")

    invalid_types = {
        key: value
        for key, value in variable_types.items()
        if value not in ALLOWED_VARIABLE_TYPES
    }
    if invalid_types:
        raise ValueError(f"Unsupported variable type declarations: {invalid_types}")
