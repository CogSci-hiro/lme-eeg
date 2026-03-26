from __future__ import annotations

import numpy as np


# ==============================
# Contrast handling
# ==============================

def effect_to_contrast(effect: str, fixed_column_names: list[str]) -> np.ndarray:
    """Create a one-degree-of-freedom contrast vector for one fixed-effect column.

    Parameters
    ----------
    effect : str
        Exact fixed-effect column name.
    fixed_column_names : list[str]
        Available fixed-effect column names.

    Returns
    -------
    np.ndarray
        Contrast vector with shape `(n_fixed_columns,)`.
    """
    if effect not in fixed_column_names:
        raise ValueError(f"Unknown effect '{effect}'")
    contrast = np.zeros(len(fixed_column_names), dtype=float)
    contrast[fixed_column_names.index(effect)] = 1.0
    return contrast
