from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from patsy import dmatrix

from lmeeeg.core.coding import validate_variable_types
from lmeeeg.core.formulas import ParsedFormula, parse_mixed_formula


@dataclass(slots=True)
class DesignSpec:
    """Container for fixed and random design information.

    Attributes
    ----------
    parsed_formula : ParsedFormula
        Parsed formula object.
    fixed_design_matrix : np.ndarray
        Fixed-effects design matrix with shape `(n_observations, n_fixed_columns)`.
    fixed_column_names : list[str]
        Column names for the fixed-effects design.
    group_variable : str
        Grouping variable for the random intercept.
    group_codes : np.ndarray
        Integer group codes with shape `(n_observations,)`.
    group_labels : list[str]
        Group labels in code order.
    metadata_index : np.ndarray
        Original metadata index values.
    """

    parsed_formula: ParsedFormula
    fixed_design_matrix: np.ndarray
    fixed_column_names: list[str]
    group_variable: str
    group_codes: np.ndarray
    group_labels: list[str]
    metadata_index: np.ndarray


# ==============================
# Design matrix construction
# ==============================

def build_design_spec(
    metadata: pd.DataFrame,
    formula: str,
    variable_types: dict[str, str],
    fit_intercept: bool = True,
) -> DesignSpec:
    """Build the core design specification.

    Parameters
    ----------
    metadata : pd.DataFrame
        Observation-level metadata.
    formula : str
        Mixed-model formula.
    variable_types : dict[str, str]
        Explicit variable typing map.
    fit_intercept : bool
        Whether to include a fixed intercept.

    Returns
    -------
    DesignSpec
        Design specification object.
    """
    validate_variable_types(metadata=metadata, variable_types=variable_types)
    parsed = parse_mixed_formula(formula=formula)

    if parsed.group_variable not in variable_types:
        raise ValueError(f"Grouping variable '{parsed.group_variable}' must be declared in variable_types.")
    if variable_types[parsed.group_variable] != "group":
        raise ValueError(f"Grouping variable '{parsed.group_variable}' must have type 'group'.")

    design_metadata = metadata.copy()
    design_metadata["y"] = 0.0

    fixed_formula = parsed.fixed_formula if fit_intercept else parsed.fixed_formula.replace("y ~", "y ~ 0 +", 1)
    fixed_design = dmatrix(fixed_formula.split("~", maxsplit=1)[1], design_metadata, return_type="dataframe")

    group_codes, unique_groups = pd.factorize(metadata[parsed.group_variable], sort=True)

    return DesignSpec(
        parsed_formula=parsed,
        fixed_design_matrix=fixed_design.to_numpy(dtype=float),
        fixed_column_names=list(fixed_design.columns),
        group_variable=parsed.group_variable,
        group_codes=group_codes.astype(int),
        group_labels=[str(value) for value in unique_groups.tolist()],
        metadata_index=metadata.index.to_numpy(),
    )
