from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedFormula:
    """Parsed mixed-model style formula.

    Attributes
    ----------
    fixed_formula : str
        Patsy-compatible fixed-effects formula.
    group_variable : str
        Random intercept grouping variable.
    original_formula : str
        Original formula string.
    """

    fixed_formula: str
    group_variable: str
    original_formula: str


# ==============================
# Formula parsing
# ==============================

def parse_mixed_formula(formula: str) -> ParsedFormula:
    """Parse a minimal mixed formula with one random intercept.

    Supported syntax is strictly:

    ``y ~ fixed_terms + (1|group)``

    Parameters
    ----------
    formula : str
        Formula string.

    Returns
    -------
    ParsedFormula
        Parsed formula container.

    Usage example
    -------------
        parsed = parse_mixed_formula("y ~ condition + latency + (1|subject)")
    """
    cleaned_formula = " ".join(formula.strip().split())
    match = re.fullmatch(r"y\s*~\s*(.+)\s*\+\s*\(1\s*\|\s*([A-Za-z_][A-Za-z0-9_]*)\)", cleaned_formula)
    if match is None:
        raise ValueError(
            "Only formulas of the form 'y ~ fixed_terms + (1|group)' are currently supported."
        )
    fixed_terms = match.group(1).strip()
    group_variable = match.group(2).strip()
    fixed_formula = f"y ~ {fixed_terms}"
    return ParsedFormula(
        fixed_formula=fixed_formula,
        group_variable=group_variable,
        original_formula=cleaned_formula,
    )
