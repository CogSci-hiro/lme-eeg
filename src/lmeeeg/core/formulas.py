import re
from dataclasses import dataclass


@dataclass(slots=True)
class RandomEffectTerm:
    """Structured lme4 random-effect term."""

    expression: str
    group_variable: str

    @property
    def has_random_slope(self) -> bool:
        pieces = [piece.strip() for piece in _split_top_level_plus(self.expression)]
        return any(piece not in {"1", ""} for piece in pieces)


@dataclass(slots=True)
class ParsedFormula:
    """Parsed mixed-model style formula.

    Attributes
    ----------
    fixed_formula : str
        Patsy-compatible fixed-effects formula.
    group_variable : str
        Primary grouping variable. For full lme4 random-effect formulas this is
        the first grouping variable encountered; only the simple ``(1|group)``
        path lowers this into a random-intercept design.
    original_formula : str
        Original formula string.
    random_effects : tuple[RandomEffectTerm, ...]
        Structured random-effect terms in formula order.
    """

    fixed_formula: str
    group_variable: str
    original_formula: str
    random_effects: tuple[RandomEffectTerm, ...]


# ==============================
# Formula parsing
# ==============================

def parse_mixed_formula(formula: str) -> ParsedFormula:
    """Parse a mixed formula while preserving lme4 random-effect terms.

    The fixed-effect RHS is exposed as a Patsy-compatible formula. Random-effect
    terms are retained only in ``original_formula`` so lme4-capable backends can
    consume the full formula verbatim.

    The existing single-intercept path, ``y ~ fixed_terms + (1|group)``, remains
    unchanged for downstream statsmodels code.

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
    original_formula = formula.strip()
    lhs_rhs = re.fullmatch(r"\s*y\s*~\s*(.+)\s*", original_formula)
    if lhs_rhs is None:
        raise ValueError("Mixed formulas must have response variable 'y'.")

    rhs = lhs_rhs.group(1).strip()
    terms = _split_top_level_plus(rhs)
    random_terms = [term for term in terms if _is_random_effect_term(term)]
    if not random_terms:
        raise ValueError("Mixed formulas must include at least one lme4-style random-effect term.")

    fixed_terms = [term for term in terms if not _is_random_effect_term(term)]
    fixed_rhs = " + ".join(fixed_terms) if fixed_terms else "1"
    parsed_random_terms = tuple(_parse_random_effect_term(term) for term in random_terms)
    group_variable = parsed_random_terms[0].group_variable
    return ParsedFormula(
        fixed_formula=f"y ~ {fixed_rhs}",
        group_variable=group_variable,
        original_formula=original_formula,
        random_effects=parsed_random_terms,
    )


def _split_top_level_plus(rhs: str) -> list[str]:
    terms: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(rhs):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in formula RHS: {rhs}")
        elif character == "+" and depth == 0:
            term = rhs[start:index].strip()
            if term:
                terms.append(term)
            start = index + 1
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in formula RHS: {rhs}")
    final_term = rhs[start:].strip()
    if final_term:
        terms.append(final_term)
    return terms


def _is_random_effect_term(term: str) -> bool:
    return bool(re.fullmatch(r"\(.+\|.+\)", term.strip()))


def _extract_group_variable(random_term: str) -> str:
    return _parse_random_effect_term(random_term).group_variable


def _parse_random_effect_term(random_term: str) -> RandomEffectTerm:
    match = re.fullmatch(r"\((.+)\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", random_term.strip())
    if match is None:
        raise ValueError(f"Could not parse grouping variable from random-effect term: {random_term}")
    expression, group_variable = match.groups()
    return RandomEffectTerm(
        expression=expression.strip(),
        group_variable=group_variable,
    )
