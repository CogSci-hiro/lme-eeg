from lmeeeg.core.formulas import parse_mixed_formula


def test_parse_mixed_formula() -> None:
    parsed = parse_mixed_formula("y ~ condition + latency + (1|subject)")
    assert parsed.fixed_formula == "y ~ condition + latency"
    assert parsed.group_variable == "subject"


def test_parse_mixed_formula_preserves_lme4_random_effects() -> None:
    formula = "y ~ cond + (1 + cond | subject) + (1 | item)"
    parsed = parse_mixed_formula(formula)
    assert parsed.original_formula == formula
    assert parsed.fixed_formula == "y ~ cond"
    assert parsed.group_variable == "subject"
