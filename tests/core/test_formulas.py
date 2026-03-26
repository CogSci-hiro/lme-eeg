from lmeeeg.core.formulas import parse_mixed_formula


def test_parse_mixed_formula() -> None:
    parsed = parse_mixed_formula("y ~ condition + latency + (1|subject)")
    assert parsed.fixed_formula == "y ~ condition + latency"
    assert parsed.group_variable == "subject"
