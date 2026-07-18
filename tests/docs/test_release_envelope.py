from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_release_1_validation_envelope_is_documented() -> None:
    readme = " ".join((REPO_ROOT / "README.md").read_text().split())
    calibration = " ".join((REPO_ROOT / "docs" / "CALIBRATION.md").read_text().split())

    assert "Release 1 validates the fast marginal-OLS path for crossed random intercepts" in readme
    assert "Random slopes are refused pending the real-LMM path" in readme
    assert "Release 1 validates the fast marginal-OLS path for crossed random intercepts" in calibration
    assert "Random slopes are INVALID on the fast path" in calibration
