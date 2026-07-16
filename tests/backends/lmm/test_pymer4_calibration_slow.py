import os

import numpy as np
import pandas as pd
import pytest

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect

from tests.backends.lmm.test_pymer4_backend import _require_pymer4


ALPHA = 0.05
FWER_MARGIN = 0.03
SLOPE_FAILURE_LIMIT = 0.08


def _calibration_settings() -> tuple[int, int]:
    n_sims = int(os.environ.get("LMEEG_CALIBRATION_N_SIMS", "300"))
    n_permutations = int(os.environ.get("LMEEG_CALIBRATION_N_PERMUTATIONS", "64"))
    return n_sims, n_permutations


def _simulate_null_or_power(seed: int, random_slope: bool, fixed_effect: float) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = [f"s{i}" for i in range(6)]
    items = [f"i{i}" for i in range(5)]
    rows = []
    for subject in subjects:
        for item in items:
            for condition in ["A", "B"]:
                rows.append({"subject": subject, "item": item, "cond": condition})
    metadata = pd.DataFrame(rows)
    metadata["cond"] = pd.Categorical(metadata["cond"], categories=["A", "B"], ordered=False)
    cond = (metadata["cond"] == "B").to_numpy(dtype=float)
    subject_intercepts = dict(zip(subjects, rng.normal(0.0, 0.5, len(subjects)), strict=True))
    item_intercepts = dict(zip(items, rng.normal(0.0, 0.35, len(items)), strict=True))
    subject_slopes = dict(zip(subjects, rng.normal(0.0, 0.45, len(subjects)), strict=True))
    random_part = np.array(
        [
            subject_intercepts[row.subject]
            + item_intercepts[row.item]
            + (subject_slopes[row.subject] * cond_value if random_slope else 0.0)
            for row, cond_value in zip(metadata.itertuples(index=False), cond, strict=True)
        ]
    )
    eeg = np.empty((len(metadata), 2, 2), dtype=float)
    for location in range(2):
        for time in range(2):
            signal = fixed_effect + 0.03 * location - 0.02 * time
            eeg[:, location, time] = (
                1.0
                + signal * cond
                + random_part
                + rng.normal(0.0, 0.25, len(metadata))
            )
    return eeg, metadata


def _run_pipeline_rejects(seed: int, correction: str, random_slope: bool, fixed_effect: float, n_permutations: int) -> bool:
    eeg, metadata = _simulate_null_or_power(seed=seed, random_slope=random_slope, fixed_effect=fixed_effect)
    formula = "y ~ cond + (1 + cond | subject) + (1 | item)" if random_slope else "y ~ cond + (1 | subject) + (1 | item)"
    fit_result = fit_lmm_mass_univariate(
        eeg=eeg,
        metadata=metadata,
        formula=formula,
        variable_types={"cond": "categorical", "subject": "group", "item": "group"},
        config=FitConfig(show_progress=False, lmm_backend_name="pymer4"),
    )
    inference = permute_fixed_effect(
        fit_result,
        effect="cond[T.B]",
        correction=correction,
        n_permutations=n_permutations,
        seed=seed + 100_000,
        threshold=2.0 if correction == "cluster" else None,
        verbose=False,
    )
    return bool(np.nanmin(inference.corrected_p_values) <= ALPHA)


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["maxstat", "cluster", "tfce"])
def test_pymer4_calibration_c1_crossed_intercepts_h0_fwer(correction: str) -> None:
    _require_pymer4()
    n_sims, n_permutations = _calibration_settings()
    rejections = [
        _run_pipeline_rejects(
            seed=10_000 + sim,
            correction=correction,
            random_slope=False,
            fixed_effect=0.0,
            n_permutations=n_permutations,
        )
        for sim in range(n_sims)
    ]
    empirical_fwer = float(np.mean(rejections))
    assert empirical_fwer <= ALPHA + FWER_MARGIN, (
        f"C1 empirical FWER for {correction} was {empirical_fwer:.3f} "
        f"with n_sims={n_sims}, n_permutations={n_permutations}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["maxstat", "cluster", "tfce"])
def test_pymer4_calibration_c2_random_slope_h0_fwer_report(correction: str) -> None:
    _require_pymer4()
    n_sims, n_permutations = _calibration_settings()
    rejections = [
        _run_pipeline_rejects(
            seed=20_000 + sim,
            correction=correction,
            random_slope=True,
            fixed_effect=0.0,
            n_permutations=n_permutations,
        )
        for sim in range(n_sims)
    ]
    empirical_fwer = float(np.mean(rejections))
    assert empirical_fwer <= SLOPE_FAILURE_LIMIT, (
        f"C2 empirical FWER for {correction} was {empirical_fwer:.3f} "
        f"with n_sims={n_sims}, n_permutations={n_permutations}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["maxstat", "cluster", "tfce"])
def test_pymer4_calibration_c3_power_sanity(correction: str) -> None:
    _require_pymer4()
    n_sims, n_permutations = _calibration_settings()
    rejections = [
        _run_pipeline_rejects(
            seed=30_000 + sim,
            correction=correction,
            random_slope=False,
            fixed_effect=0.45,
            n_permutations=n_permutations,
        )
        for sim in range(n_sims)
    ]
    power = float(np.mean(rejections))
    if n_sims < 20:
        assert 0.0 <= power <= 1.0
        return
    assert power >= ALPHA + 0.10, (
        f"C3 power for {correction} was {power:.3f} "
        f"with n_sims={n_sims}, n_permutations={n_permutations}"
    )
