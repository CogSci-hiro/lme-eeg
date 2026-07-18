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
NULL_GUARD_ALPHA = 0.05


def _calibration_settings() -> tuple[int, int]:
    n_sims = int(os.environ.get("LMEEG_CALIBRATION_N_SIMS", "300"))
    n_permutations = int(os.environ.get("LMEEG_CALIBRATION_N_PERMUTATIONS", "64"))
    return n_sims, n_permutations


def _mc_se(p_value: float, n_sims: int) -> float:
    return float(np.sqrt(p_value * (1.0 - p_value) / n_sims))


def _condition_effect_shape(location: int, time: int) -> float:
    return 1.0 + 0.03 * location - 0.02 * time


def _simulate_null_or_power(
    seed: int,
    random_slope: bool,
    fixed_effect: float,
    n_subjects: int = 6,
    n_items: int = 5,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = [f"s{i}" for i in range(n_subjects)]
    items = [f"i{i}" for i in range(n_items)]
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
            beta_feature = fixed_effect * _condition_effect_shape(location, time)
            eeg[:, location, time] = (
                1.0
                + beta_feature * cond
                + random_part
                + rng.normal(0.0, 0.25, len(metadata))
            )
    return eeg, metadata


def _lmer_formula(random_slope: bool) -> str:
    return (
        "y ~ cond + (1 + cond | subject) + (1 | item)"
        if random_slope
        else "y ~ cond + (1 | subject) + (1 | item)"
    )


def _fit_lmer_condition_effect(
    y: np.ndarray,
    metadata: pd.DataFrame,
    random_slope: bool,
) -> tuple[float, float, float]:
    from lmeeeg.backends.lmm.pymer4_backend import _factor_levels, _resolve_term

    import polars as pl
    from pymer4.models import lmer

    data = metadata.copy(deep=False)
    data["y"] = y
    model = lmer(_lmer_formula(random_slope), data=pl.from_pandas(data))
    model.set_factors(_factor_levels(data))
    model.fit(summary=False, verbose=False)
    coefs = model.result_fit
    term = _resolve_term("cond[T.B]", coefs["term"].to_list())
    if term is None:
        raise AssertionError(f"Could not resolve cond[T.B] in lmer table {coefs['term'].to_list()!r}.")
    row = coefs.filter(pl.col("term") == term).row(0, named=True)
    return float(row["estimate"]), float(row["std_error"]), float(row["p_value"])


def _assert_generator_h0_is_null_at_every_feature(random_slope: bool, seed_offset: int) -> None:
    _require_pymer4()
    n_draws = int(os.environ.get("LMEEG_NULL_GUARD_N_DRAWS", "200"))
    n_subjects = int(os.environ.get("LMEEG_NULL_GUARD_N_SUBJECTS", "6"))
    n_items = int(os.environ.get("LMEEG_NULL_GUARD_N_ITEMS", "5"))
    progress_every = int(os.environ.get("LMEEG_PROGRESS_EVERY", "0"))
    estimates = np.empty((n_draws, 2, 2), dtype=float)
    standard_errors = np.empty((n_draws, 2, 2), dtype=float)
    p_values = np.empty((n_draws, 2, 2), dtype=float)

    for draw in range(n_draws):
        eeg, metadata = _simulate_null_or_power(
            seed=seed_offset + draw,
            random_slope=random_slope,
            fixed_effect=0.0,
            n_subjects=n_subjects,
            n_items=n_items,
        )
        for location in range(2):
            for time in range(2):
                estimate, standard_error, p_value = _fit_lmer_condition_effect(
                    eeg[:, location, time],
                    metadata,
                    random_slope=random_slope,
                )
                estimates[draw, location, time] = estimate
                standard_errors[draw, location, time] = standard_error
                p_values[draw, location, time] = p_value
        if progress_every and ((draw + 1) % progress_every == 0 or draw + 1 == n_draws):
            print(f"PROGRESS NULL_GUARD draw={draw + 1}/{n_draws}", flush=True)

    rejection_rate_margin = 3.0 * np.sqrt(NULL_GUARD_ALPHA * (1.0 - NULL_GUARD_ALPHA) / n_draws)
    for location in range(2):
        for time in range(2):
            feature_estimates = estimates[:, location, time]
            feature_standard_errors = standard_errors[:, location, time]
            feature_p_values = p_values[:, location, time]
            mean_tolerance = 3.0 * float(np.mean(feature_standard_errors)) / np.sqrt(n_draws)
            rejection_rate = float(np.mean(feature_p_values < NULL_GUARD_ALPHA))
            assert abs(float(np.mean(feature_estimates))) <= mean_tolerance
            assert abs(rejection_rate - NULL_GUARD_ALPHA) <= rejection_rate_margin


def test_pymer4_calibration_generator_h0_is_null_at_every_feature() -> None:
    _assert_generator_h0_is_null_at_every_feature(random_slope=False, seed_offset=70_000)


def test_pymer4_calibration_slope_generator_h0_is_null_at_every_feature() -> None:
    _assert_generator_h0_is_null_at_every_feature(random_slope=True, seed_offset=80_000)


def _run_pipeline_rejects(
    seed: int,
    correction: str,
    random_slope: bool,
    fixed_effect: float,
    n_permutations: int,
    permutation_scheme: str | None = None,
    n_subjects: int = 6,
    n_items: int = 5,
) -> bool:
    eeg, metadata = _simulate_null_or_power(
        seed=seed,
        random_slope=random_slope,
        fixed_effect=fixed_effect,
        n_subjects=n_subjects,
        n_items=n_items,
    )
    formula = _lmer_formula(random_slope)
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
        permutation_scheme=permutation_scheme,
    )
    return bool(np.nanmin(inference.corrected_p_values) <= ALPHA)


def _estimate_h0_fwer(
    correction: str,
    permutation_scheme: str,
    n_sims: int,
    n_permutations: int,
    n_subjects: int,
    n_items: int,
    seed_offset: int,
    random_slope: bool,
    label: str,
) -> tuple[float, float]:
    progress_every = int(os.environ.get("LMEEG_PROGRESS_EVERY", "0"))
    rejections = []
    for sim in range(n_sims):
        rejected = _run_pipeline_rejects(
            seed=seed_offset + sim,
            correction=correction,
            random_slope=random_slope,
            fixed_effect=0.0,
            n_permutations=n_permutations,
            permutation_scheme=permutation_scheme,
            n_subjects=n_subjects,
            n_items=n_items,
        )
        rejections.append(rejected)
        if progress_every and ((sim + 1) % progress_every == 0 or sim + 1 == n_sims):
            current = float(np.mean(rejections))
            print(
                f"PROGRESS {label} size={n_subjects}x{n_items} scheme={permutation_scheme} "
                f"backend={correction} sim={sim + 1}/{n_sims} "
                f"rejections={int(np.sum(rejections))} fwer={current:.3f}",
                flush=True,
            )
    fwer = float(np.mean(rejections))
    return fwer, _mc_se(fwer, n_sims)


def _estimate_c1_fwer(
    correction: str,
    permutation_scheme: str,
    n_sims: int,
    n_permutations: int,
    n_subjects: int,
    n_items: int,
    seed_offset: int,
) -> tuple[float, float]:
    return _estimate_h0_fwer(
        correction=correction,
        permutation_scheme=permutation_scheme,
        n_sims=n_sims,
        n_permutations=n_permutations,
        n_subjects=n_subjects,
        n_items=n_items,
        seed_offset=seed_offset,
        random_slope=False,
        label="C1",
    )


def _estimate_c2_fwer(
    correction: str,
    permutation_scheme: str,
    n_sims: int,
    n_permutations: int,
    n_subjects: int,
    n_items: int,
    seed_offset: int,
) -> tuple[float, float]:
    return _estimate_h0_fwer(
        correction=correction,
        permutation_scheme=permutation_scheme,
        n_sims=n_sims,
        n_permutations=n_permutations,
        n_subjects=n_subjects,
        n_items=n_items,
        seed_offset=seed_offset,
        random_slope=True,
        label="C2",
    )


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["maxstat", "cluster", "tfce"])
@pytest.mark.parametrize("permutation_scheme", ["free", "within_subject"])
def test_pymer4_d2_c1_scheme_grid(correction: str, permutation_scheme: str) -> None:
    _require_pymer4()
    n_sims = int(os.environ.get("LMEEG_D2_N_SIMS", "300"))
    n_permutations = int(os.environ.get("LMEEG_D2_N_PERMUTATIONS", "500"))
    fwer, se = _estimate_c1_fwer(
        correction=correction,
        permutation_scheme=permutation_scheme,
        n_sims=n_sims,
        n_permutations=n_permutations,
        n_subjects=6,
        n_items=5,
        seed_offset=40_000,
    )
    print(
        f"RESULT D2 size=6x5 scheme={permutation_scheme} backend={correction} "
        f"fwer={fwer:.3f} mc_se={se:.3f} n_sims={n_sims} n_permutations={n_permutations}"
    )
    assert 0.0 <= fwer <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["maxstat", "cluster", "tfce"])
@pytest.mark.parametrize(("n_subjects", "n_items"), [(6, 5), (12, 10), (24, 20), (36, 30)])
def test_pymer4_d3_c1_size_sweep(correction: str, n_subjects: int, n_items: int) -> None:
    _require_pymer4()
    n_sims = int(os.environ.get("LMEEG_D3_N_SIMS", "60"))
    n_permutations = int(os.environ.get("LMEEG_D3_N_PERMUTATIONS", "500"))
    permutation_scheme = os.environ.get("LMEEG_D3_PERMUTATION_SCHEME", "within_subject")
    fwer, se = _estimate_c1_fwer(
        correction=correction,
        permutation_scheme=permutation_scheme,
        n_sims=n_sims,
        n_permutations=n_permutations,
        n_subjects=n_subjects,
        n_items=n_items,
        seed_offset=50_000 + n_subjects * 100 + n_items,
    )
    print(
        f"RESULT D3 size={n_subjects}x{n_items} scheme={permutation_scheme} backend={correction} "
        f"fwer={fwer:.3f} mc_se={se:.3f} n_sims={n_sims} n_permutations={n_permutations}"
    )
    assert 0.0 <= fwer <= 1.0


@pytest.mark.slow
def test_pymer4_release_c1_maxstat_regression_guard() -> None:
    _require_pymer4()
    n_sims = int(os.environ.get("LMEEG_RELEASE_C1_N_SIMS", "200"))
    n_permutations = int(os.environ.get("LMEEG_RELEASE_C1_N_PERMUTATIONS", "500"))
    fwer, se = _estimate_c1_fwer(
        correction="maxstat",
        permutation_scheme="within_subject",
        n_sims=n_sims,
        n_permutations=n_permutations,
        n_subjects=6,
        n_items=5,
        seed_offset=110_000,
    )
    print(
        f"RESULT RELEASE_C1 size=6x5 scheme=within_subject backend=maxstat "
        f"fwer={fwer:.3f} mc_se={se:.3f} n_sims={n_sims} n_permutations={n_permutations}"
    )
    assert n_sims >= 200
    assert n_permutations >= 500
    assert fwer <= 0.08


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["maxstat", "cluster", "tfce"])
@pytest.mark.parametrize(("n_subjects", "n_items"), [(6, 5), (12, 10), (24, 20), (36, 30)])
def test_pymer4_c2_random_slope_size_sweep(correction: str, n_subjects: int, n_items: int) -> None:
    _require_pymer4()
    n_sims = int(os.environ.get("LMEEG_C2_N_SIMS", "300"))
    n_permutations = int(os.environ.get("LMEEG_C2_N_PERMUTATIONS", "1000"))
    permutation_scheme = os.environ.get("LMEEG_C2_PERMUTATION_SCHEME", "within_subject")
    fwer, se = _estimate_c2_fwer(
        correction=correction,
        permutation_scheme=permutation_scheme,
        n_sims=n_sims,
        n_permutations=n_permutations,
        n_subjects=n_subjects,
        n_items=n_items,
        seed_offset=90_000 + n_subjects * 100 + n_items,
    )
    print(
        f"RESULT C2 size={n_subjects}x{n_items} scheme={permutation_scheme} backend={correction} "
        f"fwer={fwer:.3f} mc_se={se:.3f} n_sims={n_sims} n_permutations={n_permutations}"
    )
    assert 0.0 <= fwer <= 1.0


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
    progress_every = int(os.environ.get("LMEEG_PROGRESS_EVERY", "0"))
    rejections = []
    for sim in range(n_sims):
        rejected = _run_pipeline_rejects(
            seed=30_000 + sim,
            correction=correction,
            random_slope=False,
            fixed_effect=0.45,
            n_permutations=n_permutations,
            permutation_scheme="within_subject",
        )
        rejections.append(rejected)
        if progress_every and ((sim + 1) % progress_every == 0 or sim + 1 == n_sims):
            current = float(np.mean(rejections))
            print(
                f"PROGRESS C3 size=6x5 scheme=within_subject backend={correction} "
                f"sim={sim + 1}/{n_sims} detections={int(np.sum(rejections))} power={current:.3f}",
                flush=True,
            )
    power = float(np.mean(rejections))
    print(
        f"RESULT C3 size=6x5 scheme=within_subject backend={correction} "
        f"power={power:.3f} mc_se={_mc_se(power, n_sims):.3f} "
        f"n_sims={n_sims} n_permutations={n_permutations}"
    )
    if n_sims < 20:
        assert 0.0 <= power <= 1.0
        return
    assert power >= ALPHA + 0.10, (
        f"C3 power for {correction} was {power:.3f} "
        f"with n_sims={n_sims}, n_permutations={n_permutations}"
    )
