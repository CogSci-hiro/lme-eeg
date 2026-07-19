import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from lmeeeg.backends.correction._regression import cluster_outputs_to_masks, configure_mne_runtime
from lmeeeg.backends.correction._regression import make_permutation_rng, max_permutation_p_values

from tests.backends.lmm.test_pymer4_backend import _require_pymer4
from tests.backends.lmm.test_pymer4_calibration_slow import (
    ALPHA,
    _assert_generator_h0_is_null_at_every_feature,
    _mc_se,
    _simulate_null_or_power,
)


def _real_lmm_settings() -> tuple[int, int]:
    n_sims = int(os.environ.get("LMEEG_REAL_LMM_N_SIMS", "3"))
    n_permutations = int(os.environ.get("LMEEG_REAL_LMM_N_PERMUTATIONS", "8"))
    return n_sims, n_permutations


def _lmer_formula(random_slope: bool) -> str:
    return (
        "y ~ cond + (1 + cond | subject) + (1 | item)"
        if random_slope
        else "y ~ cond + (1 | subject) + (1 | item)"
    )


def _fit_lmer_condition_stat_maps(
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    random_slope: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from lmeeeg.backends.lmm.pymer4_backend import _factor_levels, _resolve_term

    import polars as pl
    from pymer4.models import lmer

    n_locations, n_times = eeg.shape[1], eeg.shape[2]
    t_map = np.empty((n_locations, n_times), dtype=float)
    p_map = np.empty((n_locations, n_times), dtype=float)
    for location in range(n_locations):
        for time in range(n_times):
            data = metadata.copy(deep=False)
            data["y"] = eeg[:, location, time]
            model = lmer(_lmer_formula(random_slope), data=pl.from_pandas(data))
            model.set_factors(_factor_levels(data))
            model.fit(summary=False, verbose=False)
            coefs = model.result_fit
            term = _resolve_term("cond[T.B]", coefs["term"].to_list())
            if term is None:
                raise AssertionError(f"Could not resolve cond[T.B] in lmer table {coefs['term'].to_list()!r}.")
            row = coefs.filter(pl.col("term") == term).row(0, named=True)
            t_map[location, time] = float(row["t_stat"])
            p_map[location, time] = float(row["p_value"])
    return t_map, p_map


def _fit_lmer_condition_t_map(
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    random_slope: bool,
) -> np.ndarray:
    t_map, _ = _fit_lmer_condition_stat_maps(eeg=eeg, metadata=metadata, random_slope=random_slope)
    return t_map


def _permute_condition_within_subject(
    metadata: pd.DataFrame,
    rng: np.random.Generator | np.random.RandomState,
) -> pd.DataFrame:
    permuted = metadata.copy(deep=True)
    condition = permuted["cond"].astype(str).to_numpy(copy=True)
    group_codes = pd.Categorical(permuted["subject"]).codes
    for group_code in np.unique(group_codes):
        group_indices = np.flatnonzero(group_codes == group_code)
        if group_indices.size > 1:
            condition[group_indices] = condition[group_indices[rng.permutation(group_indices.size)]]
    categories = list(metadata["cond"].cat.categories)
    permuted["cond"] = pd.Categorical(condition, categories=categories, ordered=False)
    return permuted


def _real_lmm_t_null_maps(
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    random_slope: bool,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = make_permutation_rng(seed)
    start = time.perf_counter()
    observed_t = _fit_lmer_condition_t_map(eeg=eeg, metadata=metadata, random_slope=random_slope)
    null_t = np.empty((n_permutations,) + observed_t.shape, dtype=float)
    for permutation_index in range(n_permutations):
        permuted_metadata = _permute_condition_within_subject(metadata=metadata, rng=rng)
        null_t[permutation_index] = _fit_lmer_condition_t_map(
            eeg=eeg,
            metadata=permuted_metadata,
            random_slope=random_slope,
        )
    return observed_t, null_t, time.perf_counter() - start


def _maxstat_rejects(observed_t: np.ndarray, null_t: np.ndarray) -> bool:
    null_distribution = np.max(np.abs(null_t), axis=(1, 2))
    corrected = max_permutation_p_values(null_distribution, observed_t)
    return bool(np.nanmin(corrected) <= ALPHA)


def _cluster_rejects(observed_t: np.ndarray, null_t: np.ndarray) -> bool:
    configure_mne_runtime()
    from mne.stats.cluster_level import _find_clusters

    threshold = 2.0
    sample_shape = (observed_t.shape[1], observed_t.shape[0])
    raw_clusters, cluster_stats = _find_clusters(observed_t.T, threshold=threshold, tail=0)
    cluster_masks = cluster_outputs_to_masks(raw_clusters, sample_shape)
    null_distribution = np.zeros(null_t.shape[0], dtype=float)
    for permutation_index, permuted_t in enumerate(null_t):
        _, permuted_cluster_stats = _find_clusters(permuted_t.T, threshold=threshold, tail=0)
        null_distribution[permutation_index] = (
            float(np.max(np.abs(permuted_cluster_stats))) if len(permuted_cluster_stats) else 0.0
        )
    if not cluster_masks:
        return False
    cluster_p_values = np.asarray(
        [
            (1 + np.sum(null_distribution >= abs(cluster_stat))) / (null_t.shape[0] + 1)
            for cluster_stat in cluster_stats
        ],
        dtype=float,
    )
    corrected = np.ones_like(observed_t, dtype=float)
    for cluster_mask, cluster_p_value in zip(cluster_masks, cluster_p_values):
        corrected[cluster_mask.T] = np.minimum(corrected[cluster_mask.T], cluster_p_value)
    return bool(np.nanmin(corrected) <= ALPHA)


def _tfce_rejects(observed_t: np.ndarray, null_t: np.ndarray) -> bool:
    configure_mne_runtime()
    from mne.stats.cluster_level import _find_clusters

    threshold = {"start": 0.0, "step": 0.2}
    _, observed_tfce = _find_clusters(observed_t.T, threshold=threshold, tail=0)
    observed_tfce = np.asarray(observed_tfce, dtype=float).reshape(observed_t.shape[1], observed_t.shape[0]).T
    null_distribution = np.zeros(null_t.shape[0], dtype=float)
    for permutation_index, permuted_t in enumerate(null_t):
        _, permuted_tfce = _find_clusters(permuted_t.T, threshold=threshold, tail=0)
        null_distribution[permutation_index] = float(np.max(np.abs(permuted_tfce)))
    corrected = (
        1
        + np.sum(null_distribution[:, None, None] >= np.abs(observed_tfce)[None, :, :], axis=0)
    ) / (null_t.shape[0] + 1)
    return bool(np.nanmin(corrected) <= ALPHA)


def _real_lmm_refit_rejections(
    seed: int,
    random_slope: bool,
    fixed_effect: float,
    n_permutations: int,
    n_subjects: int,
    n_items: int,
) -> tuple[dict[str, bool], float]:
    eeg, metadata = _simulate_null_or_power(
        seed=seed,
        random_slope=random_slope,
        fixed_effect=fixed_effect,
        n_subjects=n_subjects,
        n_items=n_items,
    )
    observed_t, null_t, elapsed = _real_lmm_t_null_maps(
        eeg=eeg,
        metadata=metadata,
        random_slope=random_slope,
        n_permutations=n_permutations,
        seed=seed + 700_000,
    )
    return {
        "maxstat": _maxstat_rejects(observed_t, null_t),
        "cluster": _cluster_rejects(observed_t, null_t),
        "tfce": _tfce_rejects(observed_t, null_t),
    }, elapsed


def _estimate_parametric_feature_rates(
    label: str,
    random_slope: bool,
    fixed_effect: float,
    n_subjects: int,
    n_items: int,
    n_sims: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    progress_every = int(os.environ.get("LMEEG_PROGRESS_EVERY", "0"))
    p_values = np.empty((n_sims, 2, 2), dtype=float)
    start = time.perf_counter()
    for sim in range(n_sims):
        eeg, metadata = _simulate_null_or_power(
            seed=seed_offset + sim,
            random_slope=random_slope,
            fixed_effect=fixed_effect,
            n_subjects=n_subjects,
            n_items=n_items,
        )
        _, p_map = _fit_lmer_condition_stat_maps(eeg=eeg, metadata=metadata, random_slope=random_slope)
        p_values[sim] = p_map
        if progress_every and ((sim + 1) % progress_every == 0 or sim + 1 == n_sims):
            rates = np.mean(p_values[: sim + 1] < ALPHA, axis=0)
            print(
                f"PROGRESS REAL_LMM_PARAM {label} size={n_subjects}x{n_items} "
                f"sim={sim + 1}/{n_sims} mean_rate={float(np.mean(rates)):.3f}",
                flush=True,
            )
    return np.mean(p_values < ALPHA, axis=0), p_values, time.perf_counter() - start


def _parametric_settings() -> int:
    return int(os.environ.get("LMEEG_REAL_LMM_PARAMETRIC_N_SIMS", "300"))


@dataclass(frozen=True)
class _FixedThetaFeature:
    y: np.ndarray
    random_effect_design_factor: np.ndarray
    covariance_cholesky: np.ndarray
    sigma: float
    observed_lmer_t: float
    observed_gls_t: float


def _fixed_theta_settings() -> tuple[int, int]:
    n_sims = int(os.environ.get("LMEEG_FIXED_THETA_N_SIMS", "300"))
    n_permutations = int(os.environ.get("LMEEG_FIXED_THETA_N_PERMUTATIONS", "500"))
    return n_sims, n_permutations


def _condition_fixed_design(condition: np.ndarray) -> np.ndarray:
    condition = np.asarray(condition).astype(str)
    return np.column_stack([np.ones(condition.shape[0], dtype=float), (condition == "B").astype(float)])


def _fixed_theta_solve(feature: _FixedThetaFeature, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    was_1d = values.ndim == 1
    if was_1d:
        values = values[:, None]
    random_factor = feature.random_effect_design_factor
    projected = random_factor.T @ values
    correction = np.linalg.solve(
        feature.covariance_cholesky.T,
        np.linalg.solve(feature.covariance_cholesky, projected),
    )
    solved = (values - random_factor @ correction) / (feature.sigma ** 2)
    return solved[:, 0] if was_1d else solved


def _fixed_theta_condition_t(feature: _FixedThetaFeature, fixed_design: np.ndarray) -> float:
    fixed_design = np.asarray(fixed_design, dtype=float)
    v_inv_x = _fixed_theta_solve(feature, fixed_design)
    v_inv_y = _fixed_theta_solve(feature, feature.y)
    xt_vinv_x = fixed_design.T @ v_inv_x
    covariance = np.linalg.inv(xt_vinv_x)
    beta = covariance @ (fixed_design.T @ v_inv_y)
    standard_error = float(np.sqrt(covariance[1, 1]))
    return float(beta[1] / standard_error)


def _fit_fixed_theta_feature(
    y: np.ndarray,
    metadata: pd.DataFrame,
    random_slope: bool,
) -> _FixedThetaFeature:
    from lmeeeg.backends.lmm.pymer4_backend import _factor_levels, _resolve_term

    import polars as pl
    import rpy2.robjects as ro
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

    get_me = ro.r["getME"]
    as_matrix = ro.r["as.matrix"]
    x_matrix = np.asarray(get_me(model.r_model, "X"), dtype=float)
    z_matrix = np.asarray(as_matrix(get_me(model.r_model, "Z")), dtype=float)
    lambda_matrix = np.asarray(as_matrix(get_me(model.r_model, "Lambda")), dtype=float)
    sigma = float(model.result_fit_stats["sigma"][0])
    random_effect_design_factor = z_matrix @ lambda_matrix
    covariance_cholesky = np.linalg.cholesky(
        np.eye(random_effect_design_factor.shape[1], dtype=float)
        + random_effect_design_factor.T @ random_effect_design_factor
    )
    observed_lmer_t = float(row["t_stat"])
    placeholder = _FixedThetaFeature(
        y=np.asarray(y, dtype=float),
        random_effect_design_factor=random_effect_design_factor,
        covariance_cholesky=covariance_cholesky,
        sigma=sigma,
        observed_lmer_t=observed_lmer_t,
        observed_gls_t=np.nan,
    )
    observed_gls_t = _fixed_theta_condition_t(placeholder, x_matrix)
    return _FixedThetaFeature(
        y=placeholder.y,
        random_effect_design_factor=random_effect_design_factor,
        covariance_cholesky=covariance_cholesky,
        sigma=sigma,
        observed_lmer_t=observed_lmer_t,
        observed_gls_t=observed_gls_t,
    )


def _fit_fixed_theta_map(
    eeg: np.ndarray,
    metadata: pd.DataFrame,
    random_slope: bool,
) -> tuple[np.ndarray, np.ndarray, list[list[_FixedThetaFeature]]]:
    n_locations, n_times = eeg.shape[1], eeg.shape[2]
    lmer_t = np.empty((n_locations, n_times), dtype=float)
    gls_t = np.empty((n_locations, n_times), dtype=float)
    features: list[list[_FixedThetaFeature]] = []
    for location in range(n_locations):
        row_features = []
        for time_index in range(n_times):
            feature = _fit_fixed_theta_feature(
                y=eeg[:, location, time_index],
                metadata=metadata,
                random_slope=random_slope,
            )
            lmer_t[location, time_index] = feature.observed_lmer_t
            gls_t[location, time_index] = feature.observed_gls_t
            row_features.append(feature)
        features.append(row_features)
    return lmer_t, gls_t, features


def _permuted_condition_values_within_subject(
    metadata: pd.DataFrame,
    rng: np.random.Generator | np.random.RandomState,
) -> np.ndarray:
    condition = metadata["cond"].astype(str).to_numpy(copy=True)
    group_codes = pd.Categorical(metadata["subject"]).codes
    for group_code in np.unique(group_codes):
        group_indices = np.flatnonzero(group_codes == group_code)
        if group_indices.size > 1:
            condition[group_indices] = condition[group_indices[rng.permutation(group_indices.size)]]
    return condition


def _fixed_theta_null_maps(
    features: list[list[_FixedThetaFeature]],
    metadata: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    rng = make_permutation_rng(seed)
    n_locations = len(features)
    n_times = len(features[0])
    null_t = np.empty((n_permutations, n_locations, n_times), dtype=float)
    for permutation_index in range(n_permutations):
        permuted_condition = _permuted_condition_values_within_subject(metadata=metadata, rng=rng)
        fixed_design = _condition_fixed_design(permuted_condition)
        for location in range(n_locations):
            for time_index in range(n_times):
                null_t[permutation_index, location, time_index] = _fixed_theta_condition_t(
                    features[location][time_index],
                    fixed_design,
                )
    return null_t


def _fixed_theta_rejections(
    seed: int,
    random_slope: bool,
    fixed_effect: float,
    n_subjects: int,
    n_items: int,
    n_permutations: int,
) -> tuple[dict[str, bool], float, float]:
    eeg, metadata = _simulate_null_or_power(
        seed=seed,
        random_slope=random_slope,
        fixed_effect=fixed_effect,
        n_subjects=n_subjects,
        n_items=n_items,
    )
    start = time.perf_counter()
    lmer_t, gls_t, features = _fit_fixed_theta_map(eeg=eeg, metadata=metadata, random_slope=random_slope)
    oracle_max_abs_diff = float(np.max(np.abs(lmer_t - gls_t)))
    null_t = _fixed_theta_null_maps(
        features=features,
        metadata=metadata,
        n_permutations=n_permutations,
        seed=seed + 900_000,
    )
    return {
        "maxstat": _maxstat_rejects(gls_t, null_t),
        "cluster": _cluster_rejects(gls_t, null_t),
        "tfce": _tfce_rejects(gls_t, null_t),
    }, time.perf_counter() - start, oracle_max_abs_diff


@pytest.mark.slow
def test_real_lmm_slope_h0_guard_passes_before_refit_calibration() -> None:
    _assert_generator_h0_is_null_at_every_feature(random_slope=True, seed_offset=80_000)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("label", "random_slope", "fixed_effect"),
    [
        ("C1", False, 0.0),
        ("C2", True, 0.0),
        ("C3", False, 0.45),
    ],
)
@pytest.mark.parametrize(("n_subjects", "n_items"), [(6, 5), (12, 10), (24, 20), (36, 30)])
def test_real_lmm_parametric_feature_calibration(
    label: str,
    random_slope: bool,
    fixed_effect: float,
    n_subjects: int,
    n_items: int,
) -> None:
    _require_pymer4()
    n_sims = _parametric_settings()
    rates, p_values, elapsed = _estimate_parametric_feature_rates(
        label=label,
        random_slope=random_slope,
        fixed_effect=fixed_effect,
        n_subjects=n_subjects,
        n_items=n_items,
        n_sims=n_sims,
        seed_offset=130_000 + n_subjects * 1_000 + n_items * 10 + (10_000 if random_slope else 0),
    )
    mc_bound = 3.0 * float(np.sqrt(ALPHA * (1.0 - ALPHA) / n_sims))
    for location in range(rates.shape[0]):
        for time_index in range(rates.shape[1]):
            rate = float(rates[location, time_index])
            print(
                f"RESULT REAL_LMM_PARAM {label} size={n_subjects}x{n_items} "
                f"feature={location},{time_index} rate={rate:.3f} "
                f"mc_bound={mc_bound:.3f} n_sims={n_sims} "
                f"mean_p={float(np.mean(p_values[:, location, time_index])):.3f} "
                f"seconds={elapsed:.3f}",
                flush=True,
            )
            if fixed_effect == 0.0:
                assert abs(rate - ALPHA) <= mc_bound
            elif n_sims >= 300:
                assert rate >= ALPHA + 0.10


@pytest.mark.slow
@pytest.mark.parametrize("random_slope", [False, True])
def test_fixed_theta_gls_observed_t_matches_lmer_oracle(random_slope: bool) -> None:
    _require_pymer4()
    eeg, metadata = _simulate_null_or_power(
        seed=150_000 + int(random_slope),
        random_slope=random_slope,
        fixed_effect=0.0,
        n_subjects=12,
        n_items=10,
    )
    lmer_t, gls_t, _ = _fit_fixed_theta_map(eeg=eeg, metadata=metadata, random_slope=random_slope)
    max_abs_diff = float(np.max(np.abs(lmer_t - gls_t)))
    print(
        f"RESULT FIXED_THETA_ORACLE random_slope={random_slope} "
        f"max_abs_diff={max_abs_diff:.6g}",
        flush=True,
    )
    assert np.allclose(gls_t, lmer_t, rtol=1e-9, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("label", "random_slope", "fixed_effect"),
    [
        ("C1", False, 0.0),
        ("C2", True, 0.0),
        ("C3", True, 0.45),
    ],
)
@pytest.mark.parametrize(("n_subjects", "n_items"), [(6, 5), (12, 10), (24, 20), (36, 30)])
def test_fixed_theta_permutation_tfce_calibration(
    label: str,
    random_slope: bool,
    fixed_effect: float,
    n_subjects: int,
    n_items: int,
) -> None:
    _require_pymer4()
    if os.environ.get("LMEEG_RUN_FIXED_THETA_GRID") != "1":
        pytest.skip("Set LMEEG_RUN_FIXED_THETA_GRID=1 for the fixed-theta permutation calibration grid.")
    n_sims, n_permutations = _fixed_theta_settings()
    progress_every = int(os.environ.get("LMEEG_PROGRESS_EVERY", "0"))
    rejections = {"maxstat": [], "cluster": [], "tfce": []}
    elapsed_seconds = []
    oracle_diffs = []
    for sim in range(n_sims):
        sim_rejections, elapsed, oracle_diff = _fixed_theta_rejections(
            seed=160_000 + n_subjects * 1_000 + n_items * 10 + sim + (10_000 if random_slope else 0),
            random_slope=random_slope,
            fixed_effect=fixed_effect,
            n_subjects=n_subjects,
            n_items=n_items,
            n_permutations=n_permutations,
        )
        elapsed_seconds.append(elapsed)
        oracle_diffs.append(oracle_diff)
        for backend, rejected in sim_rejections.items():
            rejections[backend].append(rejected)
        if progress_every and ((sim + 1) % progress_every == 0 or sim + 1 == n_sims):
            summary = " ".join(
                f"{backend}={float(np.mean(values)):.3f}" for backend, values in rejections.items()
            )
            print(
                f"PROGRESS FIXED_THETA {label} size={n_subjects}x{n_items} "
                f"sim={sim + 1}/{n_sims} n_permutations={n_permutations} {summary} "
                f"max_oracle_diff={float(np.max(oracle_diffs)):.3g}",
                flush=True,
            )

    failed_backends = []
    for backend, values in rejections.items():
        rate = float(np.mean(values))
        se = _mc_se(rate, n_sims)
        print(
            f"RESULT FIXED_THETA {label} size={n_subjects}x{n_items} backend={backend} "
            f"rate={rate:.3f} mc_se={se:.3f} n_sims={n_sims} "
            f"n_permutations={n_permutations} mean_seconds_per_sim={float(np.mean(elapsed_seconds)):.3f} "
            f"max_oracle_diff={float(np.max(oracle_diffs)):.6g}",
            flush=True,
        )
        assert 0.0 <= rate <= 1.0
        assert float(np.max(oracle_diffs)) <= 1e-8
        if n_sims >= 300 and n_permutations >= 500:
            if fixed_effect == 0.0:
                if rate > ALPHA + 0.03:
                    failed_backends.append((backend, rate))
            else:
                if rate < ALPHA + 0.10:
                    failed_backends.append((backend, rate))
    assert not failed_backends, f"Fixed-theta calibration failed for {label}: {failed_backends!r}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("label", "random_slope", "fixed_effect"),
    [
        ("C1", False, 0.0),
        ("C2", True, 0.0),
        ("C3", False, 0.45),
    ],
)
def test_real_lmm_refit_null_minimal_slice(label: str, random_slope: bool, fixed_effect: float) -> None:
    _require_pymer4()
    if os.environ.get("LMEEG_RUN_REAL_LMM_REFIT_GRID") != "1":
        pytest.skip("Set LMEEG_RUN_REAL_LMM_REFIT_GRID=1 for the expensive real-LMM permutation slice.")
    n_sims, n_permutations = _real_lmm_settings()
    n_subjects = int(os.environ.get("LMEEG_REAL_LMM_REFIT_N_SUBJECTS", "12"))
    n_items = int(os.environ.get("LMEEG_REAL_LMM_REFIT_N_ITEMS", "10"))
    progress_every = int(os.environ.get("LMEEG_PROGRESS_EVERY", "0"))
    rejections = {"maxstat": [], "cluster": [], "tfce": []}
    elapsed_seconds = []
    for sim in range(n_sims):
        sim_rejections, elapsed = _real_lmm_refit_rejections(
            seed=120_000 + sim,
            random_slope=random_slope,
            fixed_effect=fixed_effect,
            n_permutations=n_permutations,
            n_subjects=n_subjects,
            n_items=n_items,
        )
        elapsed_seconds.append(elapsed)
        for backend, rejected in sim_rejections.items():
            rejections[backend].append(rejected)
        if progress_every and ((sim + 1) % progress_every == 0 or sim + 1 == n_sims):
            summary = " ".join(
                f"{backend}={float(np.mean(values)):.3f}" for backend, values in rejections.items()
            )
            print(
                f"PROGRESS REAL_LMM {label} sim={sim + 1}/{n_sims} "
                f"size={n_subjects}x{n_items} n_permutations={n_permutations} {summary}",
                flush=True,
            )

    for backend, values in rejections.items():
        rate = float(np.mean(values))
        se = _mc_se(rate, n_sims)
        print(
            f"RESULT REAL_LMM {label} backend={backend} rate={rate:.3f} mc_se={se:.3f} "
            f"size={n_subjects}x{n_items} n_sims={n_sims} n_permutations={n_permutations} "
            f"mean_seconds_per_sim={float(np.mean(elapsed_seconds)):.3f}",
            flush=True,
        )
        assert 0.0 <= rate <= 1.0
        if n_sims >= 300 and n_permutations >= 500:
            if fixed_effect == 0.0:
                assert rate <= ALPHA + 0.03
            else:
                assert rate >= ALPHA + 0.10
