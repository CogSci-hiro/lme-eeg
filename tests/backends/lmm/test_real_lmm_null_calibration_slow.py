import os
import time

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
