#!/usr/bin/env python
"""Manual source-space simulation validation for LmeEEG.

This script is intentionally slower and more explicit than the unit tests. It
checks source-space recovery, null false-positive control, and optional chunked
vs unchunked equivalence for small runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect


Scenario = Literal["recovery", "null"]
ReplicateStatus = Literal["ok", "model_failed", "permutation_failed", "other_failed"]


@dataclass(slots=True)
class SourceSimulationConfig:
    """Configuration for one synthetic source-space dataset."""

    scenario: Scenario = "recovery"
    n_subjects: int = 8
    trials_per_subject: int = 12
    n_sources: int = 32
    n_times: int = 20
    effect_size: float = 0.8
    random_intercept_sd: float = 0.7
    noise_sd: float = 1.0
    spatial_width: float | None = None
    temporal_width: float | None = None


@dataclass(slots=True)
class SourceSimulationData:
    """Synthetic source-space data and ground truth."""

    eeg: np.ndarray
    metadata: pd.DataFrame
    effect_map: np.ndarray
    effect_mask: np.ndarray
    source_names: list[str]


@dataclass(slots=True)
class ReplicateResult:
    """Summary for one simulation replicate."""

    replicate: int
    scenario: str
    seed: int
    n_observations: int
    n_sources: int
    n_times: int
    n_permutations: int
    status: str
    failure_class: str | None
    failure_message: str | None
    model_converged: bool
    convergence_rate: float
    n_lmm_features: int
    n_lmm_failed: int
    n_failed_features: int
    failed_feature_fraction: float
    n_converged_features: int
    converged_feature_fraction: float
    n_boundary_warnings: int
    significant_anywhere: bool | None
    significant_in_effect_mask: bool | None
    global_peak_in_effect_mask: bool | None
    beta_sign_correct_at_peak: bool | None
    metric_success: bool | None
    min_corrected_p: float | None
    min_corrected_p_in_effect_mask: float | None
    peak_statistic: float | None
    peak_location: int | None
    peak_time: int | None
    chunk_equivalence_checked: bool
    chunk_t_close: bool | None
    fit_seconds: float | None
    permutation_seconds: float | None
    total_seconds: float
    seconds_per_permutation: float | None
    peak_memory_mb: float | None


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Return Wilson estimate, lower, upper, and half-width for a binomial rate."""
    if n < 0:
        raise ValueError("`n` must be non-negative.")
    if successes < 0 or successes > n:
        raise ValueError("`successes` must be between 0 and `n`.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("`alpha` must be between 0 and 1.")
    if n == 0:
        return math.nan, math.nan, math.nan, math.inf

    z = _normal_quantile(1.0 - alpha / 2.0)
    phat = successes / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denominator
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return phat, lower, upper, (upper - lower) / 2.0


def _normal_quantile(probability: float) -> float:
    """Approximate the standard-normal quantile without scipy."""
    if not 0.0 < probability < 1.0:
        raise ValueError("`probability` must be between 0 and 1.")

    # Peter John Acklam's inverse-normal approximation.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability <= high:
        q = probability - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def simulate_source_dataset(config: SourceSimulationConfig, seed: int) -> SourceSimulationData:
    """Generate one synthetic source-space random-intercept dataset."""
    if config.n_subjects <= 0:
        raise ValueError("`n_subjects` must be positive.")
    if config.trials_per_subject <= 1:
        raise ValueError("`trials_per_subject` must be greater than 1.")
    if config.n_sources <= 0 or config.n_times <= 0:
        raise ValueError("`n_sources` and `n_times` must be positive.")

    rng = np.random.default_rng(seed)
    n_observations = config.n_subjects * config.trials_per_subject
    subjects = np.repeat([f"sub-{index:03d}" for index in range(config.n_subjects)], config.trials_per_subject)
    condition_by_subject = np.tile(_balanced_conditions(config.trials_per_subject), config.n_subjects)
    metadata = pd.DataFrame({"subject": subjects, "condition": condition_by_subject})

    effect_map = _make_effect_map(config)
    if config.scenario == "null":
        effect_map = np.zeros_like(effect_map)
    effect_mask = _make_effect_mask(_make_effect_map(config))

    subject_offsets = rng.normal(
        loc=0.0,
        scale=config.random_intercept_sd,
        size=(config.n_subjects, config.n_sources, config.n_times),
    ).astype(np.float32)
    noise = rng.normal(
        loc=0.0,
        scale=config.noise_sd,
        size=(n_observations, config.n_sources, config.n_times),
    ).astype(np.float32)
    eeg = noise
    for subject_index in range(config.n_subjects):
        subject_slice = slice(
            subject_index * config.trials_per_subject,
            (subject_index + 1) * config.trials_per_subject,
        )
        eeg[subject_slice, :, :] += subject_offsets[subject_index]

    condition_is_b = (metadata["condition"].to_numpy() == "B").astype(np.float32)
    eeg += condition_is_b[:, None, None] * effect_map.astype(np.float32)[None, :, :]
    source_names = [f"src-{index:05d}" for index in range(config.n_sources)]

    return SourceSimulationData(
        eeg=eeg.astype(np.float32, copy=False),
        metadata=metadata,
        effect_map=effect_map.astype(np.float32, copy=False),
        effect_mask=effect_mask,
        source_names=source_names,
    )


def _run_replicate_strict_feature_convergence_legacy(
    replicate: int,
    scenario: Scenario,
    seed: int,
    simulation_config: SourceSimulationConfig,
    n_permutations: int,
    alpha: float,
    spatial_chunk_size: int | None,
    time_chunk_size: int | None,
    check_chunk_equivalence: bool = False,
) -> ReplicateResult:
    """Legacy strict feature-convergence replicate runner kept for comparison."""
    raise RuntimeError("Use `run_replicate`; feature-level failures are now recorded without failing the replicate.")
    start = time.perf_counter()
    config = SourceSimulationConfig(**{**asdict(simulation_config), "scenario": scenario})
    simulated = simulate_source_dataset(config=config, seed=seed)

    fit_result = fit_lmm_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        formula="y ~ condition + (1|subject)",
        variable_types={"condition": "categorical", "subject": "group"},
        config=FitConfig(
            space="source",
            source_names=simulated.source_names,
            dtype="float32",
            spatial_chunk_size=spatial_chunk_size,
            time_chunk_size=time_chunk_size,
            show_progress=False,
        ),
    )
    model_converged = fit_result.convergence_summary.n_failed == 0
    chunk_t_close = None
    chunk_checked = False

    if check_chunk_equivalence:
        unchunked_fit = fit_lmm_mass_univariate(
            eeg=simulated.eeg,
            metadata=simulated.metadata,
            formula="y ~ condition + (1|subject)",
            variable_types={"condition": "categorical", "subject": "group"},
            config=FitConfig(
                space="source",
                source_names=simulated.source_names,
                dtype="float32",
                show_progress=False,
            ),
        )
        chunk_t_close = bool(
            np.allclose(
                fit_result.ols_t_values["condition[T.B]"],
                unchunked_fit.ols_t_values["condition[T.B]"],
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
            )
        )
        chunk_checked = True

    if not model_converged:
        return ReplicateResult(
            replicate=replicate,
            scenario=scenario,
            seed=seed,
            n_observations=simulated.eeg.shape[0],
            n_sources=simulated.eeg.shape[1],
            n_times=simulated.eeg.shape[2],
            n_permutations=n_permutations,
            model_converged=False,
            convergence_rate=fit_result.convergence_summary.convergence_rate,
            n_lmm_features=fit_result.convergence_summary.n_features,
            n_lmm_failed=fit_result.convergence_summary.n_failed,
            n_boundary_warnings=fit_result.convergence_summary.n_boundary_warnings,
            significant_anywhere=None,
            significant_in_effect_mask=None,
            global_peak_in_effect_mask=None,
            beta_sign_correct_at_peak=None,
            metric_success=None,
            min_corrected_p=None,
            min_corrected_p_in_effect_mask=None,
            peak_statistic=None,
            peak_location=None,
            peak_time=None,
            chunk_equivalence_checked=chunk_checked,
            chunk_t_close=chunk_t_close,
            runtime_seconds=time.perf_counter() - start,
        )

    inference = permute_fixed_effect(
        fit_result=fit_result,
        effect="condition[T.B]",
        correction="maxstat",
        n_permutations=n_permutations,
        seed=seed + 10_000,
        spatial_chunk_size=spatial_chunk_size,
        time_chunk_size=time_chunk_size,
        verbose=False,
    )
    corrected_p = inference.corrected_p_values
    observed_t = inference.observed_statistic
    beta_map = fit_result.ols_betas["condition[T.B]"]
    significant = corrected_p <= alpha
    significant_anywhere = bool(np.any(significant))
    significant_in_mask = bool(np.any(significant & simulated.effect_mask)) if scenario == "recovery" else None

    peak_flat_index = int(np.nanargmax(np.abs(observed_t)))
    peak_location, peak_time = np.unravel_index(peak_flat_index, observed_t.shape)
    peak_in_mask = bool(simulated.effect_mask[peak_location, peak_time]) if scenario == "recovery" else None
    sign_correct = None
    min_p_in_mask = None
    if scenario == "recovery":
        masked_abs_t = np.where(simulated.effect_mask, np.abs(observed_t), -np.inf)
        mask_peak_location, mask_peak_time = np.unravel_index(int(np.argmax(masked_abs_t)), observed_t.shape)
        sign_correct = bool(np.sign(beta_map[mask_peak_location, mask_peak_time]) == np.sign(config.effect_size))
        min_p_in_mask = float(np.min(corrected_p[simulated.effect_mask]))
        metric_success = bool(significant_in_mask and peak_in_mask and sign_correct)
    else:
        metric_success = significant_anywhere

    return ReplicateResult(
        replicate=replicate,
        scenario=scenario,
        seed=seed,
        n_observations=simulated.eeg.shape[0],
        n_sources=simulated.eeg.shape[1],
        n_times=simulated.eeg.shape[2],
        n_permutations=n_permutations,
        model_converged=True,
        convergence_rate=fit_result.convergence_summary.convergence_rate,
        n_lmm_features=fit_result.convergence_summary.n_features,
        n_lmm_failed=fit_result.convergence_summary.n_failed,
        n_boundary_warnings=fit_result.convergence_summary.n_boundary_warnings,
        significant_anywhere=significant_anywhere,
        significant_in_effect_mask=significant_in_mask,
        global_peak_in_effect_mask=peak_in_mask,
        beta_sign_correct_at_peak=sign_correct,
        metric_success=metric_success,
        min_corrected_p=float(np.min(corrected_p)),
        min_corrected_p_in_effect_mask=min_p_in_mask,
        peak_statistic=float(observed_t[peak_location, peak_time]),
        peak_location=int(peak_location),
        peak_time=int(peak_time),
        chunk_equivalence_checked=chunk_checked,
        chunk_t_close=chunk_t_close,
        runtime_seconds=time.perf_counter() - start,
    )


def run_replicate(
    replicate: int,
    scenario: Scenario,
    seed: int,
    simulation_config: SourceSimulationConfig,
    n_permutations: int,
    alpha: float,
    spatial_chunk_size: int | None,
    time_chunk_size: int | None,
    check_chunk_equivalence: bool = False,
) -> ReplicateResult:
    """Run one source-space fit and max-stat inference replicate."""
    total_start = time.perf_counter()
    config = SourceSimulationConfig(**{**asdict(simulation_config), "scenario": scenario})
    try:
        simulated = simulate_source_dataset(config=config, seed=seed)
    except Exception as error:
        return _failed_replicate_result(
            replicate=replicate,
            scenario=scenario,
            seed=seed,
            config=config,
            n_observations=0,
            n_permutations=n_permutations,
            status="other_failed",
            error=error,
            total_start=total_start,
        )

    fit_start = time.perf_counter()
    try:
        fit_result = fit_lmm_mass_univariate(
            eeg=simulated.eeg,
            metadata=simulated.metadata,
            formula="y ~ condition + (1|subject)",
            variable_types={"condition": "categorical", "subject": "group"},
            config=FitConfig(
                space="source",
                source_names=simulated.source_names,
                dtype="float32",
                spatial_chunk_size=spatial_chunk_size,
                time_chunk_size=time_chunk_size,
                show_progress=False,
            ),
        )
    except Exception as error:
        return _failed_replicate_result(
            replicate=replicate,
            scenario=scenario,
            seed=seed,
            config=config,
            n_observations=simulated.eeg.shape[0],
            n_permutations=n_permutations,
            status="model_failed",
            error=error,
            total_start=total_start,
            fit_seconds=time.perf_counter() - fit_start,
        )
    fit_seconds = time.perf_counter() - fit_start
    feature_counts = _feature_convergence_counts(fit_result)
    chunk_t_close = None
    chunk_checked = False

    if check_chunk_equivalence:
        try:
            unchunked_fit = fit_lmm_mass_univariate(
                eeg=simulated.eeg,
                metadata=simulated.metadata,
                formula="y ~ condition + (1|subject)",
                variable_types={"condition": "categorical", "subject": "group"},
                config=FitConfig(
                    space="source",
                    source_names=simulated.source_names,
                    dtype="float32",
                    show_progress=False,
                ),
            )
            chunk_t_close = bool(
                np.allclose(
                    fit_result.ols_t_values["condition[T.B]"],
                    unchunked_fit.ols_t_values["condition[T.B]"],
                    rtol=1e-5,
                    atol=1e-5,
                    equal_nan=True,
                )
            )
        except Exception:
            chunk_t_close = False
        chunk_checked = True

    permutation_start = time.perf_counter()
    try:
        inference = permute_fixed_effect(
            fit_result=fit_result,
            effect="condition[T.B]",
            correction="maxstat",
            n_permutations=n_permutations,
            seed=seed + 10_000,
            spatial_chunk_size=spatial_chunk_size,
            time_chunk_size=time_chunk_size,
            verbose=False,
        )
    except Exception as error:
        return _failed_replicate_result(
            replicate=replicate,
            scenario=scenario,
            seed=seed,
            config=config,
            n_observations=simulated.eeg.shape[0],
            n_permutations=n_permutations,
            status="permutation_failed",
            error=error,
            total_start=total_start,
            fit_seconds=fit_seconds,
            feature_counts=feature_counts,
            chunk_checked=chunk_checked,
            chunk_t_close=chunk_t_close,
        )
    permutation_seconds = time.perf_counter() - permutation_start
    corrected_p = inference.corrected_p_values
    observed_t = inference.observed_statistic
    beta_map = fit_result.ols_betas["condition[T.B]"]
    significant = corrected_p <= alpha
    significant_anywhere = bool(np.any(significant))
    significant_in_mask = bool(np.any(significant & simulated.effect_mask)) if scenario == "recovery" else None

    peak_flat_index = int(np.nanargmax(np.abs(observed_t)))
    peak_location, peak_time = np.unravel_index(peak_flat_index, observed_t.shape)
    peak_in_mask = bool(simulated.effect_mask[peak_location, peak_time]) if scenario == "recovery" else None
    sign_correct = None
    min_p_in_mask = None
    if scenario == "recovery":
        masked_abs_t = np.where(simulated.effect_mask, np.abs(observed_t), -np.inf)
        mask_peak_location, mask_peak_time = np.unravel_index(int(np.argmax(masked_abs_t)), observed_t.shape)
        sign_correct = bool(np.sign(beta_map[mask_peak_location, mask_peak_time]) == np.sign(config.effect_size))
        min_p_in_mask = float(np.min(corrected_p[simulated.effect_mask]))
        metric_success = bool(significant_in_mask and peak_in_mask and sign_correct)
    else:
        metric_success = significant_anywhere

    return ReplicateResult(
        replicate=replicate,
        scenario=scenario,
        seed=seed,
        n_observations=simulated.eeg.shape[0],
        n_sources=simulated.eeg.shape[1],
        n_times=simulated.eeg.shape[2],
        n_permutations=n_permutations,
        status="ok",
        failure_class=None,
        failure_message=None,
        model_converged=True,
        convergence_rate=feature_counts["converged_feature_fraction"],
        n_lmm_features=feature_counts["n_lmm_features"],
        n_lmm_failed=feature_counts["n_failed_features"],
        n_failed_features=feature_counts["n_failed_features"],
        failed_feature_fraction=feature_counts["failed_feature_fraction"],
        n_converged_features=feature_counts["n_converged_features"],
        converged_feature_fraction=feature_counts["converged_feature_fraction"],
        n_boundary_warnings=feature_counts["n_boundary_warnings"],
        significant_anywhere=significant_anywhere,
        significant_in_effect_mask=significant_in_mask,
        global_peak_in_effect_mask=peak_in_mask,
        beta_sign_correct_at_peak=sign_correct,
        metric_success=metric_success,
        min_corrected_p=float(np.min(corrected_p)),
        min_corrected_p_in_effect_mask=min_p_in_mask,
        peak_statistic=float(observed_t[peak_location, peak_time]),
        peak_location=int(peak_location),
        peak_time=int(peak_time),
        chunk_equivalence_checked=chunk_checked,
        chunk_t_close=chunk_t_close,
        fit_seconds=fit_seconds,
        permutation_seconds=permutation_seconds,
        total_seconds=time.perf_counter() - total_start,
        seconds_per_permutation=permutation_seconds / n_permutations,
        peak_memory_mb=_get_peak_memory_mb(),
    )


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    """Run sequential Monte Carlo validation and write CSV/JSON outputs."""
    if args.batch_size <= 0:
        raise ValueError("`--batch-size` must be positive.")
    if args.min_reps < 0:
        raise ValueError("`--min-reps` must be non-negative.")
    if args.max_reps <= 0:
        raise ValueError("`--max-reps` must be positive.")
    if args.min_reps > args.max_reps:
        raise ValueError("`--min-reps` must be less than or equal to `--max-reps`.")
    if args.n_permutations <= 0:
        raise ValueError("`--n-permutations` must be positive.")
    if args.tolerance <= 0.0:
        raise ValueError("`--tolerance` must be positive.")
    if not 0.0 <= args.max_model_failure_rate <= 1.0:
        raise ValueError("`--max-model-failure-rate` must be between 0 and 1.")

    start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"source_lmeeeg_{args.scenario}_replicates.csv"
    json_path = out_dir / f"source_lmeeeg_{args.scenario}_summary.json"

    simulation_config = SourceSimulationConfig(
        scenario=args.scenario,
        n_subjects=args.n_subjects,
        trials_per_subject=args.trials_per_subject,
        n_sources=args.n_sources,
        n_times=args.n_times,
        effect_size=args.effect_size,
        random_intercept_sd=args.random_intercept_sd,
        noise_sd=args.noise_sd,
        spatial_width=args.spatial_width,
        temporal_width=args.temporal_width,
    )

    rows: list[ReplicateResult] = []
    stopping_reason = "max_reps"
    replicate_index = 0
    progress = None if args.no_progress else _build_progress()
    with progress or _null_progress() as active_progress:
        task_id = None
        if progress is not None:
            task_id = active_progress.add_task(
                f"{args.scenario} validation",
                total=args.max_reps,
                status="starting",
            )
        while replicate_index < args.max_reps:
            batch_stop = min(replicate_index + args.batch_size, args.max_reps)
            while replicate_index < batch_stop:
                if progress is not None and task_id is not None:
                    active_progress.update(
                        task_id,
                        status=f"rep {replicate_index + 1}/{args.max_reps}",
                    )
                rows.append(
                    run_replicate(
                        replicate=replicate_index,
                        scenario=args.scenario,
                        seed=args.seed + replicate_index,
                        simulation_config=simulation_config,
                        n_permutations=args.n_permutations,
                        alpha=args.alpha,
                        spatial_chunk_size=args.spatial_chunk_size,
                        time_chunk_size=args.time_chunk_size,
                        check_chunk_equivalence=args.check_chunk_equivalence and replicate_index == 0,
                    )
                )
                replicate_index += 1
                if progress is not None and task_id is not None:
                    active_progress.advance(task_id)

            _write_rows(csv_path, rows)
            valid_rows = _valid_rows(rows)
            successes = sum(bool(row.metric_success) for row in valid_rows)
            estimate, ci_low, ci_high, half_width = wilson_interval(successes, len(valid_rows), args.alpha)
            model_failure_rate = _model_failure_rate(rows)
            if progress is not None and task_id is not None:
                active_progress.update(
                    task_id,
                    status=_format_progress_status(
                        successes=successes,
                        n_valid=len(valid_rows),
                        n_failed=sum(row.status != "ok" for row in rows),
                        estimate=estimate,
                        half_width=half_width,
                    ),
                )
            if len(valid_rows) >= args.min_reps and half_width <= args.tolerance:
                if model_failure_rate <= args.max_model_failure_rate:
                    stopping_reason = "tolerance"
                    break
                stopping_reason = "high_model_failure_rate"

    _write_rows(csv_path, rows)

    valid_rows = _valid_rows(rows)
    successes = sum(bool(row.metric_success) for row in valid_rows)
    estimate, ci_low, ci_high, half_width = wilson_interval(successes, len(valid_rows), args.alpha)
    model_failed_rows = [row for row in rows if row.status == "model_failed"]
    failed_rows = [row for row in rows if row.status != "ok"]
    model_failure_rate = _model_failure_rate(rows)
    tolerance_reached = len(valid_rows) >= args.min_reps and half_width <= args.tolerance
    stopping_guard_passed = model_failure_rate <= args.max_model_failure_rate
    if stopping_reason == "max_reps" and tolerance_reached and not stopping_guard_passed:
        stopping_reason = "high_model_failure_rate"
    summary = {
        "scenario": args.scenario,
        "metric": "detection_rate" if args.scenario == "recovery" else "false_positive_rate",
        "estimate": _finite_or_none(estimate),
        "ci_low": _finite_or_none(ci_low),
        "ci_high": _finite_or_none(ci_high),
        "ci_half_width": _finite_or_none(half_width),
        "alpha": args.alpha,
        "n_reps_total": len(rows),
        "n_reps_valid": len(valid_rows),
        "n_reps_failed": len(failed_rows),
        "n_reps_failed_model": len(model_failed_rows),
        "n_reps_failed_permutation": sum(row.status == "permutation_failed" for row in rows),
        "n_reps_failed_other": sum(row.status == "other_failed" for row in rows),
        "model_failure_rate": model_failure_rate,
        "model_failure_rate_ci_low": _finite_or_none(wilson_interval(len(model_failed_rows), len(rows), args.alpha)[1]),
        "model_failure_rate_ci_high": _finite_or_none(wilson_interval(len(model_failed_rows), len(rows), args.alpha)[2]),
        "max_model_failure_rate": args.max_model_failure_rate,
        "tolerance_reached": tolerance_reached,
        "stopping_guard_passed": stopping_guard_passed,
        "stopping_reason": stopping_reason,
        "runtime_seconds": time.perf_counter() - start,
        "timing": _timing_summary(valid_rows),
        "simulation_config": asdict(simulation_config),
        "n_permutations": args.n_permutations,
        "spatial_chunk_size": args.spatial_chunk_size,
        "time_chunk_size": args.time_chunk_size,
        "check_chunk_equivalence": args.check_chunk_equivalence,
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["recovery", "null"], default="recovery")
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--trials-per-subject", type=int, default=12)
    parser.add_argument("--n-sources", type=int, default=32)
    parser.add_argument("--n-times", type=int, default=20)
    parser.add_argument("--effect-size", type=float, default=0.8)
    parser.add_argument("--random-intercept-sd", type=float, default=0.7)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--spatial-width", type=float, default=None)
    parser.add_argument("--temporal-width", type=float, default=None)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--min-reps", type=int, default=20)
    parser.add_argument("--max-reps", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-model-failure-rate", type=float, default=0.05)
    parser.add_argument("--spatial-chunk-size", type=int, default=None)
    parser.add_argument("--time-chunk-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", default="outputs/source_lmeeeg_validation")
    parser.add_argument("--check-chunk-equivalence", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the replicate progress bar.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    summary = run_validation(args)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


def _balanced_conditions(trials_per_subject: int) -> np.ndarray:
    conditions = np.array(["A", "B"] * math.ceil(trials_per_subject / 2), dtype=object)
    return conditions[:trials_per_subject]


def _make_effect_map(config: SourceSimulationConfig) -> np.ndarray:
    spatial_width = config.spatial_width or max(config.n_sources / 10.0, 1.0)
    temporal_width = config.temporal_width or max(config.n_times / 10.0, 1.0)
    source_axis = np.arange(config.n_sources, dtype=float)
    time_axis = np.arange(config.n_times, dtype=float)
    source_center = config.n_sources // 2
    time_center = config.n_times // 2
    spatial = np.exp(-0.5 * ((source_axis - source_center) / spatial_width) ** 2)
    temporal = np.exp(-0.5 * ((time_axis - time_center) / temporal_width) ** 2)
    effect = config.effect_size * spatial[:, None] * temporal[None, :]
    return effect.astype(np.float32)


def _make_effect_mask(effect_map: np.ndarray) -> np.ndarray:
    if not np.any(effect_map):
        return np.zeros(effect_map.shape, dtype=bool)
    return np.abs(effect_map) >= 0.5 * float(np.max(np.abs(effect_map)))


def _feature_convergence_counts(fit_result) -> dict[str, int | float]:
    summary = fit_result.convergence_summary
    n_features = int(summary.n_features)
    n_failed = int(summary.n_failed)
    n_converged = int(summary.n_converged)
    return {
        "n_lmm_features": n_features,
        "n_failed_features": n_failed,
        "failed_feature_fraction": n_failed / n_features if n_features else math.nan,
        "n_converged_features": n_converged,
        "converged_feature_fraction": n_converged / n_features if n_features else math.nan,
        "n_boundary_warnings": int(summary.n_boundary_warnings),
    }


def _failed_replicate_result(
    replicate: int,
    scenario: str,
    seed: int,
    config: SourceSimulationConfig,
    n_observations: int,
    n_permutations: int,
    status: ReplicateStatus,
    error: Exception,
    total_start: float,
    fit_seconds: float | None = None,
    feature_counts: dict[str, int | float] | None = None,
    chunk_checked: bool = False,
    chunk_t_close: bool | None = None,
) -> ReplicateResult:
    counts = feature_counts or {
        "n_lmm_features": 0,
        "n_failed_features": 0,
        "failed_feature_fraction": math.nan,
        "n_converged_features": 0,
        "converged_feature_fraction": math.nan,
        "n_boundary_warnings": 0,
    }
    return ReplicateResult(
        replicate=replicate,
        scenario=scenario,
        seed=seed,
        n_observations=n_observations,
        n_sources=config.n_sources,
        n_times=config.n_times,
        n_permutations=n_permutations,
        status=status,
        failure_class=error.__class__.__name__,
        failure_message=str(error),
        model_converged=False,
        convergence_rate=float(counts["converged_feature_fraction"]),
        n_lmm_features=int(counts["n_lmm_features"]),
        n_lmm_failed=int(counts["n_failed_features"]),
        n_failed_features=int(counts["n_failed_features"]),
        failed_feature_fraction=float(counts["failed_feature_fraction"]),
        n_converged_features=int(counts["n_converged_features"]),
        converged_feature_fraction=float(counts["converged_feature_fraction"]),
        n_boundary_warnings=int(counts["n_boundary_warnings"]),
        significant_anywhere=None,
        significant_in_effect_mask=None,
        global_peak_in_effect_mask=None,
        beta_sign_correct_at_peak=None,
        metric_success=None,
        min_corrected_p=None,
        min_corrected_p_in_effect_mask=None,
        peak_statistic=None,
        peak_location=None,
        peak_time=None,
        chunk_equivalence_checked=chunk_checked,
        chunk_t_close=chunk_t_close,
        fit_seconds=fit_seconds,
        permutation_seconds=None,
        total_seconds=time.perf_counter() - total_start,
        seconds_per_permutation=None,
        peak_memory_mb=_get_peak_memory_mb(),
    )


def _valid_rows(rows: list[ReplicateResult]) -> list[ReplicateResult]:
    return [row for row in rows if row.status == "ok" and row.metric_success is not None]


def _model_failure_rate(rows: list[ReplicateResult]) -> float:
    return sum(row.status == "model_failed" for row in rows) / len(rows) if rows else math.nan


def _timing_summary(rows: list[ReplicateResult]) -> dict[str, float | None]:
    return {
        "fit_seconds_mean": _mean_or_none(row.fit_seconds for row in rows),
        "fit_seconds_median": _median_or_none(row.fit_seconds for row in rows),
        "permutation_seconds_mean": _mean_or_none(row.permutation_seconds for row in rows),
        "permutation_seconds_median": _median_or_none(row.permutation_seconds for row in rows),
        "total_seconds_mean": _mean_or_none(row.total_seconds for row in rows),
        "total_seconds_median": _median_or_none(row.total_seconds for row in rows),
        "seconds_per_permutation_mean": _mean_or_none(row.seconds_per_permutation for row in rows),
        "seconds_per_permutation_median": _median_or_none(row.seconds_per_permutation for row in rows),
        "peak_memory_mb_max": _max_or_none(row.peak_memory_mb for row in rows),
    }


def _mean_or_none(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _median_or_none(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _max_or_none(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else None


def _get_peak_memory_mb() -> float | None:
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        peak = _get_psutil_memory_mb()
        return peak
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _get_psutil_memory_mb() -> float | None:
    try:
        import os
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _write_rows(path: Path, rows: list[ReplicateResult]) -> None:
    fieldnames = list(ReplicateResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
    )


class _null_progress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _format_progress_status(
    successes: int,
    n_valid: int,
    n_failed: int,
    estimate: float,
    half_width: float,
) -> str:
    estimate_text = "NA" if not math.isfinite(estimate) else f"{estimate:.3f}"
    half_width_text = "NA" if not math.isfinite(half_width) else f"{half_width:.3f}"
    return f"valid={n_valid} failed={n_failed} successes={successes} est={estimate_text} CIhw={half_width_text}"


if __name__ == "__main__":
    raise SystemExit(main())
