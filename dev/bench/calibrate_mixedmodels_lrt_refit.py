"""Calibrate MixedModels.jl LRT-refit permutation maps at dev scale.

This Release 2 Route A artifact is not production code. It fits full and
fixed-effect-reduced MixedModels.jl models by ML for every observed and
within-subject-permuted feature, forms an unsigned likelihood-ratio statistic
for the condition term, and summarizes one-sided maxstat/cluster/TFCE
correction over the resulting LR maps.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dev.bench.benchmark_mixedmodels_julia import (  # noqa: E402
    JULIA_EXE,
    JULIA_PROJECT,
    _configure_julia,
    _configure_pymer4,
    _metadata_columns,
)
from lmeeeg.backends.correction._regression import (  # noqa: E402
    cluster_outputs_to_masks,
    configure_mne_runtime,
)
from tests.backends.lmm.test_pymer4_calibration_slow import (  # noqa: E402
    ALPHA,
    _assert_generator_h0_is_null_at_every_feature,
    _mc_se,
    _simulate_null_or_power,
)
from tests.backends.lmm.test_real_lmm_null_calibration_slow import (  # noqa: E402
    _permute_condition_within_subject,
)


SIZES = ((6, 5), (12, 10), (24, 20), (36, 30))
SCENARIOS = {
    "C1": {"random_slope": False, "fixed_effect": 0.0},
    "C2": {"random_slope": True, "fixed_effect": 0.0},
    "C3": {"random_slope": True, "fixed_effect": 0.45},
}
BACKENDS = ("maxstat", "cluster", "tfce")

_JL: Any | None = None


@dataclass(frozen=True)
class SimResult:
    scenario: str
    n_subjects: int
    n_items: int
    sim: int
    seed: int
    n_permutations: int
    maxstat: bool
    cluster: bool
    tfce: bool
    observed_singular_fits: int
    observed_total_fits: int
    permuted_singular_fits: int
    permuted_total_fits: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CellSummary:
    scenario: str
    n_subjects: int
    n_items: int
    backend: str
    rate: float
    mc_se: float
    n_sims: int
    n_permutations: int
    observed_singular_fraction: float
    permuted_singular_fraction: float
    mean_seconds_per_sim: float
    total_seconds: float
    complete: bool


def _load_julia_lrt() -> Any:
    global _JL
    if _JL is not None:
        return _JL
    _configure_julia()
    from juliacall import Main as jl

    jl.seval("using MixedModels, DataFrames, StatsModels")
    jl.seval(
        r"""
        function lmeeeg_lrt_dataframe(y, cond, subject, item)
            return DataFrame(
                y = Float64.(pyconvert(Vector, y)),
                cond = String.(pyconvert(Vector, cond)),
                subject = String.(pyconvert(Vector, subject)),
                item = String.(pyconvert(Vector, item)),
            )
        end

        function lmeeeg_lrt_full_formula(random_slope)
            return random_slope ?
                @formula(y ~ 1 + cond + (1 + cond | subject) + (1 | item)) :
                @formula(y ~ 1 + cond + (1 | subject) + (1 | item))
        end

        function lmeeeg_lrt_reduced_formula(random_slope)
            return random_slope ?
                @formula(y ~ 1 + (1 + cond | subject) + (1 | item)) :
                @formula(y ~ 1 + (1 | subject) + (1 | item))
        end

        function lmeeeg_condition_lrt_and_singular(y, cond, subject, item, random_slope)
            df = lmeeeg_lrt_dataframe(y, cond, subject, item)
            full = fit(MixedModel, lmeeeg_lrt_full_formula(random_slope), df; REML=false)
            reduced = fit(MixedModel, lmeeeg_lrt_reduced_formula(random_slope), df; REML=false)
            statistic = 2.0 * (loglikelihood(full) - loglikelihood(reduced))
            return (statistic, issingular(full), issingular(reduced))
        end

        function lmeeeg_lrt_contract()
            return (
                full_random_slope = string(lmeeeg_lrt_full_formula(true)),
                reduced_random_slope = string(lmeeeg_lrt_reduced_formula(true)),
                full_crossed_intercept = string(lmeeeg_lrt_full_formula(false)),
                reduced_crossed_intercept = string(lmeeeg_lrt_reduced_formula(false)),
                reml = false,
            )
        end
        """
    )
    _JL = jl
    return jl


def _worker_init() -> None:
    os.environ.setdefault("PYTHON_JULIACALL_EXE", str(JULIA_EXE))
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(JULIA_PROJECT))
    _load_julia_lrt()


def _fit_lrt_map(eeg: np.ndarray, metadata, random_slope: bool) -> tuple[np.ndarray, int, int]:
    jl = _load_julia_lrt()
    cond, subject, item = _metadata_columns(metadata)
    n_locations, n_times = eeg.shape[1], eeg.shape[2]
    lr_map = np.empty((n_locations, n_times), dtype=float)
    singular_count = 0
    total_count = 0
    for location in range(n_locations):
        for time_index in range(n_times):
            statistic, full_singular, reduced_singular = jl.lmeeeg_condition_lrt_and_singular(
                eeg[:, location, time_index],
                cond,
                subject,
                item,
                random_slope,
            )
            statistic = float(statistic)
            if statistic < -1e-6:
                raise RuntimeError(f"Negative LR statistic {statistic} at feature {location},{time_index}.")
            lr_map[location, time_index] = max(0.0, statistic)
            singular_count += int(bool(full_singular)) + int(bool(reduced_singular))
            total_count += 2
    return lr_map, singular_count, total_count


def _maxstat_lr_rejects(observed_lr: np.ndarray, null_lr: np.ndarray) -> bool:
    _validate_lr_map(observed_lr, "observed")
    _validate_lr_map(null_lr, "null")
    null_distribution = np.max(null_lr, axis=(1, 2))
    corrected = (
        1
        + np.sum(null_distribution[:, None, None] >= observed_lr[None, :, :], axis=0)
    ) / (null_lr.shape[0] + 1)
    return bool(np.nanmin(corrected) <= ALPHA)


def _cluster_lr_rejects(observed_lr: np.ndarray, null_lr: np.ndarray) -> bool:
    _validate_lr_map(observed_lr, "observed")
    _validate_lr_map(null_lr, "null")
    configure_mne_runtime()
    from mne.stats.cluster_level import _find_clusters

    threshold = 4.0
    sample_shape = (observed_lr.shape[1], observed_lr.shape[0])
    raw_clusters, cluster_stats = _find_clusters(observed_lr.T, threshold=threshold, tail=1)
    cluster_masks = cluster_outputs_to_masks(raw_clusters, sample_shape)
    null_distribution = np.zeros(null_lr.shape[0], dtype=float)
    for permutation_index, permuted_lr in enumerate(null_lr):
        _, permuted_cluster_stats = _find_clusters(permuted_lr.T, threshold=threshold, tail=1)
        null_distribution[permutation_index] = (
            float(np.max(permuted_cluster_stats)) if len(permuted_cluster_stats) else 0.0
        )
    if not cluster_masks:
        return False
    cluster_p_values = np.asarray(
        [
            (1 + np.sum(null_distribution >= cluster_stat)) / (null_lr.shape[0] + 1)
            for cluster_stat in cluster_stats
        ],
        dtype=float,
    )
    corrected = np.ones_like(observed_lr, dtype=float)
    for cluster_mask, cluster_p_value in zip(cluster_masks, cluster_p_values):
        corrected[cluster_mask.T] = np.minimum(corrected[cluster_mask.T], cluster_p_value)
    return bool(np.nanmin(corrected) <= ALPHA)


def _tfce_lr_rejects(observed_lr: np.ndarray, null_lr: np.ndarray) -> bool:
    _validate_lr_map(observed_lr, "observed")
    _validate_lr_map(null_lr, "null")
    configure_mne_runtime()
    from mne.stats.cluster_level import _find_clusters

    threshold = {"start": 0.0, "step": 0.2, "h_power": 2.0, "e_power": 0.5}
    _, observed_tfce = _find_clusters(observed_lr.T, threshold=threshold, tail=1)
    observed_tfce = np.asarray(observed_tfce, dtype=float).reshape(observed_lr.shape[1], observed_lr.shape[0]).T
    null_distribution = np.zeros(null_lr.shape[0], dtype=float)
    for permutation_index, permuted_lr in enumerate(null_lr):
        _, permuted_tfce = _find_clusters(permuted_lr.T, threshold=threshold, tail=1)
        null_distribution[permutation_index] = float(np.max(permuted_tfce))
    corrected = (
        1
        + np.sum(null_distribution[:, None, None] >= observed_tfce[None, :, :], axis=0)
    ) / (null_lr.shape[0] + 1)
    return bool(np.nanmin(corrected) <= ALPHA)


def _validate_lr_map(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} LR map contains non-finite values.")
    if float(np.min(values)) < -1e-10:
        raise RuntimeError(f"{label} LR map contains negative values.")


def _run_one_sim(
    scenario: str,
    n_subjects: int,
    n_items: int,
    sim: int,
    n_permutations: int,
    seed_offset: int,
) -> SimResult:
    settings = SCENARIOS[scenario]
    random_slope = bool(settings["random_slope"])
    fixed_effect = float(settings["fixed_effect"])
    seed = seed_offset + n_subjects * 1_000 + n_items * 10 + sim + (10_000 if random_slope else 0)
    start = time.perf_counter()
    eeg, metadata = _simulate_null_or_power(
        seed=seed,
        random_slope=random_slope,
        fixed_effect=fixed_effect,
        n_subjects=n_subjects,
        n_items=n_items,
    )
    observed_lr, observed_singular, observed_total = _fit_lrt_map(
        eeg=eeg,
        metadata=metadata,
        random_slope=random_slope,
    )
    null_lr = np.empty((n_permutations,) + observed_lr.shape, dtype=float)
    permuted_singular = 0
    permuted_total = 0
    rng = np.random.default_rng(seed + 700_000)
    for permutation_index in range(n_permutations):
        permuted_metadata = _permute_condition_within_subject(metadata=metadata, rng=rng)
        permuted_lr, singular_count, total_count = _fit_lrt_map(
            eeg=eeg,
            metadata=permuted_metadata,
            random_slope=random_slope,
        )
        null_lr[permutation_index] = permuted_lr
        permuted_singular += singular_count
        permuted_total += total_count

    return SimResult(
        scenario=scenario,
        n_subjects=n_subjects,
        n_items=n_items,
        sim=sim,
        seed=seed,
        n_permutations=n_permutations,
        maxstat=_maxstat_lr_rejects(observed_lr, null_lr),
        cluster=_cluster_lr_rejects(observed_lr, null_lr),
        tfce=_tfce_lr_rejects(observed_lr, null_lr),
        observed_singular_fits=observed_singular,
        observed_total_fits=observed_total,
        permuted_singular_fits=permuted_singular,
        permuted_total_fits=permuted_total,
        elapsed_seconds=time.perf_counter() - start,
    )


def _completed_keys(results_path: Path) -> set[tuple[str, int, int, int]]:
    keys: set[tuple[str, int, int, int]] = set()
    if not results_path.exists():
        return keys
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            keys.add(
                (
                    str(payload["scenario"]),
                    int(payload["n_subjects"]),
                    int(payload["n_items"]),
                    int(payload["sim"]),
                )
            )
    return keys


def _iter_tasks(
    scenarios: Iterable[str],
    sizes: Iterable[tuple[int, int]],
    n_sims: int,
    completed: set[tuple[str, int, int, int]],
) -> list[tuple[str, int, int, int]]:
    tasks = []
    for scenario in scenarios:
        for n_subjects, n_items in sizes:
            for sim in range(n_sims):
                key = (scenario, n_subjects, n_items, sim)
                if key not in completed:
                    tasks.append(key)
    return tasks


def _summarize(results: list[SimResult], n_permutations: int, expected_n_sims: int) -> list[CellSummary]:
    summaries = []
    by_cell: dict[tuple[str, int, int], list[SimResult]] = {}
    for result in results:
        by_cell.setdefault((result.scenario, result.n_subjects, result.n_items), []).append(result)
    for (scenario, n_subjects, n_items), cell_results in sorted(by_cell.items()):
        observed_singular_fraction = sum(r.observed_singular_fits for r in cell_results) / sum(
            r.observed_total_fits for r in cell_results
        )
        permuted_singular_fraction = sum(r.permuted_singular_fits for r in cell_results) / sum(
            r.permuted_total_fits for r in cell_results
        )
        total_seconds = float(sum(r.elapsed_seconds for r in cell_results))
        mean_seconds = float(np.mean([r.elapsed_seconds for r in cell_results]))
        for backend in BACKENDS:
            values = [bool(getattr(r, backend)) for r in cell_results]
            rate = float(np.mean(values))
            summaries.append(
                CellSummary(
                    scenario=scenario,
                    n_subjects=n_subjects,
                    n_items=n_items,
                    backend=backend,
                    rate=rate,
                    mc_se=_mc_se(rate, len(values)),
                    n_sims=len(values),
                    n_permutations=n_permutations,
                    observed_singular_fraction=float(observed_singular_fraction),
                    permuted_singular_fraction=float(permuted_singular_fraction),
                    mean_seconds_per_sim=mean_seconds,
                    total_seconds=total_seconds,
                    complete=len(values) == expected_n_sims,
                )
            )
    return summaries


def _write_summary(path: Path, summaries: list[CellSummary]) -> None:
    path.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_results(results_path: Path) -> list[SimResult]:
    results = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                results.append(SimResult(**json.loads(line)))
    return results


def _run_guard() -> None:
    _configure_pymer4()
    _assert_generator_h0_is_null_at_every_feature(random_slope=True, seed_offset=80_000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dev/bench/mixedmodels_lrt_refit_calibration"))
    parser.add_argument("--n-sims", type=int, default=int(os.environ.get("LMEEG_JULIA_LRT_N_SIMS", "300")))
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=int(os.environ.get("LMEEG_JULIA_LRT_N_PERMUTATIONS", "500")),
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LMEEG_JULIA_LRT_WORKERS", "1")))
    parser.add_argument("--seed-offset", type=int, default=310_000)
    parser.add_argument("--progress-every", type=int, default=int(os.environ.get("LMEEG_PROGRESS_EVERY", "10")))
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--sizes", nargs="+", default=[f"{subjects}x{items}" for subjects, items in SIZES])
    parser.add_argument("--skip-guard", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHON_JULIACALL_EXE", str(JULIA_EXE))
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(JULIA_PROJECT))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sim_results.jsonl"
    summary_path = args.output_dir / "summary.json"

    sizes = []
    for size in args.sizes:
        subject_text, item_text = size.lower().split("x", maxsplit=1)
        sizes.append((int(subject_text), int(item_text)))

    if args.print_contract:
        contract = _load_julia_lrt().lmeeeg_lrt_contract()
        print(f"CONTRACT {contract}", flush=True)

    if not args.skip_guard:
        guard_start = time.perf_counter()
        _run_guard()
        print(f"GUARD random_slope_h0 passed seconds={time.perf_counter() - guard_start:.3f}", flush=True)

    completed = _completed_keys(results_path)
    tasks = _iter_tasks(args.scenarios, sizes, args.n_sims, completed)
    print(
        f"START statistic=lrt scenarios={args.scenarios} sizes={sizes} n_sims={args.n_sims} "
        f"n_permutations={args.n_permutations} workers={args.workers} remaining_tasks={len(tasks)}",
        flush=True,
    )

    started = time.perf_counter()
    completed_now = 0
    with results_path.open("a", encoding="utf-8") as handle:
        if args.workers == 1:
            _worker_init()
            for task in tasks:
                result = _run_one_sim(*task, n_permutations=args.n_permutations, seed_offset=args.seed_offset)
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                handle.flush()
                completed_now += 1
                _print_progress(args.progress_every, completed_now, len(tasks), result)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
                futures = {
                    pool.submit(
                        _run_one_sim,
                        *task,
                        n_permutations=args.n_permutations,
                        seed_offset=args.seed_offset,
                    ): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                    handle.flush()
                    completed_now += 1
                    _print_progress(args.progress_every, completed_now, len(tasks), result)

    results = _load_results(results_path)
    summaries = _summarize(results, n_permutations=args.n_permutations, expected_n_sims=args.n_sims)
    _write_summary(summary_path, summaries)
    for summary in summaries:
        print(
            f"RESULT scenario={summary.scenario} size={summary.n_subjects}x{summary.n_items} "
            f"backend={summary.backend} rate={summary.rate:.3f} mc_se={summary.mc_se:.3f} "
            f"observed_singular={summary.observed_singular_fraction:.3f} "
            f"permuted_singular={summary.permuted_singular_fraction:.3f} "
            f"mean_seconds_per_sim={summary.mean_seconds_per_sim:.3f} complete={summary.complete}",
            flush=True,
        )
    print(
        f"DONE total_new_seconds={time.perf_counter() - started:.3f} "
        f"results={results_path} summary={summary_path}",
        flush=True,
    )


def _print_progress(progress_every: int, completed_now: int, total: int, result: SimResult) -> None:
    if progress_every and completed_now % progress_every == 0:
        print(
            f"PROGRESS completed_now={completed_now}/{total} "
            f"last={result.scenario} {result.n_subjects}x{result.n_items} sim={result.sim} "
            f"seconds={result.elapsed_seconds:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
