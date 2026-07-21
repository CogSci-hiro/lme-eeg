"""Calibrate MixedModels.jl Wald-refit permutation maps at dev scale.

This is a Release 2 Route A calibration artifact, not production code. It
generates the existing C1/C2/C3 simulation cells, refits the full MixedModels.jl
model for the observed and every within-subject permutation, and summarizes the
existing Python maxstat/cluster/TFCE correction helpers.
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
from tests.backends.lmm.test_pymer4_calibration_slow import (  # noqa: E402
    ALPHA,
    _assert_generator_h0_is_null_at_every_feature,
    _mc_se,
    _simulate_null_or_power,
)
from tests.backends.lmm.test_real_lmm_null_calibration_slow import (  # noqa: E402
    _cluster_rejects,
    _maxstat_rejects,
    _permute_condition_within_subject,
    _tfce_rejects,
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


def _load_julia_wald() -> Any:
    global _JL
    if _JL is not None:
        return _JL
    _configure_julia()
    from juliacall import Main as jl

    jl.seval("using MixedModels, DataFrames, StatsModels")
    jl.seval(
        r"""
        function lmeeeg_wald_dataframe(y, cond, subject, item)
            return DataFrame(
                y = Float64.(pyconvert(Vector, y)),
                cond = String.(pyconvert(Vector, cond)),
                subject = String.(pyconvert(Vector, subject)),
                item = String.(pyconvert(Vector, item)),
            )
        end

        function lmeeeg_wald_formula(random_slope)
            return random_slope ?
                @formula(y ~ 1 + cond + (1 + cond | subject) + (1 | item)) :
                @formula(y ~ 1 + cond + (1 | subject) + (1 | item))
        end

        function lmeeeg_condition_wald_t_and_singular(y, cond, subject, item, random_slope)
            df = lmeeeg_wald_dataframe(y, cond, subject, item)
            model = fit(MixedModel, lmeeeg_wald_formula(random_slope), df; REML=true)
            index = findfirst(==("cond: B"), fixefnames(model))
            if index === nothing
                error("Could not find MixedModels condition fixed-effect term 'cond: B'")
            end
            statistic = fixef(model)[index] / stderror(model)[index]
            return (statistic, issingular(model))
        end
        """
    )
    _JL = jl
    return jl


def _worker_init() -> None:
    os.environ.setdefault("PYTHON_JULIACALL_EXE", str(JULIA_EXE))
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(JULIA_PROJECT))
    _load_julia_wald()


def _fit_wald_map(eeg: np.ndarray, metadata, random_slope: bool) -> tuple[np.ndarray, int, int]:
    jl = _load_julia_wald()
    cond, subject, item = _metadata_columns(metadata)
    n_locations, n_times = eeg.shape[1], eeg.shape[2]
    t_map = np.empty((n_locations, n_times), dtype=float)
    singular_count = 0
    total_count = 0
    for location in range(n_locations):
        for time_index in range(n_times):
            statistic, singular = jl.lmeeeg_condition_wald_t_and_singular(
                eeg[:, location, time_index],
                cond,
                subject,
                item,
                random_slope,
            )
            t_map[location, time_index] = float(statistic)
            singular_count += int(bool(singular))
            total_count += 1
    return t_map, singular_count, total_count


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
    observed_t, observed_singular, observed_total = _fit_wald_map(
        eeg=eeg,
        metadata=metadata,
        random_slope=random_slope,
    )
    null_t = np.empty((n_permutations,) + observed_t.shape, dtype=float)
    permuted_singular = 0
    permuted_total = 0
    rng = np.random.default_rng(seed + 700_000)
    for permutation_index in range(n_permutations):
        permuted_metadata = _permute_condition_within_subject(metadata=metadata, rng=rng)
        permuted_t, singular_count, total_count = _fit_wald_map(
            eeg=eeg,
            metadata=permuted_metadata,
            random_slope=random_slope,
        )
        null_t[permutation_index] = permuted_t
        permuted_singular += singular_count
        permuted_total += total_count

    return SimResult(
        scenario=scenario,
        n_subjects=n_subjects,
        n_items=n_items,
        sim=sim,
        seed=seed,
        n_permutations=n_permutations,
        maxstat=_maxstat_rejects(observed_t, null_t),
        cluster=_cluster_rejects(observed_t, null_t),
        tfce=_tfce_rejects(observed_t, null_t),
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


def _summarize(results: list[SimResult], n_sims: int, n_permutations: int) -> list[CellSummary]:
    summaries = []
    by_cell: dict[tuple[str, int, int], list[SimResult]] = {}
    for result in results:
        by_cell.setdefault((result.scenario, result.n_subjects, result.n_items), []).append(result)
    for (scenario, n_subjects, n_items), cell_results in sorted(by_cell.items()):
        if len(cell_results) != n_sims:
            raise RuntimeError(
                f"Incomplete cell {scenario} {n_subjects}x{n_items}: "
                f"{len(cell_results)}/{n_sims} sims."
            )
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
    parser.add_argument("--output-dir", type=Path, default=Path("dev/bench/mixedmodels_wald_refit_calibration"))
    parser.add_argument("--n-sims", type=int, default=int(os.environ.get("LMEEG_JULIA_WALD_N_SIMS", "300")))
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=int(os.environ.get("LMEEG_JULIA_WALD_N_PERMUTATIONS", "500")),
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LMEEG_JULIA_WALD_WORKERS", "1")))
    parser.add_argument("--seed-offset", type=int, default=210_000)
    parser.add_argument("--progress-every", type=int, default=int(os.environ.get("LMEEG_PROGRESS_EVERY", "10")))
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--sizes", nargs="+", default=[f"{subjects}x{items}" for subjects, items in SIZES])
    parser.add_argument("--skip-guard", action="store_true")
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

    if not args.skip_guard:
        guard_start = time.perf_counter()
        _run_guard()
        print(f"GUARD random_slope_h0 passed seconds={time.perf_counter() - guard_start:.3f}", flush=True)

    completed = _completed_keys(results_path)
    tasks = _iter_tasks(args.scenarios, sizes, args.n_sims, completed)
    print(
        f"START scenarios={args.scenarios} sizes={sizes} n_sims={args.n_sims} "
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
                if args.progress_every and completed_now % args.progress_every == 0:
                    print(
                        f"PROGRESS completed_now={completed_now}/{len(tasks)} "
                        f"last={result.scenario} {result.n_subjects}x{result.n_items} sim={result.sim} "
                        f"seconds={result.elapsed_seconds:.3f}",
                        flush=True,
                    )
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
                    if args.progress_every and completed_now % args.progress_every == 0:
                        print(
                            f"PROGRESS completed_now={completed_now}/{len(tasks)} "
                            f"last={result.scenario} {result.n_subjects}x{result.n_items} sim={result.sim} "
                            f"seconds={result.elapsed_seconds:.3f}",
                            flush=True,
                        )

    results = _load_results(results_path)
    summaries = _summarize(results, n_sims=args.n_sims, n_permutations=args.n_permutations)
    _write_summary(summary_path, summaries)
    for summary in summaries:
        print(
            f"RESULT scenario={summary.scenario} size={summary.n_subjects}x{summary.n_items} "
            f"backend={summary.backend} rate={summary.rate:.3f} mc_se={summary.mc_se:.3f} "
            f"observed_singular={summary.observed_singular_fraction:.3f} "
            f"permuted_singular={summary.permuted_singular_fraction:.3f} "
            f"mean_seconds_per_sim={summary.mean_seconds_per_sim:.3f}",
            flush=True,
        )
    print(
        f"DONE total_new_seconds={time.perf_counter() - started:.3f} "
        f"results={results_path} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
