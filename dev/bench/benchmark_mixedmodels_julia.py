"""Benchmark MixedModels.jl through juliacall as an engine-only LMM fitter.

This script is intentionally a dev/calibration artifact, not production code.
It keeps permutation/correction logic in Python and asks Julia only to fit one
dataset and return the Satterthwaite condition t statistic.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.backends.lmm.test_pymer4_calibration_slow import (
    _fit_lmer_condition_effect,
    _simulate_null_or_power,
)


JULIA_EXE = ROOT / ".venv" / "julia_env" / "pyjuliapkg" / "install" / "bin" / "julia"
JULIA_PROJECT = ROOT / "dev" / "julia" / "mixedmodels_bench"
SIZES = [(6, 5), (12, 10), (24, 20), (36, 30)]
N_FEATURES_CALIBRATION = 4
N_SCENARIOS = 3
N_CALIBRATION_SIMS = 300
N_CALIBRATION_PERMUTATIONS = 500
REAL_ANALYSIS_FEATURES = 64 * 500
REAL_ANALYSIS_PERMUTATIONS = 1000


def _configure_julia() -> None:
    os.environ.setdefault("PYTHON_JULIACALL_EXE", str(JULIA_EXE))
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(JULIA_PROJECT))


def _configure_pymer4() -> None:
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
    local_r_library = ROOT / ".venv" / "R" / "library"
    if local_r_library.exists():
        current = os.environ.get("R_LIBS_USER")
        local = str(local_r_library)
        if current:
            if local not in current.split(os.pathsep):
                os.environ["R_LIBS_USER"] = os.pathsep.join([local, current])
        else:
            os.environ["R_LIBS_USER"] = local


def _load_julia() -> tuple[Any, float]:
    _configure_julia()
    start = time.perf_counter()
    from juliacall import Main as jl

    jl.seval("using MixedModels, MixedModelsSmallSample, DataFrames, StatsModels")
    jl.seval(
        r"""
        function lmeeeg_condition_t(y, cond, subject, item, random_slope)
            df = DataFrame(
                y = Float64.(pyconvert(Vector, y)),
                cond = String.(pyconvert(Vector, cond)),
                subject = String.(pyconvert(Vector, subject)),
                item = String.(pyconvert(Vector, item)),
            )
            formula = random_slope ?
                @formula(y ~ 1 + cond + (1 + cond | subject) + (1 | item)) :
                @formula(y ~ 1 + cond + (1 | subject) + (1 | item))
            model = fit(MixedModel, formula, df; REML=true)
            adjusted = small_sample_adjust(model, Satterthwaite())
            index = findfirst(==("cond: B"), fixefnames(model))
            if index === nothing
                error("Could not find MixedModels condition fixed-effect term 'cond: B'")
            end
            table = coeftable(adjusted)
            return table.cols[4][index]
        end
        """
    )
    return jl, time.perf_counter() - start


def _metadata_columns(metadata) -> tuple[list[str], list[str], list[str]]:
    return (
        metadata["cond"].astype(str).to_list(),
        metadata["subject"].astype(str).to_list(),
        metadata["item"].astype(str).to_list(),
    )


def _julia_condition_t(jl: Any, y: np.ndarray, metadata, random_slope: bool) -> float:
    cond, subject, item = _metadata_columns(metadata)
    return float(jl.lmeeeg_condition_t(y, cond, subject, item, random_slope))


def _pymer4_condition_t(y: np.ndarray, metadata, random_slope: bool) -> float:
    estimate, standard_error, _ = _fit_lmer_condition_effect(y, metadata, random_slope=random_slope)
    return float(estimate / standard_error)


def _oracle_agreement(jl: Any) -> dict[str, float]:
    differences: dict[str, float] = {}
    for label, random_slope in [("crossed_intercept", False), ("random_slope", True)]:
        per_feature_diffs = []
        eeg, metadata = _simulate_null_or_power(
            seed=170_000 + int(random_slope),
            random_slope=random_slope,
            fixed_effect=0.0,
            n_subjects=12,
            n_items=10,
        )
        for location in range(eeg.shape[1]):
            for time_index in range(eeg.shape[2]):
                y = eeg[:, location, time_index]
                pymer4_t = _pymer4_condition_t(y, metadata, random_slope=random_slope)
                julia_t = _julia_condition_t(jl, y, metadata, random_slope=random_slope)
                per_feature_diffs.append(abs(julia_t - pymer4_t))
        differences[label] = float(np.max(per_feature_diffs))
    return differences


def _benchmark_size(
    jl: Any,
    n_subjects: int,
    n_items: int,
    random_slope: bool,
    n_repeats: int,
) -> dict[str, float]:
    datasets = [
        _simulate_null_or_power(
            seed=180_000 + n_subjects * 1_000 + n_items * 10 + repeat + (10_000 if random_slope else 0),
            random_slope=random_slope,
            fixed_effect=0.0,
            n_subjects=n_subjects,
            n_items=n_items,
        )
        for repeat in range(n_repeats + 1)
    ]

    warm_eeg, warm_metadata = datasets[0]
    warm_start = time.perf_counter()
    _julia_condition_t(jl, warm_eeg[:, 0, 0], warm_metadata, random_slope=random_slope)
    warm_seconds = time.perf_counter() - warm_start

    timings = []
    for eeg, metadata in datasets[1:]:
        start = time.perf_counter()
        _julia_condition_t(jl, eeg[:, 0, 0], metadata, random_slope=random_slope)
        timings.append(time.perf_counter() - start)

    return {
        "warm_seconds": warm_seconds,
        "mean_seconds": float(np.mean(timings)),
        "median_seconds": float(np.median(timings)),
        "min_seconds": float(np.min(timings)),
        "max_seconds": float(np.max(timings)),
    }


def _format_duration(seconds: float) -> str:
    hours = seconds / 3600.0
    days = hours / 24.0
    if days >= 1.0:
        return f"{days:.2f} days"
    if hours >= 1.0:
        return f"{hours:.2f} hours"
    return f"{seconds / 60.0:.2f} minutes"


def main() -> None:
    _configure_pymer4()
    n_repeats = int(os.environ.get("LMEEG_JULIA_BENCH_REPEATS", "10"))
    jl, load_seconds = _load_julia()
    oracle = _oracle_agreement(jl)

    print(f"JULIA_LOAD_SECONDS {load_seconds:.3f}", flush=True)
    for label, difference in oracle.items():
        print(f"ORACLE {label} max_abs_t_diff={difference:.9g}", flush=True)
    if oracle["crossed_intercept"] > 1e-4 or oracle["random_slope"] > 1e-4:
        raise SystemExit(f"NO-GO: MixedModels t mismatch exceeded tolerance: {oracle!r}")

    timings: dict[tuple[str, str], dict[str, float]] = {}
    for n_subjects, n_items in SIZES:
        for label, random_slope in [("crossed_intercept", False), ("random_slope", True)]:
            result = _benchmark_size(
                jl=jl,
                n_subjects=n_subjects,
                n_items=n_items,
                random_slope=random_slope,
                n_repeats=n_repeats,
            )
            timings[(f"{n_subjects}x{n_items}", label)] = result
            print(
                "TIMING "
                f"model={label} size={n_subjects}x{n_items} repeats={n_repeats} "
                f"warm_seconds={result['warm_seconds']:.6f} "
                f"mean_seconds={result['mean_seconds']:.6f} "
                f"median_seconds={result['median_seconds']:.6f} "
                f"min_seconds={result['min_seconds']:.6f} "
                f"max_seconds={result['max_seconds']:.6f}",
                flush=True,
            )

    calibration_fits = N_SCENARIOS * len(SIZES) * N_CALIBRATION_SIMS * N_CALIBRATION_PERMUTATIONS
    calibration_feature_fits = calibration_fits * N_FEATURES_CALIBRATION
    slope_36x30 = timings[("36x30", "random_slope")]["mean_seconds"]
    calibration_seconds = calibration_feature_fits * slope_36x30
    real_analysis_fits = REAL_ANALYSIS_FEATURES * REAL_ANALYSIS_PERMUTATIONS
    real_analysis_seconds = real_analysis_fits * slope_36x30
    print(
        f"PROJECTION calibration_feature_fits={calibration_feature_fits} "
        f"using_slope_36x30_seconds={slope_36x30:.6f} "
        f"wall_time={_format_duration(calibration_seconds)}",
        flush=True,
    )
    print(
        f"PROJECTION real_analysis_fits={real_analysis_fits} "
        f"features={REAL_ANALYSIS_FEATURES} permutations={REAL_ANALYSIS_PERMUTATIONS} "
        f"using_slope_36x30_seconds={slope_36x30:.6f} "
        f"wall_time={_format_duration(real_analysis_seconds)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
