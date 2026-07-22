"""Calibrate nested sEEG parametric-bootstrap LMM nulls at dev scale.

This Release 2 artifact is not production code. It simulates a nested
contacts-in-subjects sEEG design, fits full/reduced MixedModels.jl models by ML,
uses parametric bootstrap from the fitted reduced model as the family-wise null,
and summarizes maxstat, cluster, and TFCE over one-dimensional time maps.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dev.bench.benchmark_mixedmodels_julia import JULIA_EXE, JULIA_PROJECT, _configure_julia  # noqa: E402
from lmeeeg.backends.correction._regression import cluster_outputs_to_masks, configure_mne_runtime  # noqa: E402
from tests.backends.lmm.test_pymer4_calibration_slow import ALPHA, _mc_se  # noqa: E402


SUBJECT_SIZES = (6, 15)
RE_VARIANTS = ("R1", "R2")
SCENARIOS = ("C1", "C2", "C3")
BACKENDS = ("maxstat", "cluster", "tfce")

DEFAULT_CONTACTS_PER_SUBJECT = 5
DEFAULT_TRIALS_PER_SUBJECT = 40
DEFAULT_N_FEATURES = 128

_JL: Any | None = None


@dataclass(frozen=True)
class SimConfig:
    n_subjects: int
    re_variant: str
    scenario: str
    contacts_per_subject: int
    trials_per_subject: int
    n_features: int
    n_boot: int
    seed: int


@dataclass(frozen=True)
class SimResult:
    n_subjects: int
    re_variant: str
    scenario: str
    sim: int
    seed: int
    contacts_per_subject: int
    trials_per_subject: int
    n_features: int
    n_boot: int
    maxstat: bool
    cluster: bool
    tfce: bool
    observed_singular_fits: int
    observed_total_fits: int
    boot_singular_fits: int
    boot_total_fits: int
    observed_refit_maps: int
    observed_total_maps: int
    observed_unresolved_negative_maps: int
    boot_refit_maps: int
    boot_total_maps: int
    boot_unresolved_negative_maps: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CellSummary:
    n_subjects: int
    re_variant: str
    scenario: str
    backend: str
    rate: float
    mc_se: float
    n_sims: int
    n_boot: int
    contacts_per_subject: int
    trials_per_subject: int
    n_features: int
    observed_singular_fraction: float
    boot_singular_fraction: float
    observed_refit_fraction: float
    boot_refit_fraction: float
    observed_unresolved_negative_maps: int
    boot_unresolved_negative_maps: int
    mean_seconds_per_sim: float
    total_seconds: float
    complete: bool


@dataclass(frozen=True)
class _FeatureBootstrap:
    reduced_model: Any


@dataclass(frozen=True)
class _BootstrapSetup:
    observed_lr: np.ndarray
    features: list[_FeatureBootstrap]
    singular_count: int
    total_count: int
    refit_maps: int
    unresolved_negative_maps: int


@dataclass(frozen=True)
class _MapFit:
    lr: np.ndarray
    singular_count: int
    total_count: int
    refit_maps: int
    unresolved_negative_maps: int


def _load_julia() -> Any:
    global _JL
    if _JL is not None:
        return _JL
    _configure_julia()
    from juliacall import Main as jl

    jl.seval("using MixedModels, DataFrames, Random, StatsModels")
    jl.seval(
        r"""
        function lmeeeg_nested_dataframe(y, cond, subject, contact)
            return DataFrame(
                y = Float64.(pyconvert(Vector, y)),
                cond = String.(pyconvert(Vector, cond)),
                subject = String.(pyconvert(Vector, subject)),
                contact = String.(pyconvert(Vector, contact)),
            )
        end

        function lmeeeg_nested_full_formula(re_variant)
            if re_variant == "R1"
                return @formula(y ~ 1 + cond + (1 + cond | subject) + (1 + cond | contact))
            elseif re_variant == "R2"
                return @formula(y ~ 1 + cond + (1 + cond | subject) + (1 | contact))
            else
                error("Unknown re_variant: " * string(re_variant))
            end
        end

        function lmeeeg_nested_reduced_formula(re_variant)
            if re_variant == "R1"
                return @formula(y ~ 1 + (1 + cond | subject) + (1 + cond | contact))
            elseif re_variant == "R2"
                return @formula(y ~ 1 + (1 + cond | subject) + (1 | contact))
            else
                error("Unknown re_variant: " * string(re_variant))
            end
        end

        function lmeeeg_nested_fit_model(formula, df, optimizer, initial_scale)
            model = LinearMixedModel(formula, df)
            model.optsum.ftol_rel = 1.0e-14
            model.optsum.ftol_abs = 1.0e-10
            model.optsum.xtol_rel = 0.0
            model.optsum.xtol_abs = fill(1.0e-12, length(model.optsum.initial))
            model.optsum.maxfeval = 10_000
            model.optsum.xtol_zero_abs = 1.0e-8
            model.optsum.ftol_zero_abs = 1.0e-10
            if initial_scale != 1.0
                model.optsum.initial .= model.optsum.initial .* initial_scale
            end
            fit!(model; REML=false, progress=false, backend=:nlopt, optimizer=optimizer)
            return model
        end

        function lmeeeg_nested_lrt_attempt(y, cond, subject, contact, re_variant, optimizer, initial_scale)
            df = lmeeeg_nested_dataframe(y, cond, subject, contact)
            full = lmeeeg_nested_fit_model(
                lmeeeg_nested_full_formula(re_variant),
                df,
                optimizer,
                initial_scale,
            )
            reduced = lmeeeg_nested_fit_model(
                lmeeeg_nested_reduced_formula(re_variant),
                df,
                optimizer,
                initial_scale,
            )
            statistic = 2.0 * (loglikelihood(full) - loglikelihood(reduced))
            return (statistic, issingular(full), issingular(reduced))
        end

        function lmeeeg_nested_lrt(y, cond, subject, contact, re_variant)
            attempts = (
                (:LN_BOBYQA, 1.0),
                (:LN_BOBYQA, 0.5),
                (:LN_NEWUOA, 1.0),
                (:LN_NELDERMEAD, 1.0),
                (:LN_NELDERMEAD, 0.5),
                (:LN_COBYLA, 1.0),
            )
            statistic, full_singular, reduced_singular =
                lmeeeg_nested_lrt_attempt(y, cond, subject, contact, re_variant, attempts[1]...)
            refit_triggered = false
            unresolved_negative = false
            if statistic < -1.0e-8
                refit_triggered = true
                best_statistic = statistic
                best_full_singular = full_singular
                best_reduced_singular = reduced_singular
                for attempt in attempts[2:end]
                    retry_statistic, retry_full_singular, retry_reduced_singular =
                        lmeeeg_nested_lrt_attempt(y, cond, subject, contact, re_variant, attempt...)
                    if retry_statistic > best_statistic
                        best_statistic = retry_statistic
                        best_full_singular = retry_full_singular
                        best_reduced_singular = retry_reduced_singular
                    end
                    if retry_statistic >= -1.0e-8
                        return (
                            retry_statistic,
                            retry_full_singular,
                            retry_reduced_singular,
                            refit_triggered,
                            unresolved_negative,
                        )
                    end
                end
                statistic = best_statistic
                full_singular = best_full_singular
                reduced_singular = best_reduced_singular
                unresolved_negative = true
            end
            return (statistic, full_singular, reduced_singular, refit_triggered, unresolved_negative)
        end

        function lmeeeg_nested_bootstrap_parts(y, cond, subject, contact, re_variant)
            attempts = (
                (:LN_BOBYQA, 1.0),
                (:LN_BOBYQA, 0.5),
                (:LN_NEWUOA, 1.0),
                (:LN_NELDERMEAD, 1.0),
                (:LN_NELDERMEAD, 0.5),
                (:LN_COBYLA, 1.0),
            )
            df = lmeeeg_nested_dataframe(y, cond, subject, contact)
            full = lmeeeg_nested_fit_model(
                lmeeeg_nested_full_formula(re_variant),
                df,
                attempts[1]...,
            )
            reduced = lmeeeg_nested_fit_model(
                lmeeeg_nested_reduced_formula(re_variant),
                df,
                attempts[1]...,
            )
            statistic = 2.0 * (loglikelihood(full) - loglikelihood(reduced))
            refit_triggered = false
            unresolved_negative = false
            if statistic < -1.0e-8
                refit_triggered = true
                best_statistic = statistic
                best_full = full
                best_reduced = reduced
                for attempt in attempts[2:end]
                    retry_full = lmeeeg_nested_fit_model(
                        lmeeeg_nested_full_formula(re_variant),
                        df,
                        attempt...,
                    )
                    retry_reduced = lmeeeg_nested_fit_model(
                        lmeeeg_nested_reduced_formula(re_variant),
                        df,
                        attempt...,
                    )
                    retry_statistic = 2.0 * (loglikelihood(retry_full) - loglikelihood(retry_reduced))
                    if retry_statistic > best_statistic
                        best_statistic = retry_statistic
                        best_full = retry_full
                        best_reduced = retry_reduced
                    end
                    if retry_statistic >= -1.0e-8
                        return (
                            retry_statistic,
                            issingular(retry_full),
                            issingular(retry_reduced),
                            retry_reduced,
                            refit_triggered,
                            unresolved_negative,
                        )
                    end
                end
                statistic = best_statistic
                full = best_full
                reduced = best_reduced
                unresolved_negative = true
            end
            return (
                statistic,
                issingular(full),
                issingular(reduced),
                reduced,
                refit_triggered,
                unresolved_negative,
            )
        end

        function lmeeeg_nested_simulate_reduced(reduced_model, seed)
            return simulate(MersenneTwister(seed), reduced_model)
        end

        function lmeeeg_nested_contract()
            return (
                R1_full = string(lmeeeg_nested_full_formula("R1")),
                R1_reduced = string(lmeeeg_nested_reduced_formula("R1")),
                R2_full = string(lmeeeg_nested_full_formula("R2")),
                R2_reduced = string(lmeeeg_nested_reduced_formula("R2")),
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
    _load_julia()


def _simulate_nested_seeg(config: SimConfig) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    rows = []
    cond_values = []
    subjects = []
    contacts = []
    for subject_index in range(config.n_subjects):
        subject = f"s{subject_index:02d}"
        trial_cond = np.array(["A"] * (config.trials_per_subject // 2), dtype=object)
        trial_cond = np.concatenate(
            [trial_cond, np.array(["B"] * (config.trials_per_subject - trial_cond.size), dtype=object)]
        )
        rng.shuffle(trial_cond)
        for contact_index in range(config.contacts_per_subject):
            contact = f"{subject}:c{contact_index:02d}"
            for trial_index, condition in enumerate(trial_cond):
                rows.append((subject, contact, f"{subject}:t{trial_index:03d}"))
                cond_values.append(condition)
                subjects.append(subject)
                contacts.append(contact)
    metadata = pd.DataFrame(
        {
            "subject": subjects,
            "contact": contacts,
            "trial": [row[2] for row in rows],
            "cond": cond_values,
        }
    )
    cond_numeric = (metadata["cond"].to_numpy() == "B").astype(float)
    subject_codes = metadata["subject"].astype("category").cat.codes.to_numpy()
    contact_codes = metadata["contact"].astype("category").cat.codes.to_numpy()

    n_observations = metadata.shape[0]
    time = np.linspace(-1.0, 1.0, config.n_features)
    effect_shape = 0.35 + np.exp(-0.5 * (time / 0.25) ** 2)
    effect_shape /= float(np.max(effect_shape))
    fixed_effect = 0.0 if config.scenario != "C3" else 0.45
    beta = fixed_effect * effect_shape

    subject_intercepts = rng.normal(0.0, 0.55, size=config.n_subjects)
    contact_intercepts = rng.normal(0.0, 0.35, size=config.n_subjects * config.contacts_per_subject)
    subject_slope_sd = 0.0 if config.scenario == "C1" else 0.40
    contact_slope_sd = 0.0 if config.scenario == "C1" or config.re_variant == "R2" else 0.25
    subject_slopes = rng.normal(0.0, subject_slope_sd, size=config.n_subjects)
    contact_slopes = rng.normal(0.0, contact_slope_sd, size=config.n_subjects * config.contacts_per_subject)

    eeg = np.empty((n_observations, config.n_features), dtype=float)
    for feature_index in range(config.n_features):
        noise = rng.normal(0.0, 1.0, size=n_observations)
        eeg[:, feature_index] = (
            0.5
            + beta[feature_index] * cond_numeric
            + subject_intercepts[subject_codes]
            + contact_intercepts[contact_codes]
            + subject_slopes[subject_codes] * cond_numeric
            + contact_slopes[contact_codes] * cond_numeric
            + noise
        )
    return eeg, metadata


def _metadata_columns(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        metadata["cond"].to_numpy(dtype=str),
        metadata["subject"].to_numpy(dtype=str),
        metadata["contact"].to_numpy(dtype=str),
    )


def _fit_bootstrap_setup(eeg: np.ndarray, metadata: pd.DataFrame, re_variant: str) -> _BootstrapSetup:
    jl = _load_julia()
    cond, subject, contact = _metadata_columns(metadata)
    observed_lr = np.empty(eeg.shape[1], dtype=float)
    features = []
    singular_count = 0
    total_count = 0
    refit_maps = 0
    unresolved = 0
    for feature_index in range(eeg.shape[1]):
        statistic, full_singular, reduced_singular, reduced_model, refit_triggered, unresolved_negative = (
            jl.lmeeeg_nested_bootstrap_parts(
                eeg[:, feature_index],
                cond,
                subject,
                contact,
                re_variant,
            )
        )
        statistic = float(statistic)
        if statistic < -1e-8:
            raise RuntimeError(
                f"Observed negative LR statistic {statistic} remained after convergence refit "
                f"at feature {feature_index}."
            )
        observed_lr[feature_index] = max(0.0, statistic)
        singular_count += int(bool(full_singular)) + int(bool(reduced_singular))
        total_count += 2
        refit_maps += int(bool(refit_triggered))
        unresolved += int(bool(unresolved_negative))
        features.append(_FeatureBootstrap(reduced_model=reduced_model))
    return _BootstrapSetup(
        observed_lr=observed_lr,
        features=features,
        singular_count=singular_count,
        total_count=total_count,
        refit_maps=refit_maps,
        unresolved_negative_maps=unresolved,
    )


def _fit_lrt_map(eeg: np.ndarray, metadata: pd.DataFrame, re_variant: str) -> _MapFit:
    jl = _load_julia()
    cond, subject, contact = _metadata_columns(metadata)
    lr = np.empty(eeg.shape[1], dtype=float)
    singular_count = 0
    total_count = 0
    refit_maps = 0
    unresolved = 0
    for feature_index in range(eeg.shape[1]):
        statistic, full_singular, reduced_singular, refit_triggered, unresolved_negative = jl.lmeeeg_nested_lrt(
            eeg[:, feature_index],
            cond,
            subject,
            contact,
            re_variant,
        )
        statistic = float(statistic)
        if statistic < -1e-8:
            raise RuntimeError(
                f"Bootstrap negative LR statistic {statistic} remained after convergence refit "
                f"at feature {feature_index}."
            )
        lr[feature_index] = max(0.0, statistic)
        singular_count += int(bool(full_singular)) + int(bool(reduced_singular))
        total_count += 2
        refit_maps += int(bool(refit_triggered))
        unresolved += int(bool(unresolved_negative))
    return _MapFit(
        lr=lr,
        singular_count=singular_count,
        total_count=total_count,
        refit_maps=refit_maps,
        unresolved_negative_maps=unresolved,
    )


def _simulate_bootstrap_map(setup: _BootstrapSetup, seed: int, bootstrap_index: int, n_observations: int) -> np.ndarray:
    jl = _load_julia()
    simulated = np.empty((n_observations, len(setup.features)), dtype=float)
    for feature_index, feature in enumerate(setup.features):
        draw_seed = seed + 1_700_000 + bootstrap_index * 10_000 + feature_index
        simulated[:, feature_index] = np.asarray(
            jl.lmeeeg_nested_simulate_reduced(feature.reduced_model, draw_seed),
            dtype=float,
        )
    return simulated


def _maxstat_rejects(observed_lr: np.ndarray, null_lr: np.ndarray) -> bool:
    null_distribution = np.max(null_lr, axis=1)
    corrected = (
        1
        + np.sum(null_distribution[:, None] >= observed_lr[None, :], axis=0)
    ) / (null_lr.shape[0] + 1)
    return bool(np.nanmin(corrected) <= ALPHA)


def _cluster_rejects(observed_lr: np.ndarray, null_lr: np.ndarray) -> bool:
    configure_mne_runtime()
    from mne.stats.cluster_level import _find_clusters

    threshold = 4.0
    observed_2d = observed_lr[None, :]
    raw_clusters, cluster_stats = _find_clusters(observed_2d.T, threshold=threshold, tail=1)
    cluster_masks = cluster_outputs_to_masks(raw_clusters, sample_shape=(observed_lr.size, 1))
    if not cluster_masks:
        return False
    null_distribution = np.zeros(null_lr.shape[0], dtype=float)
    for index, permuted_lr in enumerate(null_lr):
        _, permuted_cluster_stats = _find_clusters(permuted_lr[None, :].T, threshold=threshold, tail=1)
        null_distribution[index] = float(np.max(permuted_cluster_stats)) if len(permuted_cluster_stats) else 0.0
    p_values = np.asarray(
        [(1 + np.sum(null_distribution >= cluster_stat)) / (null_lr.shape[0] + 1) for cluster_stat in cluster_stats],
        dtype=float,
    )
    return bool(np.nanmin(p_values) <= ALPHA)


def _tfce_rejects(observed_lr: np.ndarray, null_lr: np.ndarray) -> bool:
    configure_mne_runtime()
    from mne.stats.cluster_level import _find_clusters

    threshold = {"start": 0.0, "step": 0.2, "h_power": 2.0, "e_power": 0.5}
    _, observed_tfce = _find_clusters(observed_lr[None, :].T, threshold=threshold, tail=1)
    observed_tfce = np.asarray(observed_tfce, dtype=float).reshape(observed_lr.size)
    null_distribution = np.zeros(null_lr.shape[0], dtype=float)
    for index, permuted_lr in enumerate(null_lr):
        _, permuted_tfce = _find_clusters(permuted_lr[None, :].T, threshold=threshold, tail=1)
        null_distribution[index] = float(np.max(permuted_tfce))
    corrected = (
        1
        + np.sum(null_distribution[:, None] >= observed_tfce[None, :], axis=0)
    ) / (null_lr.shape[0] + 1)
    return bool(np.nanmin(corrected) <= ALPHA)


def _run_one_sim(config: SimConfig, sim: int) -> SimResult:
    start = time.perf_counter()
    eeg, metadata = _simulate_nested_seeg(config)
    setup = _fit_bootstrap_setup(eeg=eeg, metadata=metadata, re_variant=config.re_variant)
    null_lr = np.empty((config.n_boot, config.n_features), dtype=float)
    boot_singular = 0
    boot_total = 0
    boot_refits = 0
    boot_unresolved = 0
    for bootstrap_index in range(config.n_boot):
        simulated = _simulate_bootstrap_map(
            setup=setup,
            seed=config.seed,
            bootstrap_index=bootstrap_index,
            n_observations=eeg.shape[0],
        )
        fit = _fit_lrt_map(eeg=simulated, metadata=metadata, re_variant=config.re_variant)
        null_lr[bootstrap_index] = fit.lr
        boot_singular += fit.singular_count
        boot_total += fit.total_count
        boot_refits += fit.refit_maps
        boot_unresolved += fit.unresolved_negative_maps
    return SimResult(
        n_subjects=config.n_subjects,
        re_variant=config.re_variant,
        scenario=config.scenario,
        sim=sim,
        seed=config.seed,
        contacts_per_subject=config.contacts_per_subject,
        trials_per_subject=config.trials_per_subject,
        n_features=config.n_features,
        n_boot=config.n_boot,
        maxstat=_maxstat_rejects(setup.observed_lr, null_lr),
        cluster=_cluster_rejects(setup.observed_lr, null_lr),
        tfce=_tfce_rejects(setup.observed_lr, null_lr),
        observed_singular_fits=setup.singular_count,
        observed_total_fits=setup.total_count,
        boot_singular_fits=boot_singular,
        boot_total_fits=boot_total,
        observed_refit_maps=setup.refit_maps,
        observed_total_maps=config.n_features,
        observed_unresolved_negative_maps=setup.unresolved_negative_maps,
        boot_refit_maps=boot_refits,
        boot_total_maps=config.n_boot * config.n_features,
        boot_unresolved_negative_maps=boot_unresolved,
        elapsed_seconds=time.perf_counter() - start,
    )


def _seed_for(n_subjects: int, re_variant: str, scenario: str, sim: int, seed_offset: int) -> int:
    variant_offset = {"R1": 10_000, "R2": 20_000}[re_variant]
    scenario_offset = {"C1": 100_000, "C2": 200_000, "C3": 300_000}[scenario]
    return seed_offset + scenario_offset + variant_offset + n_subjects * 1_000 + sim


def _completed_keys(results_path: Path) -> set[tuple[int, str, str, int]]:
    keys = set()
    if not results_path.exists():
        return keys
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                keys.add(
                    (
                        int(payload["n_subjects"]),
                        str(payload["re_variant"]),
                        str(payload["scenario"]),
                        int(payload["sim"]),
                    )
                )
    return keys


def _iter_tasks(
    n_subjects_values: Iterable[int],
    re_variants: Iterable[str],
    scenarios: Iterable[str],
    n_sims: int,
    completed: set[tuple[int, str, str, int]],
) -> list[tuple[int, str, str, int]]:
    tasks = []
    for n_subjects in n_subjects_values:
        for re_variant in re_variants:
            for scenario in scenarios:
                for sim in range(n_sims):
                    key = (n_subjects, re_variant, scenario, sim)
                    if key not in completed:
                        tasks.append(key)
    return tasks


def _load_results(results_path: Path) -> list[SimResult]:
    results = []
    if not results_path.exists():
        return results
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                results.append(SimResult(**json.loads(line)))
    return results


def _summarize(results: list[SimResult], expected_n_sims: int) -> list[CellSummary]:
    grouped: dict[tuple[int, str, str], list[SimResult]] = {}
    for result in results:
        grouped.setdefault((result.n_subjects, result.re_variant, result.scenario), []).append(result)
    summaries = []
    for (n_subjects, re_variant, scenario), cell_results in sorted(grouped.items()):
        observed_singular_fraction = sum(r.observed_singular_fits for r in cell_results) / sum(
            r.observed_total_fits for r in cell_results
        )
        boot_singular_fraction = sum(r.boot_singular_fits for r in cell_results) / sum(
            r.boot_total_fits for r in cell_results
        )
        observed_refit_fraction = sum(r.observed_refit_maps for r in cell_results) / sum(
            r.observed_total_maps for r in cell_results
        )
        boot_refit_fraction = sum(r.boot_refit_maps for r in cell_results) / sum(
            r.boot_total_maps for r in cell_results
        )
        total_seconds = float(sum(r.elapsed_seconds for r in cell_results))
        mean_seconds = float(np.mean([r.elapsed_seconds for r in cell_results]))
        for backend in BACKENDS:
            values = [bool(getattr(result, backend)) for result in cell_results]
            rate = float(np.mean(values))
            first = cell_results[0]
            summaries.append(
                CellSummary(
                    n_subjects=n_subjects,
                    re_variant=re_variant,
                    scenario=scenario,
                    backend=backend,
                    rate=rate,
                    mc_se=_mc_se(rate, len(values)),
                    n_sims=len(values),
                    n_boot=first.n_boot,
                    contacts_per_subject=first.contacts_per_subject,
                    trials_per_subject=first.trials_per_subject,
                    n_features=first.n_features,
                    observed_singular_fraction=float(observed_singular_fraction),
                    boot_singular_fraction=float(boot_singular_fraction),
                    observed_refit_fraction=float(observed_refit_fraction),
                    boot_refit_fraction=float(boot_refit_fraction),
                    observed_unresolved_negative_maps=sum(r.observed_unresolved_negative_maps for r in cell_results),
                    boot_unresolved_negative_maps=sum(r.boot_unresolved_negative_maps for r in cell_results),
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


def _run_guard(args: argparse.Namespace) -> None:
    for n_subjects in args.n_subjects:
        for re_variant in args.re_variants:
            config = SimConfig(
                n_subjects=n_subjects,
                re_variant=re_variant,
                scenario="C2",
                contacts_per_subject=args.contacts_per_subject,
                trials_per_subject=args.trials_per_subject,
                n_features=min(4, args.n_features),
                n_boot=2,
                seed=_seed_for(n_subjects, re_variant, "C2", 0, args.seed_offset),
            )
            eeg, metadata = _simulate_nested_seeg(config)
            counts = metadata.groupby("subject")["cond"].value_counts().unstack(fill_value=0)
            if not (counts["A"].to_numpy() == counts["B"].to_numpy()).all():
                raise RuntimeError(f"Condition is not balanced within subject for {n_subjects=} {re_variant=}.")
            setup = _fit_bootstrap_setup(eeg=eeg, metadata=metadata, re_variant=re_variant)
            if not np.all(np.isfinite(setup.observed_lr)):
                raise RuntimeError(f"Guard produced non-finite LR map for {n_subjects=} {re_variant=}.")
            print(
                f"GUARD n_subjects={n_subjects} re_variant={re_variant} "
                f"features={config.n_features} max_lr={float(np.max(setup.observed_lr)):.3f} "
                f"singular={setup.singular_count / setup.total_count:.3f}",
                flush=True,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dev/bench/nested_seeg_parametric_bootstrap"))
    parser.add_argument("--n-subjects", nargs="+", type=int, default=list(SUBJECT_SIZES))
    parser.add_argument("--re-variants", nargs="+", choices=RE_VARIANTS, default=list(RE_VARIANTS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--contacts-per-subject", type=int, default=DEFAULT_CONTACTS_PER_SUBJECT)
    parser.add_argument("--trials-per-subject", type=int, default=DEFAULT_TRIALS_PER_SUBJECT)
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--n-sims", type=int, default=300)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LMEEG_NESTED_SEEG_WORKERS", "1")))
    parser.add_argument("--seed-offset", type=int, default=510_000)
    parser.add_argument("--skip-guard", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--progress-every", type=int, default=int(os.environ.get("LMEEG_PROGRESS_EVERY", "10")))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTHON_JULIACALL_EXE", str(JULIA_EXE))
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(JULIA_PROJECT))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sim_results.jsonl"
    summary_path = args.output_dir / "summary.json"

    if args.print_contract:
        print(f"CONTRACT {_load_julia().lmeeeg_nested_contract()}", flush=True)
    if not args.skip_guard:
        guard_start = time.perf_counter()
        _run_guard(args)
        print(f"GUARD completed seconds={time.perf_counter() - guard_start:.3f}", flush=True)

    completed = _completed_keys(results_path)
    tasks = _iter_tasks(args.n_subjects, args.re_variants, args.scenarios, args.n_sims, completed)
    fits_per_sim = args.n_features * (1 + args.n_boot) * 2
    print(
        f"START nested_seeg n_subjects={args.n_subjects} re_variants={args.re_variants} "
        f"scenarios={args.scenarios} n_sims={args.n_sims} n_boot={args.n_boot} "
        f"n_features={args.n_features} workers={args.workers} remaining_tasks={len(tasks)} "
        f"approx_fits_per_sim={fits_per_sim}",
        flush=True,
    )

    started = time.perf_counter()
    completed_now = 0
    with results_path.open("a", encoding="utf-8") as handle:
        if args.workers == 1:
            _worker_init()
            for n_subjects, re_variant, scenario, sim in tasks:
                config = SimConfig(
                    n_subjects=n_subjects,
                    re_variant=re_variant,
                    scenario=scenario,
                    contacts_per_subject=args.contacts_per_subject,
                    trials_per_subject=args.trials_per_subject,
                    n_features=args.n_features,
                    n_boot=args.n_boot,
                    seed=_seed_for(n_subjects, re_variant, scenario, sim, args.seed_offset),
                )
                result = _run_one_sim(config=config, sim=sim)
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                handle.flush()
                completed_now += 1
                _print_progress(args.progress_every, completed_now, len(tasks), result)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
                futures = {}
                for n_subjects, re_variant, scenario, sim in tasks:
                    config = SimConfig(
                        n_subjects=n_subjects,
                        re_variant=re_variant,
                        scenario=scenario,
                        contacts_per_subject=args.contacts_per_subject,
                        trials_per_subject=args.trials_per_subject,
                        n_features=args.n_features,
                        n_boot=args.n_boot,
                        seed=_seed_for(n_subjects, re_variant, scenario, sim, args.seed_offset),
                    )
                    futures[pool.submit(_run_one_sim, config, sim)] = config
                for future in as_completed(futures):
                    result = future.result()
                    handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                    handle.flush()
                    completed_now += 1
                    _print_progress(args.progress_every, completed_now, len(tasks), result)

    summaries = _summarize(_load_results(results_path), expected_n_sims=args.n_sims)
    _write_summary(summary_path, summaries)
    for summary in summaries:
        print(
            f"RESULT n_subjects={summary.n_subjects} re_variant={summary.re_variant} "
            f"scenario={summary.scenario} backend={summary.backend} rate={summary.rate:.3f} "
            f"mc_se={summary.mc_se:.3f} observed_singular={summary.observed_singular_fraction:.3f} "
            f"boot_singular={summary.boot_singular_fraction:.3f} observed_refit={summary.observed_refit_fraction:.3f} "
            f"boot_refit={summary.boot_refit_fraction:.3f} unresolved_negative_maps="
            f"{summary.observed_unresolved_negative_maps + summary.boot_unresolved_negative_maps} "
            f"mean_seconds_per_sim={summary.mean_seconds_per_sim:.3f} complete={summary.complete}",
            flush=True,
        )
    print(
        f"DONE total_new_seconds={time.perf_counter() - started:.3f} "
        f"results={results_path} summary={summary_path}",
        flush=True,
    )


def _print_progress(progress_every: int, completed_now: int, total: int, result: SimResult) -> None:
    if progress_every and (completed_now % progress_every == 0 or completed_now == total):
        print(
            f"PROGRESS completed_now={completed_now}/{total} n_subjects={result.n_subjects} "
            f"re_variant={result.re_variant} scenario={result.scenario} sim={result.sim} "
            f"seconds={result.elapsed_seconds:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
