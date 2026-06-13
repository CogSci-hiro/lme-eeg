import math
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_source_lmeeeg_simulation.py"
SPEC = importlib.util.spec_from_file_location("validate_source_lmeeeg_simulation", SCRIPT_PATH)
validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)

SourceSimulationConfig = validation.SourceSimulationConfig
ReplicateResult = validation.ReplicateResult
run_replicate = validation.run_replicate
run_validation = validation.run_validation
simulate_source_dataset = validation.simulate_source_dataset
wilson_interval = validation.wilson_interval


def test_wilson_interval_matches_known_reference() -> None:
    estimate, lower, upper, half_width = wilson_interval(successes=5, n=10, alpha=0.05)

    assert estimate == 0.5
    assert math.isclose(lower, 0.2366, abs_tol=5e-4)
    assert math.isclose(upper, 0.7634, abs_tol=5e-4)
    assert math.isclose(half_width, 0.2634, abs_tol=5e-4)


def test_simulate_source_dataset_smoke() -> None:
    config = SourceSimulationConfig(
        scenario="recovery",
        n_subjects=3,
        trials_per_subject=4,
        n_sources=5,
        n_times=6,
        effect_size=0.7,
    )

    simulated = simulate_source_dataset(config=config, seed=1)

    assert simulated.eeg.shape == (12, 5, 6)
    assert simulated.eeg.dtype.name == "float32"
    assert list(simulated.metadata.columns) == ["subject", "condition"]
    assert simulated.effect_map.shape == (5, 6)
    assert simulated.effect_mask.any()
    assert simulated.source_names == ["src-00000", "src-00001", "src-00002", "src-00003", "src-00004"]


def test_run_recovery_replicate_source_space_smoke() -> None:
    config = SourceSimulationConfig(
        scenario="recovery",
        n_subjects=2,
        trials_per_subject=4,
        n_sources=2,
        n_times=2,
        effect_size=1.0,
        random_intercept_sd=0.2,
        noise_sd=0.5,
    )

    result = run_replicate(
        replicate=0,
        scenario="recovery",
        seed=3,
        simulation_config=config,
        n_permutations=2,
        alpha=0.05,
        spatial_chunk_size=1,
        time_chunk_size=1,
        check_chunk_equivalence=True,
    )

    assert result.scenario == "recovery"
    assert result.n_observations == 8
    assert result.n_sources == 2
    assert result.n_times == 2
    assert result.n_permutations == 2
    assert result.chunk_equivalence_checked is True
    assert result.chunk_t_close is True
    assert result.status in {"ok", "model_failed", "permutation_failed", "other_failed"}
    assert result.total_seconds >= 0.0
    assert hasattr(result, "fit_seconds")
    assert hasattr(result, "permutation_seconds")
    assert hasattr(result, "seconds_per_permutation")
    if result.status == "ok":
        assert result.metric_success in {True, False}
        assert result.min_corrected_p is not None
        assert result.fit_seconds is not None
        assert result.permutation_seconds is not None


def test_stopping_guard_blocks_tolerance_when_model_failure_rate_high(monkeypatch, tmp_path) -> None:
    statuses = ["ok", "model_failed", "ok", "model_failed"]

    def fake_run_replicate(**kwargs):
        index = kwargs["replicate"]
        return _fake_replicate(index=index, status=statuses[index])

    monkeypatch.setattr(validation, "run_replicate", fake_run_replicate)

    summary = run_validation(_args(tmp_path, max_model_failure_rate=0.05))

    assert summary["tolerance_reached"] is True
    assert summary["stopping_guard_passed"] is False
    assert summary["stopping_reason"] == "high_model_failure_rate"
    assert summary["model_failure_rate"] == 0.5


def test_failure_statuses_are_written_to_csv(monkeypatch, tmp_path) -> None:
    statuses = ["ok", "model_failed", "permutation_failed", "other_failed"]

    def fake_run_replicate(**kwargs):
        index = kwargs["replicate"]
        return _fake_replicate(index=index, status=statuses[index])

    monkeypatch.setattr(validation, "run_replicate", fake_run_replicate)
    summary = run_validation(_args(tmp_path, max_model_failure_rate=1.0))

    csv_text = Path(summary["csv_path"]).read_text()
    assert "status" in csv_text
    assert "model_failed" in csv_text
    assert "permutation_failed" in csv_text
    assert "RuntimeError" in csv_text
    assert summary["timing"]["fit_seconds_mean"] is not None


def _args(tmp_path, max_model_failure_rate: float) -> Namespace:
    return Namespace(
        scenario="recovery",
        n_subjects=2,
        trials_per_subject=2,
        n_sources=2,
        n_times=2,
        effect_size=1.0,
        random_intercept_sd=1.0,
        noise_sd=1.0,
        spatial_width=None,
        temporal_width=None,
        n_permutations=2,
        batch_size=4,
        min_reps=2,
        max_reps=4,
        tolerance=1.0,
        alpha=0.05,
        max_model_failure_rate=max_model_failure_rate,
        spatial_chunk_size=1,
        time_chunk_size=1,
        seed=1,
        out_dir=str(tmp_path),
        check_chunk_equivalence=False,
        no_progress=True,
    )


def _fake_replicate(index: int, status: str) -> ReplicateResult:
    is_ok = status == "ok"
    return ReplicateResult(
        replicate=index,
        scenario="recovery",
        seed=100 + index,
        n_observations=4,
        n_sources=2,
        n_times=2,
        n_permutations=2,
        status=status,
        failure_class=None if is_ok else "RuntimeError",
        failure_message=None if is_ok else f"{status} failure",
        model_converged=is_ok,
        convergence_rate=1.0 if is_ok else 0.0,
        n_lmm_features=4,
        n_lmm_failed=0 if is_ok else 4,
        n_failed_features=0 if is_ok else 4,
        failed_feature_fraction=0.0 if is_ok else 1.0,
        n_converged_features=4 if is_ok else 0,
        converged_feature_fraction=1.0 if is_ok else 0.0,
        n_boundary_warnings=0,
        significant_anywhere=is_ok,
        significant_in_effect_mask=is_ok,
        global_peak_in_effect_mask=is_ok,
        beta_sign_correct_at_peak=is_ok,
        metric_success=True if is_ok else None,
        min_corrected_p=0.01 if is_ok else None,
        min_corrected_p_in_effect_mask=0.01 if is_ok else None,
        peak_statistic=3.0 if is_ok else None,
        peak_location=1 if is_ok else None,
        peak_time=1 if is_ok else None,
        chunk_equivalence_checked=False,
        chunk_t_close=None,
        fit_seconds=0.1 if is_ok else None,
        permutation_seconds=0.2 if is_ok else None,
        total_seconds=0.3,
        seconds_per_permutation=0.1 if is_ok else None,
        peak_memory_mb=123.0,
    )
