import warnings

import numpy as np

from lmeeeg.backends.correction._regression import (
    compute_effect_t_statistics,
    emit_info,
    configure_mne_runtime,
    permute_within_groups,
    progress_context,
    prepare_effect_regression,
)
from lmeeeg.backends.correction.base import BaseCorrectionBackend
from lmeeeg.core.results import FitResult, InferenceResult


# ==============================
# MNE TFCE correction backend
# ==============================

class MNETFCorrectionBackend(BaseCorrectionBackend):
    """TFCE correction backend using MNE-Python."""

    def run(
        self,
        fit_result: FitResult,
        effect: str,
        n_permutations: int,
        seed: int,
        tail: int,
        threshold: float | dict[str, float] | None,
        adjacency,
        verbose: bool | str | int | None = "info",
        spatial_chunk_size: int | None = None,
        time_chunk_size: int | None = None,
        store_null_maps: bool = False,
    ) -> InferenceResult:
        """Run TFCE permutation correction with MNE."""
        del spatial_chunk_size, time_chunk_size
        if store_null_maps:
            raise ValueError("TFCE correction stores only max TFCE statistics, not full null maps.")
        job_size = fit_result.n_locations * fit_result.n_times * n_permutations
        if job_size >= 100_000_000:
            warnings.warn(
                "Large TFCE job requested. LmeEEG stores only one max TFCE value per permutation, "
                "but MNE TFCE still processes full statistic maps internally; consider parcellation, "
                "a shorter time window, fewer development permutations, or a coarser TFCE step.",
                RuntimeWarning,
                stacklevel=2,
            )
        emit_info(verbose, "Running TFCE correction for {0} with {1} permutations.", effect, n_permutations)
        configure_mne_runtime()
        try:
            from mne.stats.cluster_level import _find_clusters, _setup_adjacency
        except Exception as error:  # pragma: no cover
            raise ImportError("MNE-Python is required for TFCE correction.") from error

        prepared = prepare_effect_regression(fit_result=fit_result, effect=effect)
        y_residualized = prepared["y_residualized"]
        effect_residualized = prepared["effect_residualized"]
        effect_sum_squares = float(prepared["effect_sum_squares"])
        degrees_of_freedom = int(prepared["degrees_of_freedom"])
        group_codes = prepared["group_codes"]
        n_locations = int(prepared["n_locations"])
        n_times = int(prepared["n_times"])
        tfce_threshold = threshold if threshold is not None else {"start": 0.0, "step": 0.2}
        prepared_adjacency = adjacency
        if adjacency is not None:
            prepared_adjacency = _setup_adjacency(
                adjacency=adjacency,
                n_tests=n_locations * n_times,
                n_times=n_times,
            )

        observed_t = compute_effect_t_statistics(
            y_residualized=y_residualized,
            effect_residualized=effect_residualized,
            effect_sum_squares=effect_sum_squares,
            degrees_of_freedom=degrees_of_freedom,
        ).reshape(n_locations, n_times)
        _, observed_tfce = _find_clusters(
            observed_t.T if prepared_adjacency is None else observed_t.T.ravel(),
            threshold=tfce_threshold,
            tail=tail,
            adjacency=prepared_adjacency,
        )
        observed_tfce = np.asarray(observed_tfce, dtype=float).reshape(n_times, n_locations).T

        rng = np.random.default_rng(seed)
        null_distribution = np.zeros(n_permutations, dtype=float)
        with progress_context(verbose) as active_progress:
            task_id = None
            if active_progress is not None:
                task_id = active_progress.add_task(
                    f"TFCE permutations for {effect}",
                    total=n_permutations,
                )
            for permutation_index in range(n_permutations):
                y_permuted = permute_within_groups(
                    y_residualized=y_residualized,
                    group_codes=group_codes,
                    rng=rng,
                )
                permuted_t = compute_effect_t_statistics(
                    y_residualized=y_permuted,
                    effect_residualized=effect_residualized,
                    effect_sum_squares=effect_sum_squares,
                    degrees_of_freedom=degrees_of_freedom,
                ).reshape(n_locations, n_times)
                _, permuted_tfce = _find_clusters(
                    permuted_t.T if prepared_adjacency is None else permuted_t.T.ravel(),
                    threshold=tfce_threshold,
                    tail=tail,
                    adjacency=prepared_adjacency,
                )
                null_distribution[permutation_index] = float(np.max(np.abs(permuted_tfce)))
                if active_progress is not None and task_id is not None:
                    active_progress.advance(task_id)

        corrected_p_values = (
            1
            + np.sum(
                null_distribution[:, None, None] >= np.abs(observed_tfce)[None, :, :],
                axis=0,
            )
        ) / (n_permutations + 1)

        emit_info(verbose, "Finished TFCE correction for {0}.", effect)

        return InferenceResult(
            effect=effect,
            correction="tfce",
            observed_statistic=observed_tfce,
            corrected_p_values=corrected_p_values,
            null_distribution=np.asarray(null_distribution),
            clusters=None,
            cluster_p_values=None,
            backend_metadata={
                "backend": "mne_tfce",
                "n_permutations": n_permutations,
                "threshold": tfce_threshold,
                "permutation_scheme": "within_group_row_shuffle",
                "statistic": "tfce_on_partial_effect_t",
                "verbose": verbose,
                "space": fit_result.space,
                "n_locations": fit_result.n_locations,
                "n_times": fit_result.n_times,
                "store_null_maps": False,
                "streams_null_maps": False,
            },
        )
