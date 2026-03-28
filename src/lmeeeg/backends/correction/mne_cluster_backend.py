from __future__ import annotations

import numpy as np

from lmeeeg.backends.correction._regression import (
    cluster_outputs_to_masks,
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
# MNE cluster correction backend
# ==============================

class MNEClusterCorrectionBackend(BaseCorrectionBackend):
    """Cluster-based permutation correction using MNE-Python."""

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
    ) -> InferenceResult:
        """Run cluster-based permutation correction with MNE.

        The statistic is the selected fixed-effect t map on marginalized EEG.
        Permutations are restricted within the grouping factor used by the
        random-intercept model to preserve exchangeability.
        """
        emit_info(verbose, "Running cluster correction for {0} with {1} permutations.", effect, n_permutations)
        configure_mne_runtime()
        try:
            from mne.stats.cluster_level import _find_clusters, _setup_adjacency
        except Exception as error:  # pragma: no cover
            raise ImportError("MNE-Python is required for cluster correction.") from error

        prepared = prepare_effect_regression(fit_result=fit_result, effect=effect)
        y_residualized = prepared["y_residualized"]
        effect_residualized = prepared["effect_residualized"]
        effect_sum_squares = float(prepared["effect_sum_squares"])
        degrees_of_freedom = int(prepared["degrees_of_freedom"])
        group_codes = prepared["group_codes"]
        n_channels = int(prepared["n_channels"])
        n_times = int(prepared["n_times"])
        observed_t = compute_effect_t_statistics(
            y_residualized=y_residualized,
            effect_residualized=effect_residualized,
            effect_sum_squares=effect_sum_squares,
            degrees_of_freedom=degrees_of_freedom,
        ).reshape(n_channels, n_times)
        cluster_threshold = threshold if threshold is not None else 2.0
        sample_shape = (n_times, n_channels)
        prepared_adjacency = adjacency
        if adjacency is not None:
            prepared_adjacency = _setup_adjacency(
                adjacency=adjacency,
                n_tests=n_channels * n_times,
                n_times=n_times,
            )

        cluster_input = observed_t.T
        raw_clusters, cluster_stats = _find_clusters(
            cluster_input if prepared_adjacency is None else cluster_input.ravel(),
            threshold=cluster_threshold,
            tail=tail,
            adjacency=prepared_adjacency,
        )
        cluster_masks = cluster_outputs_to_masks(raw_clusters, sample_shape)

        rng = np.random.default_rng(seed)
        null_distribution = np.zeros(n_permutations, dtype=float)
        with progress_context(verbose) as active_progress:
            task_id = None
            if active_progress is not None:
                task_id = active_progress.add_task(
                    f"Cluster permutations for {effect}",
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
                ).reshape(n_channels, n_times)
                _, permuted_cluster_stats = _find_clusters(
                    permuted_t.T if prepared_adjacency is None else permuted_t.T.ravel(),
                    threshold=cluster_threshold,
                    tail=tail,
                    adjacency=prepared_adjacency,
                )
                null_distribution[permutation_index] = (
                    float(np.max(np.abs(permuted_cluster_stats))) if len(permuted_cluster_stats) else 0.0
                )
                if active_progress is not None and task_id is not None:
                    active_progress.advance(task_id)

        cluster_p_values = np.asarray(
            [
                (1 + np.sum(null_distribution >= abs(cluster_stat))) / (n_permutations + 1)
                for cluster_stat in cluster_stats
            ],
            dtype=float,
        )
        corrected_p_values = np.ones_like(observed_t, dtype=float)
        for cluster_mask, cluster_p_value in zip(cluster_masks, cluster_p_values):
            corrected_p_values[cluster_mask.T] = np.minimum(corrected_p_values[cluster_mask.T], cluster_p_value)

        emit_info(
            verbose,
            "Finished cluster correction for {0}. Found {1} clusters.",
            effect,
            len(cluster_masks),
        )

        return InferenceResult(
            effect=effect,
            correction="cluster",
            observed_statistic=observed_t,
            corrected_p_values=corrected_p_values,
            null_distribution=np.asarray(null_distribution),
            clusters=cluster_masks,
            cluster_p_values=cluster_p_values,
            backend_metadata={
                "backend": "mne_cluster",
                "n_permutations": n_permutations,
                "threshold": cluster_threshold,
                "permutation_scheme": "within_group_row_shuffle",
                "statistic": "partial_effect_t",
                "verbose": verbose,
            },
        )
