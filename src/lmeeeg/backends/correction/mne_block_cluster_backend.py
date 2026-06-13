import numpy as np

from lmeeeg.backends.correction._regression import (
    cluster_outputs_to_masks,
    compute_block_f_statistics,
    configure_mne_runtime,
    emit_info,
    make_permutation_rng,
    permute_within_groups,
    prepare_block_regression,
    progress_context,
    validate_f_statistics,
)
from lmeeeg.backends.correction.base import BaseCorrectionBackend
from lmeeeg.core.results import FitResult, InferenceResult


class MNEBlockClusterCorrectionBackend(BaseCorrectionBackend):
    """Cluster correction for nested fixed-effect blocks using partial F maps."""

    def run(
        self,
        fit_result: FitResult,
        reduced_formula: str | list[str],
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
        """Run one-tailed cluster correction on a nested-block partial F statistic."""
        del spatial_chunk_size, time_chunk_size
        if tail != 1:
            raise ValueError("Nested-block F tests are one-tailed; use `tail=1`.")
        if store_null_maps:
            raise ValueError("Cluster correction stores only max cluster statistics, not full null maps.")
        emit_info(verbose, "Running nested-block cluster correction with {0} permutations.", n_permutations)
        configure_mne_runtime()
        try:
            from mne.stats.cluster_level import _find_clusters, _setup_adjacency
        except Exception as error:  # pragma: no cover
            raise ImportError("MNE-Python is required for cluster correction.") from error

        prepared = prepare_block_regression(fit_result=fit_result, reduced_formula=reduced_formula)
        y_residualized = prepared["y_residualized"]
        y_sum_squares = prepared["y_sum_squares"]
        block_projection = prepared["block_projection"]
        block_rank = int(prepared["block_rank"])
        degrees_of_freedom = int(prepared["degrees_of_freedom"])
        group_codes = prepared["group_codes"]
        n_locations = int(prepared["n_locations"])
        n_times = int(prepared["n_times"])
        effect_label = " | ".join(prepared["block_column_names"])

        observed_f = compute_block_f_statistics(
            y_residualized=y_residualized,
            block_projection=block_projection,
            y_sum_squares=y_sum_squares,
            block_rank=block_rank,
            degrees_of_freedom=degrees_of_freedom,
        ).reshape(n_locations, n_times)
        validate_f_statistics(observed_f, "observed data")
        cluster_threshold = threshold if threshold is not None else 4.0
        sample_shape = (n_times, n_locations)
        prepared_adjacency = adjacency
        if adjacency is not None:
            prepared_adjacency = _setup_adjacency(
                adjacency=adjacency,
                n_tests=n_locations * n_times,
                n_times=n_times,
            )

        cluster_input = observed_f.T
        raw_clusters, cluster_stats = _find_clusters(
            cluster_input if prepared_adjacency is None else cluster_input.ravel(),
            threshold=cluster_threshold,
            tail=1,
            adjacency=prepared_adjacency,
        )
        cluster_masks = cluster_outputs_to_masks(raw_clusters, sample_shape)

        rng = make_permutation_rng(seed)
        null_distribution = np.zeros(n_permutations, dtype=float)
        with progress_context(verbose) as active_progress:
            task_id = None
            if active_progress is not None:
                task_id = active_progress.add_task(
                    "Nested-block cluster permutations",
                    total=n_permutations,
                )
            for permutation_index in range(n_permutations):
                y_permuted = permute_within_groups(
                    y_residualized=y_residualized,
                    group_codes=group_codes,
                    rng=rng,
                )
                permuted_f = compute_block_f_statistics(
                    y_residualized=y_permuted,
                    block_projection=block_projection,
                    y_sum_squares=y_sum_squares,
                    block_rank=block_rank,
                    degrees_of_freedom=degrees_of_freedom,
                ).reshape(n_locations, n_times)
                validate_f_statistics(permuted_f, f"permutation {permutation_index}")
                _, permuted_cluster_stats = _find_clusters(
                    permuted_f.T if prepared_adjacency is None else permuted_f.T.ravel(),
                    threshold=cluster_threshold,
                    tail=1,
                    adjacency=prepared_adjacency,
                )
                null_distribution[permutation_index] = (
                    float(np.max(permuted_cluster_stats)) if len(permuted_cluster_stats) else 0.0
                )
                if active_progress is not None and task_id is not None:
                    active_progress.advance(task_id)

        cluster_p_values = np.asarray(
            [
                (1 + np.sum(null_distribution >= cluster_stat)) / (n_permutations + 1)
                for cluster_stat in cluster_stats
            ],
            dtype=float,
        )
        corrected_p_values = np.ones_like(observed_f, dtype=float)
        for cluster_mask, cluster_p_value in zip(cluster_masks, cluster_p_values):
            corrected_p_values[cluster_mask.T] = np.minimum(corrected_p_values[cluster_mask.T], cluster_p_value)

        emit_info(
            verbose,
            "Finished nested-block cluster correction. Found {0} clusters.",
            len(cluster_masks),
        )

        return InferenceResult(
            effect=effect_label,
            correction="cluster",
            observed_statistic=observed_f,
            corrected_p_values=corrected_p_values,
            null_distribution=np.asarray(null_distribution),
            clusters=cluster_masks,
            cluster_p_values=cluster_p_values,
            backend_metadata={
                "backend": "mne_block_cluster",
                "n_permutations": n_permutations,
                "threshold": cluster_threshold,
                "permutation_scheme": "within_group_row_shuffle",
                "statistic": "nested_block_partial_f",
                "verbose": verbose,
                "space": fit_result.space,
                "n_locations": fit_result.n_locations,
                "n_times": fit_result.n_times,
                "store_null_maps": False,
                "reduced_formula": prepared["normalized_reduced_formula"],
                "reduced_columns": prepared["reduced_column_names"],
                "block_columns": prepared["block_column_names"],
                "block_rank": block_rank,
                "degrees_of_freedom": degrees_of_freedom,
            },
        )
