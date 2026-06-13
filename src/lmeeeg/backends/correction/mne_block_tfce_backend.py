import warnings

import numpy as np

from lmeeeg.backends.correction._regression import (
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


class MNEBlockTFCECorrectionBackend(BaseCorrectionBackend):
    """TFCE correction for nested fixed-effect blocks using partial F maps."""

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
        tfce_h_power: float = 2.0,
        tfce_e_power: float = 0.5,
    ) -> InferenceResult:
        """Run one-tailed TFCE correction on a nested-block partial F statistic."""
        del spatial_chunk_size, time_chunk_size
        if tail != 1:
            raise ValueError("Nested-block F tests are one-tailed; use `tail=1`.")
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
        emit_info(verbose, "Running nested-block TFCE correction with {0} permutations.", n_permutations)
        configure_mne_runtime()
        try:
            from mne.stats.cluster_level import _find_clusters, _setup_adjacency
        except Exception as error:  # pragma: no cover
            raise ImportError("MNE-Python is required for TFCE correction.") from error

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

        # MNE's default TFCE powers were tuned around signed t maps; they are
        # exposed here because they are not validated defaults for F maps.
        if threshold is None:
            tfce_threshold = {
                "start": 0.0,
                "step": 0.2,
                "h_power": tfce_h_power,
                "e_power": tfce_e_power,
            }
        elif isinstance(threshold, dict):
            tfce_threshold = {
                "h_power": tfce_h_power,
                "e_power": tfce_e_power,
                **threshold,
            }
        else:
            raise ValueError("Nested-block TFCE requires a threshold dictionary or None.")

        prepared_adjacency = adjacency
        if adjacency is not None:
            prepared_adjacency = _setup_adjacency(
                adjacency=adjacency,
                n_tests=n_locations * n_times,
                n_times=n_times,
            )

        observed_f = compute_block_f_statistics(
            y_residualized=y_residualized,
            block_projection=block_projection,
            y_sum_squares=y_sum_squares,
            block_rank=block_rank,
            degrees_of_freedom=degrees_of_freedom,
        ).reshape(n_locations, n_times)
        validate_f_statistics(observed_f, "observed data")
        _, observed_tfce = _find_clusters(
            observed_f.T if prepared_adjacency is None else observed_f.T.ravel(),
            threshold=tfce_threshold,
            tail=1,
            adjacency=prepared_adjacency,
        )
        observed_tfce = np.asarray(observed_tfce, dtype=float).reshape(n_times, n_locations).T

        rng = make_permutation_rng(seed)
        null_distribution = np.zeros(n_permutations, dtype=float)
        with progress_context(verbose) as active_progress:
            task_id = None
            if active_progress is not None:
                task_id = active_progress.add_task(
                    "Nested-block TFCE permutations",
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
                _, permuted_tfce = _find_clusters(
                    permuted_f.T if prepared_adjacency is None else permuted_f.T.ravel(),
                    threshold=tfce_threshold,
                    tail=1,
                    adjacency=prepared_adjacency,
                )
                null_distribution[permutation_index] = float(np.max(permuted_tfce))
                if active_progress is not None and task_id is not None:
                    active_progress.advance(task_id)

        corrected_p_values = (
            1
            + np.sum(
                null_distribution[:, None, None] >= observed_tfce[None, :, :],
                axis=0,
            )
        ) / (n_permutations + 1)

        emit_info(verbose, "Finished nested-block TFCE correction.")

        return InferenceResult(
            effect=effect_label,
            correction="tfce",
            observed_statistic=observed_tfce,
            corrected_p_values=corrected_p_values,
            null_distribution=np.asarray(null_distribution),
            clusters=None,
            cluster_p_values=None,
            backend_metadata={
                "backend": "mne_block_tfce",
                "n_permutations": n_permutations,
                "threshold": tfce_threshold,
                "permutation_scheme": "within_group_row_shuffle",
                "statistic": "tfce_on_nested_block_partial_f",
                "verbose": verbose,
                "space": fit_result.space,
                "n_locations": fit_result.n_locations,
                "n_times": fit_result.n_times,
                "store_null_maps": False,
                "streams_null_maps": False,
                "reduced_formula": prepared["normalized_reduced_formula"],
                "reduced_columns": prepared["reduced_column_names"],
                "block_columns": prepared["block_column_names"],
                "block_rank": block_rank,
                "degrees_of_freedom": degrees_of_freedom,
                "tfce_h_power": tfce_h_power,
                "tfce_e_power": tfce_e_power,
            },
        )
