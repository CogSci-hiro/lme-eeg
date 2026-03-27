from __future__ import annotations

import os

import numpy as np

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
    ) -> InferenceResult:
        """Run cluster-based permutation correction with MNE.

        The test is performed on observation-level marginalized data using a
        simple two-model Freedman-Lane style residualization for the selected
        effect. This keeps the correction layer separated from the LMM layer.
        """
        # Some environments expose MNE through a Numba caching path that is not
        # available at runtime. Disabling JIT here keeps the optional backend
        # usable without affecting the public API.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
        try:
            from mne.stats import permutation_cluster_1samp_test
        except Exception as error:  # pragma: no cover
            raise ImportError("MNE-Python is required for cluster correction.") from error

        x_matrix = fit_result.design_spec.fixed_design_matrix
        column_names = fit_result.design_spec.fixed_column_names
        effect_index = column_names.index(effect)
        reduced_columns = [index for index in range(len(column_names)) if index != effect_index]

        y = fit_result.marginal_eeg
        n_observations, n_channels, n_times = y.shape
        y_2d = y.reshape(n_observations, n_channels * n_times)

        if reduced_columns:
            x_reduced = x_matrix[:, reduced_columns]
            beta_reduced = np.linalg.pinv(x_reduced) @ y_2d
            residuals = y_2d - x_reduced @ beta_reduced
        else:
            residuals = y_2d.copy()

        residuals_3d = residuals.reshape(n_observations, n_channels, n_times)
        data_for_mne = np.transpose(residuals_3d, (0, 2, 1))
        cluster_threshold = threshold if threshold is not None else 2.0

        observed_t, clusters, cluster_p_values, null_distribution = permutation_cluster_1samp_test(
            X=data_for_mne,
            threshold=cluster_threshold,
            n_permutations=n_permutations,
            tail=tail,
            adjacency=adjacency,
            out_type="mask",
            seed=seed,
            verbose=False,
        )
        observed_t = observed_t.T
        corrected_p_values = np.ones_like(observed_t, dtype=float)
        for cluster_mask, cluster_p_value in zip(clusters, cluster_p_values):
            corrected_p_values[cluster_mask.T] = np.minimum(corrected_p_values[cluster_mask.T], cluster_p_value)

        return InferenceResult(
            effect=effect,
            correction="cluster",
            observed_statistic=observed_t,
            corrected_p_values=corrected_p_values,
            null_distribution=np.asarray(null_distribution),
            clusters=clusters,
            cluster_p_values=np.asarray(cluster_p_values),
            backend_metadata={
                "backend": "mne_cluster",
                "n_permutations": n_permutations,
                "threshold": cluster_threshold,
            },
        )
