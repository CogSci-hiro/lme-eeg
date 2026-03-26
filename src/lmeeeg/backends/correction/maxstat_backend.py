from __future__ import annotations

import numpy as np

from lmeeeg.backends.correction.base import BaseCorrectionBackend
from lmeeeg.core.results import FitResult, InferenceResult


# ==============================
# Max-stat correction backend
# ==============================

class MaxStatCorrectionBackend(BaseCorrectionBackend):
    """Permutation max-statistic backend on OLS t maps."""

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
        """Run max-statistic correction.

        Notes
        -----
        This backend uses row shuffling of the design matrix as a simple MVP
        permutation scheme on marginalized data. It is intentionally explicit
        and easy to inspect.
        """
        del threshold, adjacency, tail
        rng = np.random.default_rng(seed)
        observed_t = fit_result.ols_t_values[effect]
        x_matrix = fit_result.design_spec.fixed_design_matrix
        y = fit_result.marginal_eeg
        n_observations, n_channels, n_times = y.shape
        y_2d = y.reshape(n_observations, n_channels * n_times)

        effect_index = fit_result.design_spec.fixed_column_names.index(effect)
        null_distribution = np.zeros(n_permutations, dtype=float)

        for permutation_index in range(n_permutations):
            permuted_indices = rng.permutation(n_observations)
            x_perm = x_matrix[permuted_indices, :]
            xtx_inv = np.linalg.inv(x_perm.T @ x_perm)
            beta = xtx_inv @ x_perm.T @ y_2d
            residuals = y_2d - x_perm @ beta
            residual_variance = np.sum(residuals ** 2, axis=0) / (n_observations - x_perm.shape[1])
            standard_error = np.sqrt(residual_variance * xtx_inv[effect_index, effect_index])
            t_values = beta[effect_index, :] / standard_error
            null_distribution[permutation_index] = np.max(np.abs(t_values))

        corrected_p_values = (1 + np.sum(null_distribution[:, None, None] >= np.abs(observed_t)[None, :, :], axis=0)) / (n_permutations + 1)

        return InferenceResult(
            effect=effect,
            correction="maxstat",
            observed_statistic=observed_t,
            corrected_p_values=corrected_p_values,
            null_distribution=null_distribution,
            clusters=None,
            cluster_p_values=None,
            backend_metadata={
                "backend": "maxstat",
                "n_permutations": n_permutations,
                "permutation_scheme": "row_shuffle_on_marginal_design",
            },
        )
