import numpy as np

from lmeeeg.backends.correction._regression import make_permutation_rng
from lmeeeg.backends.correction.base import BaseCorrectionBackend
from lmeeeg.core.results import FitResult, InferenceResult
from lmeeeg.core.space import iter_spatiotemporal_chunks


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
        verbose: bool | str | int | None = "info",
        spatial_chunk_size: int | None = None,
        time_chunk_size: int | None = None,
        store_null_maps: bool = False,
    ) -> InferenceResult:
        """Run max-statistic correction.

        Notes
        -----
        This backend uses row shuffling of the design matrix as a simple MVP
        permutation scheme on marginalized data. It is intentionally explicit
        and easy to inspect.
        """
        del threshold, adjacency, tail, verbose
        if store_null_maps:
            raise ValueError("Max-stat correction stores only the max statistic per permutation.")
        rng = make_permutation_rng(seed)
        observed_t = fit_result.ols_t_values[effect]
        x_matrix = np.asarray(fit_result.design_spec.fixed_design_matrix, dtype=np.float64)
        if fit_result.marginal_eeg is None:
            raise ValueError(
                "Permutation inference requires `fit_result.marginal_eeg`. "
                "Run `fit_lmm_mass_univariate(..., config=FitConfig(store_marginal_eeg=True))`."
            )
        y = fit_result.marginal_eeg
        n_observations, n_locations, n_times = y.shape

        effect_index = fit_result.design_spec.fixed_column_names.index(effect)
        null_distribution = np.zeros(n_permutations, dtype=float)

        for permutation_index in range(n_permutations):
            permuted_indices = rng.permutation(n_observations)
            x_perm = x_matrix[permuted_indices, :]
            xtx_inv = np.linalg.inv(x_perm.T @ x_perm)
            max_statistic = 0.0
            for location_slice, time_slice in iter_spatiotemporal_chunks(
                n_locations=n_locations,
                n_times=n_times,
                spatial_chunk_size=spatial_chunk_size,
                time_chunk_size=time_chunk_size,
            ):
                y_chunk = np.asanyarray(y[:, location_slice, time_slice])
                y_2d = np.asarray(y_chunk, dtype=np.float64).reshape(n_observations, -1)
                beta = xtx_inv @ x_perm.T @ y_2d
                residuals = y_2d - x_perm @ beta
                residual_variance = np.sum(residuals ** 2, axis=0) / (n_observations - x_perm.shape[1])
                standard_error = np.sqrt(residual_variance * xtx_inv[effect_index, effect_index])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_values = np.divide(
                        beta[effect_index, :],
                        standard_error,
                        out=np.zeros_like(standard_error),
                        where=standard_error > 0,
                    )
                if t_values.size:
                    max_statistic = max(max_statistic, float(np.max(np.abs(t_values))))
            null_distribution[permutation_index] = max_statistic

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
                "space": fit_result.space,
                "n_locations": fit_result.n_locations,
                "n_times": fit_result.n_times,
                "spatial_chunk_size": spatial_chunk_size,
                "time_chunk_size": time_chunk_size,
                "store_null_maps": False,
            },
        )
