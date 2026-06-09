import numpy as np

from lmeeeg.backends.ols.base import BaseOLSBackend, OLSBackendResult
from lmeeeg.core.space import iter_spatiotemporal_chunks


# ==============================
# NumPy OLS backend
# ==============================

class NumPyOLSBackend(BaseOLSBackend):
    """Fast OLS backend using closed-form matrix algebra."""

    def fit_mass_univariate(
        self,
        eeg: np.ndarray,
        design_matrix: np.ndarray,
        column_names: list[str],
        spatial_chunk_size: int | None = None,
        time_chunk_size: int | None = None,
    ) -> OLSBackendResult:
        """Fit OLS for all location × timepoint features.

        Parameters
        ----------
        eeg : np.ndarray
            Marginal EEG/source data with shape `(n_observations, n_locations, n_times)`.
        design_matrix : np.ndarray
            Fixed-effects design matrix.
        column_names : list[str]
            Fixed-effect column names.

        Returns
        -------
        OLSBackendResult
            OLS result object.
        """
        n_observations, n_locations, n_times = eeg.shape
        x_matrix = np.asarray(design_matrix, dtype=np.float64)
        xtx_inv = np.linalg.inv(x_matrix.T @ x_matrix)
        residual_dof = n_observations - x_matrix.shape[1]

        beta_maps: dict[str, np.ndarray] = {}
        t_value_maps: dict[str, np.ndarray] = {}
        for column_index, column_name in enumerate(column_names):
            beta_maps[column_name] = np.full((n_locations, n_times), np.nan, dtype=float)
            t_value_maps[column_name] = np.full((n_locations, n_times), np.nan, dtype=float)
        residual_variance_map = np.full((n_locations, n_times), np.nan, dtype=float)

        for location_slice, time_slice in iter_spatiotemporal_chunks(
            n_locations=n_locations,
            n_times=n_times,
            spatial_chunk_size=spatial_chunk_size,
            time_chunk_size=time_chunk_size,
        ):
            y_chunk = np.asanyarray(eeg[:, location_slice, time_slice])
            chunk_shape = y_chunk.shape[1:]
            y_2d = np.asarray(y_chunk, dtype=np.float64).reshape(n_observations, -1)
            beta_matrix = xtx_inv @ x_matrix.T @ y_2d
            residuals = y_2d - x_matrix @ beta_matrix
            residual_variance = np.sum(residuals ** 2, axis=0) / residual_dof

            residual_variance_map[location_slice, time_slice] = residual_variance.reshape(chunk_shape)

            for column_index, column_name in enumerate(column_names):
                beta_vector = beta_matrix[column_index, :]
                standard_error_vector = np.sqrt(residual_variance * xtx_inv[column_index, column_index])
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_vector = np.divide(
                        beta_vector,
                        standard_error_vector,
                        out=np.zeros_like(beta_vector),
                        where=standard_error_vector > 0,
                    )
                beta_maps[column_name][location_slice, time_slice] = beta_vector.reshape(chunk_shape)
                t_value_maps[column_name][location_slice, time_slice] = t_vector.reshape(chunk_shape)

        return OLSBackendResult(
            beta_maps=beta_maps,
            t_value_maps=t_value_maps,
            residual_variance_map=residual_variance_map,
        )
