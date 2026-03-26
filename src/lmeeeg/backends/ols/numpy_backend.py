from __future__ import annotations

import numpy as np

from lmeeeg.backends.ols.base import BaseOLSBackend, OLSBackendResult


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
    ) -> OLSBackendResult:
        """Fit OLS for all channel × timepoint features.

        Parameters
        ----------
        eeg : np.ndarray
            Marginal EEG with shape `(n_observations, n_channels, n_times)`.
        design_matrix : np.ndarray
            Fixed-effects design matrix.
        column_names : list[str]
            Fixed-effect column names.

        Returns
        -------
        OLSBackendResult
            OLS result object.
        """
        n_observations, n_channels, n_times = eeg.shape
        y_2d = eeg.reshape(n_observations, n_channels * n_times)
        x_matrix = design_matrix
        xtx_inv = np.linalg.inv(x_matrix.T @ x_matrix)
        beta_matrix = xtx_inv @ x_matrix.T @ y_2d
        residuals = y_2d - x_matrix @ beta_matrix
        residual_dof = n_observations - x_matrix.shape[1]
        residual_variance = np.sum(residuals ** 2, axis=0) / residual_dof

        beta_maps: dict[str, np.ndarray] = {}
        t_value_maps: dict[str, np.ndarray] = {}

        for column_index, column_name in enumerate(column_names):
            beta_vector = beta_matrix[column_index, :]
            standard_error_vector = np.sqrt(residual_variance * xtx_inv[column_index, column_index])
            t_vector = beta_vector / standard_error_vector
            beta_maps[column_name] = beta_vector.reshape(n_channels, n_times)
            t_value_maps[column_name] = t_vector.reshape(n_channels, n_times)

        return OLSBackendResult(
            beta_maps=beta_maps,
            t_value_maps=t_value_maps,
            residual_variance_map=residual_variance.reshape(n_channels, n_times),
        )
