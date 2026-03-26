from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class OLSBackendResult:
    """Result produced by an OLS backend."""

    beta_maps: dict[str, np.ndarray]
    t_value_maps: dict[str, np.ndarray]
    residual_variance_map: np.ndarray


class BaseOLSBackend(ABC):
    """Abstract base class for OLS backends."""

    @abstractmethod
    def fit_mass_univariate(
        self,
        eeg: np.ndarray,
        design_matrix: np.ndarray,
        column_names: list[str],
    ) -> OLSBackendResult:
        """Fit OLS mass-univariate models."""
