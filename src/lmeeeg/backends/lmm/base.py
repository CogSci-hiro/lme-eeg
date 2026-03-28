from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lmeeeg.core.design import DesignSpec


@dataclass(slots=True)
class LMMBackendResult:
    """Result produced by an LMM backend."""

    fixed_effects_maps: dict[str, np.ndarray]
    fitted_random_effects: np.ndarray | None
    marginal_eeg: np.ndarray | None
    random_effect_variance_map: np.ndarray
    residual_variance_map: np.ndarray
    feature_diagnostics: pd.DataFrame


class BaseLMMBackend(ABC):
    """Abstract base class for LMM backends."""

    @abstractmethod
    def fit_mass_univariate(
        self,
        eeg: np.ndarray,
        metadata: pd.DataFrame,
        design_spec: DesignSpec,
        show_progress: bool = True,
        store_fitted_random_effects: bool = False,
        store_marginal_eeg: bool = True,
        output_dtype: np.dtype | None = None,
    ) -> LMMBackendResult:
        """Fit the backend over all channel × timepoint features."""
