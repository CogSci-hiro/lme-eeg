from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lmeeeg.core.design import DesignSpec


@dataclass(slots=True)
class ConvergenceSummary:
    """Summary of per-feature LMM convergence."""

    n_features: int
    n_converged: int
    n_failed: int
    convergence_rate: float
    n_boundary_warnings: int

    @classmethod
    def from_feature_table(cls, feature_table: pd.DataFrame) -> "ConvergenceSummary":
        """Create a convergence summary from a diagnostics table."""
        n_features = int(len(feature_table))
        n_converged = int(feature_table["converged"].sum()) if n_features > 0 else 0
        n_failed = n_features - n_converged
        n_boundary_warnings = int(feature_table["boundary_warning"].sum()) if n_features > 0 else 0
        convergence_rate = float(n_converged / n_features) if n_features > 0 else 0.0
        return cls(
            n_features=n_features,
            n_converged=n_converged,
            n_failed=n_failed,
            convergence_rate=convergence_rate,
            n_boundary_warnings=n_boundary_warnings,
        )


@dataclass(slots=True)
class FitResult:
    """Output of the fitting stage."""

    formula: str
    variable_types: dict[str, str]
    design_spec: DesignSpec
    fixed_effects_maps: dict[str, np.ndarray]
    random_effect_variance_map: np.ndarray
    residual_variance_map: np.ndarray
    fitted_random_effects: np.ndarray | None
    feature_diagnostics: pd.DataFrame
    convergence_summary: ConvergenceSummary
    marginal_eeg: np.ndarray | None
    ols_betas: dict[str, np.ndarray]
    ols_t_values: dict[str, np.ndarray]
    ols_residual_variance: np.ndarray
    backend_metadata: dict[str, Any]


@dataclass(slots=True)
class InferenceResult:
    """Output of the inference stage."""

    effect: str
    correction: str
    observed_statistic: np.ndarray
    corrected_p_values: np.ndarray
    null_distribution: np.ndarray
    clusters: Any
    cluster_p_values: Any
    backend_metadata: dict[str, Any]
