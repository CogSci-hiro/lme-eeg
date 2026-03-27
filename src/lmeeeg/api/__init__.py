"""Public API layer."""

from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
from lmeeeg.api.infer import PermutationConfig, permute_fixed_effect
from lmeeeg.api.simulate import simulate_erp_random_intercept_dataset, simulate_random_intercept_dataset

__all__ = [
    "FitConfig",
    "PermutationConfig",
    "fit_lmm_mass_univariate",
    "permute_fixed_effect",
    "simulate_erp_random_intercept_dataset",
    "simulate_random_intercept_dataset",
]
