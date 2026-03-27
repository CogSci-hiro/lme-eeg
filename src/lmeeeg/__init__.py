"""Top-level package for LmeEEG."""

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect
from lmeeeg.api.simulate import simulate_erp_random_intercept_dataset, simulate_random_intercept_dataset

__all__ = [
    "fit_lmm_mass_univariate",
    "permute_fixed_effect",
    "simulate_erp_random_intercept_dataset",
    "simulate_random_intercept_dataset",
]
