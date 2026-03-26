"""Top-level package for LmeEEG."""

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect

__all__ = [
    "fit_lmm_mass_univariate",
    "permute_fixed_effect",
]
