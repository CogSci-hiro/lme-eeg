from __future__ import annotations

from dataclasses import dataclass

from lmeeeg.backends.correction.maxstat_backend import MaxStatCorrectionBackend
from lmeeeg.backends.correction.mne_cluster_backend import MNEClusterCorrectionBackend
from lmeeeg.backends.correction.mne_tfce_backend import MNETFCorrectionBackend
from lmeeeg.core.results import FitResult, InferenceResult


@dataclass(slots=True)
class PermutationConfig:
    """Configuration for permutation inference."""

    n_permutations: int = 1000
    seed: int = 0
    tail: int = 0


# ==============================
# Public inference entry point
# ==============================

def permute_fixed_effect(
    fit_result: FitResult,
    effect: str,
    correction: str = "cluster",
    n_permutations: int = 1000,
    seed: int = 0,
    tail: int = 0,
    threshold: float | dict[str, float] | None = None,
    adjacency=None,
) -> InferenceResult:
    """Run permutation-based inference for one fixed effect.

    Parameters
    ----------
    fit_result : FitResult
        Result returned by :func:`fit_lmm_mass_univariate`.
    effect : str
        Exact fixed-effect column name to test.
    correction : str
        Correction backend: ``maxstat``, ``cluster``, or ``tfce``.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.
    tail : int
        Tail for MNE-compatible permutation code. Use 0 for two-sided,
        1 for positive, -1 for negative.
    threshold : float | dict[str, float] | None
        Cluster threshold or TFCE threshold dictionary.
    adjacency : Any
        Optional adjacency matrix passed through to MNE correction backends.

    Returns
    -------
    InferenceResult
        Corrected inference output.

    Usage example
    -------------
        inference = permute_fixed_effect(
            fit_result=fit_result,
            effect="condition[T.B]",
            correction="tfce",
            n_permutations=500,
            seed=1,
        )
    """
    if effect not in fit_result.design_spec.fixed_column_names:
        available = ", ".join(fit_result.design_spec.fixed_column_names)
        raise ValueError(f"Unknown effect '{effect}'. Available fixed effects: {available}")

    if correction == "maxstat":
        backend = MaxStatCorrectionBackend()
    elif correction == "cluster":
        backend = MNEClusterCorrectionBackend()
    elif correction == "tfce":
        backend = MNETFCorrectionBackend()
    else:
        raise ValueError(f"Unsupported correction backend: {correction}")

    return backend.run(
        fit_result=fit_result,
        effect=effect,
        n_permutations=n_permutations,
        seed=seed,
        tail=tail,
        threshold=threshold,
        adjacency=adjacency,
    )
