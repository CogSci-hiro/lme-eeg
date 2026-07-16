from dataclasses import dataclass
from typing import Any

from lmeeeg.backends.correction.mne_block_cluster_backend import MNEBlockClusterCorrectionBackend
from lmeeeg.backends.correction.mne_block_tfce_backend import MNEBlockTFCECorrectionBackend
from lmeeeg.backends.correction.maxstat_backend import MaxStatCorrectionBackend
from lmeeeg.backends.correction.mne_cluster_backend import MNEClusterCorrectionBackend
from lmeeeg.backends.correction.mne_tfce_backend import MNETFCorrectionBackend
from lmeeeg.core.results import FitResult, InferenceResult
from lmeeeg.core.space import SpaceKind


@dataclass(slots=True)
class PermutationConfig:
    """Configuration for permutation inference."""

    n_permutations: int = 1000
    seed: int = 0
    tail: int = 0
    verbose: bool | str | int | None = "info"
    space: SpaceKind | None = None
    adjacency: Any | None = None
    spatial_chunk_size: int | None = None
    time_chunk_size: int | None = None
    store_null_maps: bool = False


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
    verbose: bool | str | int | None = "info",
    spatial_chunk_size: int | None = None,
    time_chunk_size: int | None = None,
    store_null_maps: bool = False,
    permutation_scheme: str | None = None,
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
    verbose : bool | str | int | None
        Verbosity forwarded to MNE-based correction backends. Defaults to
        ``"info"`` so cluster and TFCE inference report progress. Ignored by
        the max-stat backend.
    spatial_chunk_size : int | None
        Optional chunk size over locations for backends that support streaming.
    time_chunk_size : int | None
        Optional chunk size over time for backends that support streaming.
    store_null_maps : bool
        Reserved for backends that can expose full null maps. Defaults to
        ``False``; current correction backends store compact max-statistic null
        distributions only.
    permutation_scheme : {"free", "within_subject"} | None
        Optional effect-vector permutation scheme for maxstat, cluster, and
        TFCE correction. ``None`` uses each backend's default.

    Returns
    -------
    InferenceResult
        Corrected inference output.
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
        verbose=verbose,
        spatial_chunk_size=spatial_chunk_size,
        time_chunk_size=time_chunk_size,
        store_null_maps=store_null_maps,
        **({} if permutation_scheme is None else {"permutation_scheme": permutation_scheme}),
    )


def permute_fixed_block(
    fit_result: FitResult,
    reduced_formula: str | list[str],
    correction: str = "cluster",
    n_permutations: int = 1000,
    seed: int = 0,
    tail: int = 1,
    threshold: float | dict[str, float] | None = None,
    adjacency=None,
    verbose: bool | str | int | None = "info",
    spatial_chunk_size: int | None = None,
    time_chunk_size: int | None = None,
    store_null_maps: bool = False,
    tfce_h_power: float = 2.0,
    tfce_e_power: float = 0.5,
) -> InferenceResult:
    """Run nested-block permutation inference for fixed-effect columns.

    The tested block is defined as the set difference between the full fixed
    design stored on `fit_result` and the reduced fixed design built from
    `reduced_formula` using the same design-matrix builder. The resulting
    residualized block is tested jointly with a partial F statistic.

    Parameters
    ----------
    fit_result : FitResult
        Result returned by :func:`fit_lmm_mass_univariate`.
    reduced_formula : str | list[str]
        Reduced mixed formula, fixed formula, fixed RHS, or fixed-term list.
        Examples: ``"y ~ latency + (1|subject)"``, ``"latency"``, or
        ``["latency"]``. The reduced design must be nested in the full design.
    correction : str
        Correction backend: ``cluster`` or ``tfce``.
    n_permutations : int
        Number of within-group residual permutations.
    seed : int
        Random seed.
    tail : int
        Must be ``1`` because partial F statistics are non-negative.
    threshold : float | dict[str, float] | None
        Cluster threshold or TFCE threshold dictionary.
    adjacency : Any
        Optional adjacency matrix passed through to MNE correction backends.
    verbose : bool | str | int | None
        Verbosity forwarded to MNE-based correction backends.
    spatial_chunk_size : int | None
        Reserved for API symmetry; MNE cluster/TFCE currently process full maps.
    time_chunk_size : int | None
        Reserved for API symmetry; MNE cluster/TFCE currently process full maps.
    store_null_maps : bool
        Current backends store compact max-statistic null distributions only.
    tfce_h_power : float
        TFCE height exponent for ``correction="tfce"``.
    tfce_e_power : float
        TFCE extent exponent for ``correction="tfce"``.

    Returns
    -------
    InferenceResult
        Corrected nested-block inference output.
    """
    if correction == "cluster":
        backend = MNEBlockClusterCorrectionBackend()
        return backend.run(
            fit_result=fit_result,
            reduced_formula=reduced_formula,
            n_permutations=n_permutations,
            seed=seed,
            tail=tail,
            threshold=threshold,
            adjacency=adjacency,
            verbose=verbose,
            spatial_chunk_size=spatial_chunk_size,
            time_chunk_size=time_chunk_size,
            store_null_maps=store_null_maps,
        )
    if correction == "tfce":
        backend = MNEBlockTFCECorrectionBackend()
        return backend.run(
            fit_result=fit_result,
            reduced_formula=reduced_formula,
            n_permutations=n_permutations,
            seed=seed,
            tail=tail,
            threshold=threshold,
            adjacency=adjacency,
            verbose=verbose,
            spatial_chunk_size=spatial_chunk_size,
            time_chunk_size=time_chunk_size,
            store_null_maps=store_null_maps,
            tfce_h_power=tfce_h_power,
            tfce_e_power=tfce_e_power,
        )
    raise ValueError("Nested-block correction supports only 'cluster' and 'tfce'.")
