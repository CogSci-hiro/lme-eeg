from __future__ import annotations

import os
import tempfile
import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from lmeeeg.core.results import FitResult


def configure_mne_runtime() -> None:
    """Keep optional MNE imports usable in sandboxed environments."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


def prepare_effect_regression(
    fit_result: FitResult,
    effect: str,
) -> dict[str, np.ndarray | int]:
    """Residualize the selected effect and EEG against the reduced model."""
    if fit_result.marginal_eeg is None:
        raise ValueError(
            "Permutation inference requires `fit_result.marginal_eeg`. "
            "Run `fit_lmm_mass_univariate(..., config=FitConfig(store_marginal_eeg=True))`."
        )

    column_names = fit_result.design_spec.fixed_column_names
    effect_index = column_names.index(effect)
    x_matrix = fit_result.design_spec.fixed_design_matrix
    n_observations = x_matrix.shape[0]
    n_channels = fit_result.marginal_eeg.shape[1]
    n_times = fit_result.marginal_eeg.shape[2]
    y_2d = fit_result.marginal_eeg.reshape(n_observations, n_channels * n_times)

    reduced_columns = [index for index in range(len(column_names)) if index != effect_index]
    effect_column = x_matrix[:, effect_index]

    if reduced_columns:
        x_reduced = x_matrix[:, reduced_columns]
        x_reduced_pinv = np.linalg.pinv(x_reduced)
        y_residualized = y_2d - x_reduced @ (x_reduced_pinv @ y_2d)
        effect_residualized = effect_column - x_reduced @ (x_reduced_pinv @ effect_column)
    else:
        y_residualized = y_2d.copy()
        effect_residualized = effect_column.copy()

    effect_ss = float(effect_residualized @ effect_residualized)
    if np.isclose(effect_ss, 0.0):
        raise ValueError(
            f"Selected effect '{effect}' has no residualized variation after removing nuisance regressors."
        )

    degrees_of_freedom = n_observations - np.linalg.matrix_rank(x_matrix)
    if degrees_of_freedom <= 0:
        raise ValueError("Permutation inference requires positive residual degrees of freedom.")

    group_codes = fit_result.design_spec.group_codes
    block_sizes = np.bincount(group_codes)
    if np.all(block_sizes <= 1):
        raise ValueError(
            "Within-group permutation requires at least one group with more than one observation."
        )
    if np.any(block_sizes == 1):
        warnings.warn(
            "Some groups contain a single observation and therefore cannot contribute within-group permutations.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "y_residualized": y_residualized,
        "effect_residualized": effect_residualized,
        "effect_sum_squares": effect_ss,
        "degrees_of_freedom": degrees_of_freedom,
        "group_codes": group_codes,
        "n_channels": n_channels,
        "n_times": n_times,
    }


def compute_effect_t_statistics(
    y_residualized: np.ndarray,
    effect_residualized: np.ndarray,
    effect_sum_squares: float,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Compute the partial-regression t statistic for one fixed effect."""
    beta = (effect_residualized[:, None] * y_residualized).sum(axis=0) / effect_sum_squares
    fitted = effect_residualized[:, None] * beta[None, :]
    residuals = y_residualized - fitted
    residual_variance = np.sum(residuals ** 2, axis=0) / degrees_of_freedom
    standard_error = np.sqrt(residual_variance / effect_sum_squares)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(beta, standard_error, out=np.zeros_like(beta), where=standard_error > 0)
    return t_values


def permute_within_groups(
    y_residualized: np.ndarray,
    group_codes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute observations only within exchangeability blocks."""
    permuted_indices = np.arange(group_codes.shape[0])
    for group_code in np.unique(group_codes):
        group_indices = np.flatnonzero(group_codes == group_code)
        if group_indices.size > 1:
            permuted_indices[group_indices] = group_indices[rng.permutation(group_indices.size)]
    return y_residualized[permuted_indices, :]


def cluster_outputs_to_masks(
    clusters: list | tuple | None,
    sample_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Normalize MNE cluster outputs to boolean masks."""
    if not clusters:
        return []

    masks: list[np.ndarray] = []
    for cluster in clusters:
        mask = np.zeros(sample_shape, dtype=bool)
        if isinstance(cluster, tuple):
            mask[cluster] = True
        else:
            cluster_array = np.asarray(cluster)
            if cluster_array.dtype == bool:
                mask = cluster_array.reshape(sample_shape)
            else:
                mask.reshape(-1)[cluster_array] = True
        masks.append(mask)
    return masks


def should_log_info(verbose: bool | str | int | None) -> bool:
    """Interpret user verbosity settings conservatively."""
    if verbose is None or verbose is False:
        return False
    if isinstance(verbose, str):
        return verbose.lower() in {"info", "debug"}
    if isinstance(verbose, bool):
        return verbose
    if isinstance(verbose, int):
        return verbose > 0
    return False


def emit_info(verbose: bool | str | int | None, message: str, *args: Any) -> None:
    """Emit lightweight backend progress when verbosity requests it."""
    if should_log_info(verbose):
        if args:
            message = message.format(*args)
        print(message)


def build_progress(verbose: bool | str | int | None) -> Progress | None:
    """Create a rich progress bar when verbosity requests live updates."""
    if not should_log_info(verbose):
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def progress_context(verbose: bool | str | int | None):
    """Return a context manager for optional progress reporting."""
    progress = build_progress(verbose)
    return progress if progress is not None else nullcontext()
