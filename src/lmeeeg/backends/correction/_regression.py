import os
import tempfile
import warnings
from contextlib import nullcontext
from collections.abc import Sequence
from typing import Any

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from lmeeeg.core.design import build_design_spec
from lmeeeg.core.results import FitResult

PermutationSeed = int | np.random.Generator | np.random.RandomState


def configure_mne_runtime() -> None:
    """Keep optional MNE imports usable in sandboxed environments."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


def make_permutation_rng(seed: PermutationSeed) -> np.random.Generator | np.random.RandomState:
    """Return a permutation RNG while preserving existing integer-seed behavior."""
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        return seed
    return np.random.default_rng(seed)


def residualize_against_nuisance(
    values: np.ndarray,
    nuisance_design: np.ndarray,
) -> np.ndarray:
    """Residualize columns of `values` against nuisance regressors."""
    values = np.asarray(values, dtype=np.float64)
    nuisance_design = np.asarray(nuisance_design, dtype=np.float64)
    if nuisance_design.size == 0 or nuisance_design.shape[1] == 0:
        return values.copy()
    nuisance_pinv = np.linalg.pinv(nuisance_design)
    return values - nuisance_design @ (nuisance_pinv @ values)


def _validate_exchangeability_blocks(group_codes: np.ndarray) -> None:
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
    n_locations = fit_result.marginal_eeg.shape[1]
    n_times = fit_result.marginal_eeg.shape[2]
    y_2d = fit_result.marginal_eeg.reshape(n_observations, n_locations * n_times)

    reduced_columns = [index for index in range(len(column_names)) if index != effect_index]
    effect_column = x_matrix[:, effect_index]

    if reduced_columns:
        x_reduced = x_matrix[:, reduced_columns]
        y_residualized = residualize_against_nuisance(y_2d, x_reduced)
        effect_residualized = residualize_against_nuisance(effect_column, x_reduced)
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
    _validate_exchangeability_blocks(group_codes)

    return {
        "y_residualized": y_residualized,
        "effect_residualized": effect_residualized,
        "effect_column": effect_column,
        "nuisance_design": x_reduced if reduced_columns else np.empty((n_observations, 0)),
        "effect_sum_squares": effect_ss,
        "degrees_of_freedom": degrees_of_freedom,
        "group_codes": group_codes,
        "n_locations": n_locations,
        "n_channels": n_locations,
        "n_times": n_times,
    }


def max_permutation_p_values(null_distribution: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute max-stat corrected p-values using the (b + 1) / (m + 1) form."""
    null_distribution = np.asarray(null_distribution, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return (
        1
        + np.sum(
            null_distribution[(slice(None),) + (None,) * observed.ndim] >= np.abs(observed)[None, ...],
            axis=0,
        )
    ) / (null_distribution.shape[0] + 1)


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


def _coerce_reduced_formula(
    reduced_formula: str | Sequence[str],
    group_variable: str,
) -> str:
    """Normalize supported reduced-model specifications to a mixed formula."""
    if isinstance(reduced_formula, str):
        reduced_text = reduced_formula.strip()
        if not reduced_text:
            reduced_text = "1"
        if "(1|" in reduced_text.replace(" ", ""):
            return reduced_text
        if "~" in reduced_text:
            return f"{reduced_text} + (1|{group_variable})"
        return f"y ~ {reduced_text} + (1|{group_variable})"

    reduced_terms = [str(term).strip() for term in reduced_formula if str(term).strip()]
    fixed_terms = " + ".join(reduced_terms) if reduced_terms else "1"
    return f"y ~ {fixed_terms} + (1|{group_variable})"


def prepare_block_regression(
    fit_result: FitResult,
    reduced_formula: str | Sequence[str],
) -> dict[str, Any]:
    """Prepare residualized data and projection for a nested fixed-effect block."""
    if fit_result.marginal_eeg is None:
        raise ValueError(
            "Permutation inference requires `fit_result.marginal_eeg`. "
            "Run `fit_lmm_mass_univariate(..., config=FitConfig(store_marginal_eeg=True))`."
        )
    if fit_result.design_spec.metadata is None:
        raise ValueError(
            "Nested block inference requires the metadata stored on `fit_result.design_spec`. "
            "Refit with the current package version before using reduced formulas."
        )

    full_design = np.asarray(fit_result.design_spec.fixed_design_matrix, dtype=np.float64)
    full_column_names = list(fit_result.design_spec.fixed_column_names)
    full_column_lookup = {column_name: index for index, column_name in enumerate(full_column_names)}
    group_variable = fit_result.design_spec.group_variable
    normalized_reduced_formula = _coerce_reduced_formula(
        reduced_formula=reduced_formula,
        group_variable=group_variable,
    )
    fit_intercept = "Intercept" in full_column_lookup
    reduced_design_spec = build_design_spec(
        metadata=fit_result.design_spec.metadata,
        formula=normalized_reduced_formula,
        variable_types=fit_result.variable_types,
        fit_intercept=fit_intercept,
    )
    if reduced_design_spec.group_variable != group_variable:
        raise ValueError(
            "Reduced formula must use the same random-intercept grouping variable "
            f"as the full model: '{group_variable}'."
        )

    missing_reduced_columns = [
        column_name
        for column_name in reduced_design_spec.fixed_column_names
        if column_name not in full_column_lookup
    ]
    if missing_reduced_columns:
        missing = ", ".join(missing_reduced_columns)
        available = ", ".join(full_column_names)
        raise ValueError(
            "Reduced model is not nested in the full model. "
            f"Reduced columns absent from the full design: {missing}. "
            f"Full design columns: {available}"
        )

    reduced_indices = [full_column_lookup[column_name] for column_name in reduced_design_spec.fixed_column_names]
    for reduced_position, full_index in enumerate(reduced_indices):
        reduced_values = reduced_design_spec.fixed_design_matrix[:, reduced_position]
        full_values = full_design[:, full_index]
        if not np.allclose(reduced_values, full_values, equal_nan=True):
            column_name = reduced_design_spec.fixed_column_names[reduced_position]
            raise ValueError(
                "Reduced model is not nested in the full model with identical design values. "
                f"Column '{column_name}' has different encoded values."
            )

    reduced_column_set = set(reduced_design_spec.fixed_column_names)
    block_indices = [
        index
        for index, column_name in enumerate(full_column_names)
        if column_name not in reduced_column_set
    ]
    if not block_indices:
        raise ValueError("Reduced model removes no full-design columns; nested block is empty.")

    n_observations = full_design.shape[0]
    n_locations = fit_result.marginal_eeg.shape[1]
    n_times = fit_result.marginal_eeg.shape[2]
    y_2d = np.asarray(fit_result.marginal_eeg, dtype=np.float64).reshape(
        n_observations,
        n_locations * n_times,
    )
    x_nuisance = full_design[:, reduced_indices] if reduced_indices else np.empty((n_observations, 0))
    x_block = full_design[:, block_indices]
    y_residualized = residualize_against_nuisance(y_2d, x_nuisance)
    block_residualized = residualize_against_nuisance(x_block, x_nuisance)

    block_rank = int(np.linalg.matrix_rank(block_residualized))
    if block_rank == 0:
        block_columns = ", ".join(full_column_names[index] for index in block_indices)
        raise ValueError(
            "Nested block is fully collinear with the nuisance design after residualization: "
            f"{block_columns}"
        )

    degrees_of_freedom = int(n_observations - np.linalg.matrix_rank(full_design))
    if degrees_of_freedom <= 0:
        raise ValueError("Nested block inference requires positive residual degrees of freedom.")

    group_codes = fit_result.design_spec.group_codes
    _validate_exchangeability_blocks(group_codes)
    block_projection = block_residualized @ np.linalg.pinv(block_residualized)

    return {
        "y_residualized": y_residualized,
        "y_sum_squares": np.sum(y_residualized ** 2, axis=0),
        "block_residualized": block_residualized,
        "block_projection": block_projection,
        "block_rank": block_rank,
        "degrees_of_freedom": degrees_of_freedom,
        "group_codes": group_codes,
        "n_locations": n_locations,
        "n_channels": n_locations,
        "n_times": n_times,
        "full_column_names": full_column_names,
        "reduced_column_names": list(reduced_design_spec.fixed_column_names),
        "block_column_names": [full_column_names[index] for index in block_indices],
        "normalized_reduced_formula": normalized_reduced_formula,
    }


def compute_block_f_statistics(
    y_residualized: np.ndarray,
    block_projection: np.ndarray,
    y_sum_squares: np.ndarray,
    block_rank: int,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Compute a partial F statistic for a residualized fixed-effect block."""
    projected = block_projection @ y_residualized
    block_sum_squares = np.sum(projected ** 2, axis=0)
    residual_sum_squares = y_sum_squares - block_sum_squares
    with np.errstate(divide="ignore", invalid="ignore"):
        return (block_sum_squares / block_rank) / (residual_sum_squares / degrees_of_freedom)


def validate_f_statistics(f_values: np.ndarray, label: str) -> None:
    """Fail loudly when partial-F maps contain invalid values."""
    invalid_mask = ~np.isfinite(f_values)
    if np.any(invalid_mask):
        invalid_count = int(np.sum(invalid_mask))
        raise ValueError(
            f"Nested-block partial F statistic contains {invalid_count} non-finite values in {label}."
        )
    negative_mask = f_values < 0
    if np.any(negative_mask):
        negative_count = int(np.sum(negative_mask))
        minimum = float(np.min(f_values))
        raise ValueError(
            "Nested-block partial F statistic must be non-negative, "
            f"but {negative_count} values were negative in {label}; minimum={minimum}."
        )


def permute_within_groups(
    y_residualized: np.ndarray,
    group_codes: np.ndarray,
    rng: np.random.Generator | np.random.RandomState,
) -> np.ndarray:
    """Permute observations only within exchangeability blocks."""
    permuted_indices = np.arange(group_codes.shape[0])
    for group_code in np.unique(group_codes):
        group_indices = np.flatnonzero(group_codes == group_code)
        if group_indices.size > 1:
            permuted_indices[group_indices] = group_indices[rng.permutation(group_indices.size)]
    return y_residualized[permuted_indices, :]


def permute_vector_within_groups(
    values: np.ndarray,
    group_codes: np.ndarray,
    rng: np.random.Generator | np.random.RandomState,
) -> np.ndarray:
    """Permute a one-dimensional vector within exchangeability blocks."""
    values = np.asarray(values)
    permuted = values.copy()
    for group_code in np.unique(group_codes):
        group_indices = np.flatnonzero(group_codes == group_code)
        if group_indices.size > 1:
            permuted[group_indices] = values[group_indices[rng.permutation(group_indices.size)]]
    return permuted


def permuted_effect_for_scheme(
    effect_column: np.ndarray,
    nuisance_design: np.ndarray,
    group_codes: np.ndarray,
    rng: np.random.Generator | np.random.RandomState,
    permutation_scheme: str,
) -> tuple[np.ndarray, float]:
    """Permute the tested effect vector and residualize it against nuisance terms."""
    if permutation_scheme == "free":
        permuted_effect = np.asarray(effect_column)[rng.permutation(effect_column.shape[0])]
    elif permutation_scheme == "within_subject":
        permuted_effect = permute_vector_within_groups(
            values=effect_column,
            group_codes=group_codes,
            rng=rng,
        )
    else:
        raise ValueError("permutation_scheme must be 'free' or 'within_subject'.")
    effect_residualized = residualize_against_nuisance(permuted_effect, nuisance_design)
    if effect_residualized.ndim == 2:
        effect_residualized = effect_residualized[:, 0]
    effect_sum_squares = float(effect_residualized @ effect_residualized)
    if np.isclose(effect_sum_squares, 0.0):
        raise ValueError("Permuted effect has no residualized variation after removing nuisance regressors.")
    return effect_residualized, effect_sum_squares


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
