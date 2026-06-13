"""Acceptance check for nested-block single-column equivalence.

Run from the repository root:

    python dev/nested_block_equivalence.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmeeeg.api.fit import fit_lmm_mass_univariate  # noqa: E402
from lmeeeg.api.infer import permute_fixed_block, permute_fixed_effect  # noqa: E402
from lmeeeg.backends.correction._regression import (  # noqa: E402
    compute_block_f_statistics,
    prepare_block_regression,
)
from lmeeeg.simulation.generator import simulate_random_intercept_dataset  # noqa: E402


def main() -> None:
    simulated = simulate_random_intercept_dataset(
        n_subjects=10,
        n_trials_per_subject=12,
        n_channels=1,
        n_times=1,
        effect_channels=[0],
        effect_times=[0],
        beta=1.5,
        seed=31,
    )
    fit_result = fit_lmm_mass_univariate(
        eeg=simulated.eeg,
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
    )

    prepared = prepare_block_regression(
        fit_result=fit_result,
        reduced_formula="y ~ latency + (1|subject)",
    )
    f_values = compute_block_f_statistics(
        y_residualized=prepared["y_residualized"],
        block_projection=prepared["block_projection"],
        y_sum_squares=prepared["y_sum_squares"],
        block_rank=int(prepared["block_rank"]),
        degrees_of_freedom=int(prepared["degrees_of_freedom"]),
    ).reshape(1, 1)
    t_values = fit_result.ols_t_values["condition[T.B]"]
    np.testing.assert_allclose(f_values, t_values**2, atol=1e-8)

    n_permutations = 50
    seed = 123
    t_result = permute_fixed_effect(
        fit_result=fit_result,
        effect="condition[T.B]",
        correction="cluster",
        n_permutations=n_permutations,
        seed=seed,
        tail=0,
        threshold=2.0,
        verbose=False,
    )
    f_result = permute_fixed_block(
        fit_result=fit_result,
        reduced_formula="y ~ latency + (1|subject)",
        correction="cluster",
        n_permutations=n_permutations,
        seed=seed,
        tail=1,
        threshold=4.0,
        verbose=False,
    )

    np.testing.assert_allclose(f_result.observed_statistic, t_result.observed_statistic**2, atol=1e-8)
    np.testing.assert_allclose(f_result.null_distribution, t_result.null_distribution**2, atol=1e-8)
    np.testing.assert_allclose(f_result.corrected_p_values, t_result.corrected_p_values, atol=0.0)
    np.testing.assert_array_equal(
        f_result.corrected_p_values <= 0.05,
        t_result.corrected_p_values <= 0.05,
    )
    print("nested-block single-column equivalence passed")
    print(f"t={float(t_values[0, 0]):.6f}, F={float(f_values[0, 0]):.6f}")
    print(f"corrected_p={float(f_result.corrected_p_values[0, 0]):.6f}")


if __name__ == "__main__":
    main()
