import numpy as np

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def main() -> None:
    simulated = simulate_random_intercept_dataset(seed=7)

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
    print(fit_result.convergence_summary)

    inference_result = permute_fixed_effect(
        fit_result=fit_result,
        effect="condition[T.B]",
        correction="maxstat",
        n_permutations=200,
        seed=7,
    )
    print(np.nanmin(inference_result.corrected_p_values))


if __name__ == "__main__":
    main()
