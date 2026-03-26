from lmeeeg.core.design import build_design_spec
from lmeeeg.simulation.generator import simulate_random_intercept_dataset


def test_build_design_spec() -> None:
    simulated = simulate_random_intercept_dataset(n_subjects=2, n_trials_per_subject=4, seed=3)
    design_spec = build_design_spec(
        metadata=simulated.metadata,
        formula="y ~ condition + latency + (1|subject)",
        variable_types={
            "condition": "categorical",
            "latency": "numeric",
            "subject": "group",
        },
    )
    assert design_spec.fixed_design_matrix.shape[0] == len(simulated.metadata)
    assert design_spec.group_variable == "subject"
