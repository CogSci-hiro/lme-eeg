# LmeEEG

Minimal Python package implementing the core lmeEEG workflow for epoched M/EEG data with **random intercepts**:

1. parse a mixed-model style formula at the API edge,
2. fit a random-intercept mixed model at each channel × timepoint,
3. subtract the fitted random effects to obtain marginal EEG,
4. run fast mass-univariate OLS on the marginalized data,
5. perform max-stat, cluster, or TFCE correction.

## Current scope

- One grouping factor in the public API
- Random intercept only
- Trial-wise epoched data shaped `(n_observations, n_channels, n_times)`
- Cluster / TFCE correction via MNE-Python when installed
- Tiny simulation utilities for recovery / null checks

## Not yet included

- Random slopes
- Two grouping factors in the public API
- Real-data validation workflows
- Parallel / distributed optimization

## Basic example

```python
import numpy as np
import pandas as pd

from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_effect
from lmeeeg.simulation.generator import simulate_random_intercept_dataset

simulated = simulate_random_intercept_dataset(
    n_subjects=10,
    n_trials_per_subject=12,
    n_channels=4,
    n_times=25,
    effect_channels=[1, 2],
    effect_times=range(8, 14),
    beta=0.8,
    seed=13,
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

inference = permute_fixed_effect(
    fit_result=fit_result,
    effect="condition[T.B]",
    correction="maxstat",
    n_permutations=200,
    seed=13,
)

print(inference.corrected_p_values.shape)
```
