minimal_example
===============

This tutorial mirrors the repository notebook :download:`minimal_example.ipynb <notebooks/minimal_example.ipynb>` and shows the smallest end-to-end LmeEEG workflow.

Goal
----

1. Simulate a random-intercept dataset.
2. Fit the mixed-model plus marginal-OLS pipeline.
3. Run max-stat permutation correction for the condition effect.

Imports
-------

.. code-block:: python

   import numpy as np

   from lmeeeg.api.fit import fit_lmm_mass_univariate
   from lmeeeg.api.infer import permute_fixed_effect
   from lmeeeg.simulation.generator import simulate_random_intercept_dataset

Simulate data
-------------

.. code-block:: python

   simulated = simulate_random_intercept_dataset(
       n_subjects=10,
       n_trials_per_subject=12,
       n_channels=10,
       n_times=40,
       effect_channels=[2, 3],
       effect_times=range(12, 22),
       beta=1.0,
       seed=10,
   )

Fit the model
-------------

.. code-block:: python

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

Run corrected inference
-----------------------

.. code-block:: python

   inference = permute_fixed_effect(
       fit_result=fit_result,
       effect="condition[T.B]",
       correction="maxstat",
       n_permutations=100,
       seed=7,
   )

Inspect outputs
---------------

The main arrays produced by this workflow are:

- ``simulated.ground_truth_effect`` for the simulated signal.
- ``fit_result.ols_betas["condition[T.B]"]`` for the estimated marginal coefficient map.
- ``fit_result.ols_t_values["condition[T.B]"]`` for the uncorrected t-map.
- ``inference.corrected_p_values`` for feature-wise corrected p-values.

Typical next step
-----------------

The source notebook plots the ground-truth effect, the estimated t-map, and the corrected p-value map side by side. Use that notebook when you want the full visual walkthrough.
