API Reference
=============

The public API is intentionally compact. Most users interact with LmeEEG through one simulation helper, one fitting function, and one inference function.

This page is meant to be a practical reference rather than only a symbol index. Each main entry point includes a short description, guidance on when to use it, and one or more examples that can be adapted directly.

Workflow overview
-----------------

1. Generate or load trial-wise data shaped ``(n_observations, n_channels, n_times)`` for sensors or ``(n_observations, n_sources, n_times)`` for source space.
2. Fit the random-intercept workflow with :func:`lmeeeg.fit_lmm_mass_univariate`.
3. Run corrected inference with :func:`lmeeeg.permute_fixed_effect`.

End-to-end example
------------------

The shortest useful workflow looks like this:

.. code-block:: python

   from lmeeeg import (
       fit_lmm_mass_univariate,
       permute_fixed_effect,
       simulate_random_intercept_dataset,
   )

   simulated = simulate_random_intercept_dataset(
       n_subjects=10,
       n_trials_per_subject=12,
       n_channels=6,
       n_times=30,
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

   print(simulated.eeg.shape)
   print(fit_result.ols_betas["condition[T.B]"].shape)
   print(inference.corrected_p_values.min())

Fitting API
-----------

Use the fitting API when you already have trial-wise EEG and observation-level metadata and want to estimate the random-intercept lmeEEG pipeline.

The function returns a :class:`lmeeeg.core.results.FitResult` containing the design specification, per-feature mixed-model diagnostics, the marginalized EEG, and the mass-univariate OLS outputs.

.. autoclass:: lmeeeg.api.fit.FitConfig
   :members:
   :no-index:

.. autofunction:: lmeeeg.fit_lmm_mass_univariate
   :no-index:

Example
^^^^^^^

Minimal fit on simulated data:

.. code-block:: python

   from lmeeeg.api.fit import fit_lmm_mass_univariate
   from lmeeeg.simulation.generator import simulate_random_intercept_dataset

   simulated = simulate_random_intercept_dataset(seed=13)

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

   print(fit_result.convergence_summary.convergence_rate)

Inspect the main outputs:

.. code-block:: python

   print(fit_result.design_spec.fixed_column_names)
   print(fit_result.marginal_eeg.shape)
   print(fit_result.ols_betas["condition[T.B]"].shape)
   print(fit_result.ols_t_values["condition[T.B]"].shape)

Typical metadata pattern:

.. code-block:: python

   import pandas as pd

   metadata = pd.DataFrame(
       {
           "subject": ["sub-001", "sub-001", "sub-002", "sub-002"],
           "condition": ["A", "B", "A", "B"],
           "latency": [0.1, -0.2, 0.3, 0.0],
       }
   )

The matching EEG or source-space array for this metadata must have shape ``(4, n_channels, n_times)`` or ``(4, n_sources, n_times)`` because there is one metadata row per observation.

Common use cases
^^^^^^^^^^^^^^^^

- Use ``formula="y ~ condition + latency + (1|subject)"`` when you want a categorical condition effect with one numeric covariate and one random intercept factor.
- Set ``fit_intercept=False`` when you want the fixed-effects design matrix without an intercept term.
- Pass a :class:`lmeeeg.api.fit.FitConfig` when selecting explicit backend names, recording source-space metadata with ``space="source"``, choosing a signal-array dtype policy, or enabling chunked OLS with ``spatial_chunk_size`` / ``time_chunk_size``.
- The per-feature mixed-model fit shows a progress bar by default. Use ``FitConfig(show_progress=False)`` when you want a quiet run.

Source-space data
^^^^^^^^^^^^^^^^^

Source-space analysis uses the same model and expects data shaped ``(n_observations, n_sources, n_times)``. The core package does not require MNE objects.

.. code-block:: python

   fit_result = fit_lmm_mass_univariate(
       eeg=source_data.astype("float32", copy=False),
       metadata=metadata,
       formula="y ~ condition + latency + (1|subject)",
       variable_types={
           "condition": "categorical",
           "latency": "numeric",
           "subject": "group",
       },
       config=FitConfig(
           space="source",
           source_names=source_names,
           dtype="float32",
           spatial_chunk_size=512,
           time_chunk_size=50,
       ),
   )

Example without a fixed intercept:

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
       fit_intercept=False,
   )

   print(fit_result.design_spec.fixed_column_names)

Inference API
-------------

Use the inference API after fitting when you want corrected significance maps for one fixed effect. The selected ``effect`` must exactly match one of the fixed-effect column names stored in ``fit_result.design_spec.fixed_column_names``.

The function returns an :class:`lmeeeg.core.results.InferenceResult` with the observed statistic map, corrected p-values, null distribution, and backend-specific metadata.

.. autoclass:: lmeeeg.api.infer.PermutationConfig
   :members:
   :no-index:

.. autofunction:: lmeeeg.permute_fixed_effect
   :no-index:

Example
^^^^^^^

Max-stat correction:

.. code-block:: python

   from lmeeeg.api.infer import permute_fixed_effect

   inference = permute_fixed_effect(
       fit_result=fit_result,
       effect="condition[T.B]",
       correction="maxstat",
       n_permutations=200,
       seed=13,
   )

   print(inference.corrected_p_values.min())

Inspect the effect names before testing:

.. code-block:: python

   print(fit_result.design_spec.fixed_column_names)
   effect_name = "condition[T.B]"

Cluster correction with MNE-Python installed:

.. code-block:: python

   inference = permute_fixed_effect(
       fit_result=fit_result,
       effect="condition[T.B]",
       correction="cluster",
       n_permutations=500,
       seed=7,
       threshold=2.0,
   )

TFCE correction with MNE-Python installed:

.. code-block:: python

   inference = permute_fixed_effect(
       fit_result=fit_result,
       effect="condition[T.B]",
       correction="tfce",
       n_permutations=500,
       seed=7,
       threshold={"start": 0.0, "step": 0.2},
   )

How cluster and TFCE inference work
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Both MNE-backed inference modes operate on ``fit_result.marginal_eeg`` and first residualize the selected fixed effect and the EEG against the reduced model that excludes that effect.
- The package then computes a partial-regression t map across the full location-time grid.
- Max-stat inference stores only one maximum statistic per permutation and can process location/time chunks.
- TFCE inference stores only one maximum TFCE statistic per permutation, but MNE still processes complete statistic maps internally.
- Permutations shuffle observations only within the grouping-factor exchangeability blocks, not across all rows globally.
- Cluster correction uses the observed partial-regression t map, forms clusters at the chosen threshold, and compares each observed cluster statistic to a null distribution of the maximum absolute cluster statistic from each permutation.
- TFCE correction applies the TFCE transform to the partial-regression t map and compares each observed TFCE score to a null distribution of the maximum absolute TFCE score from each permutation.
- In both cases, the reported p-values are already multiple-comparison corrected because they are based on permutation maxima.

Common use cases
^^^^^^^^^^^^^^^^

- Choose ``correction="maxstat"`` for the built-in dependency-light correction backend.
- Choose ``correction="cluster"`` when you want cluster-based correction through MNE-Python.
- Choose ``correction="tfce"`` when you want threshold-free cluster enhancement through MNE-Python.
- Use ``tail=0`` for two-sided testing, ``tail=1`` for positive effects, and ``tail=-1`` for negative effects when using MNE-compatible backends.
- Use ``threshold=2.0`` as a simple cluster-forming threshold for cluster correction unless you have a stronger domain-specific choice.
- Use ``threshold={"start": 0.0, "step": 0.2}`` for TFCE unless you intentionally want to tune the TFCE integration grid.

Simulation API
--------------

The simulation helpers are useful for testing pipelines, writing examples, and validating whether the package can recover known effects under controlled conditions.

.. autofunction:: lmeeeg.simulate_random_intercept_dataset
   :no-index:

Example
^^^^^^^

Minimal random-intercept simulation:

.. code-block:: python

   simulated = simulate_random_intercept_dataset(
       n_subjects=12,
       n_trials_per_subject=20,
       n_channels=8,
       n_times=40,
       beta=0.8,
       seed=0,
   )

   print(simulated.eeg.shape)

Inspect the returned container:

.. code-block:: python

   print(simulated.metadata.head())
   print(simulated.ground_truth_effect.shape)

Localized effect example:

.. code-block:: python

   simulated = simulate_random_intercept_dataset(
       n_subjects=16,
       n_trials_per_subject=20,
       n_channels=12,
       n_times=60,
       effect_channels=[4, 5, 6],
       effect_times=range(18, 28),
       beta=1.2,
       random_intercept_sd=0.7,
       noise_sd=0.8,
       seed=21,
   )

.. autofunction:: lmeeeg.simulate_erp_random_intercept_dataset
   :no-index:

Example
^^^^^^^

Single-electrode ERP-style simulation:

.. code-block:: python

   from lmeeeg.api.simulate import simulate_erp_random_intercept_dataset

   simulation = simulate_erp_random_intercept_dataset(
       n_subjects=20,
       n_trials_per_subject=15,
       n_channels=1,
       sampling_rate_hz=512.0,
       tmin_s=-0.1,
       tmax_s=0.35,
       seed=7,
   )

   print(simulation.time_ms.shape)

Multi-channel ERP-style simulation:

.. code-block:: python

   simulation = simulate_erp_random_intercept_dataset(
       n_subjects=24,
       n_trials_per_subject=30,
       n_channels=32,
       sampling_rate_hz=250.0,
       tmin_s=-0.2,
       tmax_s=0.6,
       include_channel_covariance=True,
       noise_sd=1.0,
       ar1_rho=0.6,
       seed=3,
   )

   print(simulation.eeg.shape)
   print(simulation.ground_truth_beta_condition.shape)

Inspect the structured metadata:

.. code-block:: python

   print(simulation.metadata.head())
   print(simulation.simulation_metadata.component_names)
   print(simulation.simulation_metadata.channel_positions.shape)

Related simulation classes
--------------------------

These classes document the structure of the objects returned by the simulation helpers.

.. autoclass:: lmeeeg.simulation.generator.SimulatedDataset
   :members:
   :no-index:

.. autoclass:: lmeeeg.simulation.generator.ERPComponentSpec
   :members:
   :no-index:

.. autoclass:: lmeeeg.simulation.generator.ERPSimulationConfig
   :members:
   :no-index:

.. autoclass:: lmeeeg.simulation.generator.ERPSimulationMetadata
   :members:
   :no-index:

.. autoclass:: lmeeeg.simulation.generator.ERPSimulationResult
   :members:
   :no-index:

Package and module reference
----------------------------

.. autosummary::
   :toctree: documentation/generated
   :recursive:

   lmeeeg
   lmeeeg.api
   lmeeeg.backends
   lmeeeg.core
   lmeeeg.simulation
   lmeeeg.utils
   lmeeeg.viz
