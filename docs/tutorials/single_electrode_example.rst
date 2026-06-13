single_electrode_example
========================

This tutorial mirrors the repository notebook :download:`single_electrode_example.ipynb <notebooks/single_electrode_example.ipynb>` and focuses on interpretation at one electrode in a more ERP-like simulation.

Goal
----

1. Simulate smooth ERP-like trial-wise data.
2. Inspect the condition averages and the ground-truth condition kernel.
3. Compare the fitted marginal coefficient to the known generating effect.

Imports
-------

.. code-block:: python

   from lmeeeg.api.fit import fit_lmm_mass_univariate
   from lmeeeg.api.simulate import simulate_erp_random_intercept_dataset

Simulate one-electrode ERP data
-------------------------------

.. code-block:: python

   simulation = simulate_erp_random_intercept_dataset(
       n_subjects=20,
       n_trials_per_subject=15,
       n_channels=1,
       sampling_rate_hz=512.0,
       tmin_s=-0.1,
       tmax_s=0.35,
       include_channel_covariance=True,
       noise_sd=0.8,
       ar1_rho=0.6,
       seed=7,
   )

Important interpretation
------------------------

The default simulator includes multiple ERP components, but the default condition manipulation is concentrated in the N200-like component. That means the condition-average waveform can show structure around both component latencies while the ground-truth condition beta remains concentrated later in time.

Fit the model
-------------

.. code-block:: python

   fit_result = fit_lmm_mass_univariate(
       eeg=simulation.eeg,
       metadata=simulation.metadata,
       formula="y ~ condition + latency + (1|subject)",
       variable_types={
           "condition": "categorical",
           "latency": "numeric",
           "subject": "group",
       },
   )

Compare estimated and true kernels
----------------------------------

.. code-block:: python

   electrode_index = 0
   estimated_kernel = fit_result.ols_betas["condition[T.B]"][electrode_index]
   estimated_t_values = fit_result.ols_t_values["condition[T.B]"][electrode_index]
   ground_truth = simulation.ground_truth_beta_condition[electrode_index]

What to inspect
---------------

- ``simulation.ground_truth_component_maps`` separates the contribution of each ERP component.
- ``simulation.ground_truth_beta_condition`` is the true condition effect.
- ``estimated_kernel`` is the recovered marginal coefficient at the selected electrode.
- ``estimated_t_values`` highlights the most statistically stable time regions.

The linked notebook includes the plotting code for these comparisons.
