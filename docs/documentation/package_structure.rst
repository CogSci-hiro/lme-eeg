Package Structure
=================

The repository is organized around the analysis pipeline.

Repository layout
-----------------

.. code-block:: text

   lme-eeg/
   ├── docs/
   ├── notebooks/
   │   ├── minimal_example.ipynb
   │   └── single_electrode_example.ipynb
   ├── src/
   │   └── lmeeeg/
   │       ├── api/
   │       ├── backends/
   │       ├── core/
   │       ├── simulation/
   │       ├── utils/
   │       └── viz/
   └── tests/

Subpackages
-----------

``lmeeeg.api``
   User-facing entry points for fitting, inference, and simulation helpers.

``lmeeeg.core``
   Formula parsing, design-spec construction, contrast helpers, marginalization logic, and structured result containers.

``lmeeeg.backends``
   Concrete implementations for mixed-model fitting, OLS estimation, and permutation correction.

``lmeeeg.simulation``
   Minimal and ERP-style dataset simulators plus reusable scenario definitions.

``lmeeeg.utils``
   Validation, reshape, and summary helpers.

``lmeeeg.viz``
   Optional visualization helpers that integrate with MNE-Python concepts.

Module responsibilities
-----------------------

- :mod:`lmeeeg.core.formulas` parses the mixed-model style formula.
- :mod:`lmeeeg.core.design` constructs the fixed-effects design matrix and group codes.
- :mod:`lmeeeg.backends.lmm.statsmodels_backend` fits the feature-wise random-intercept models.
- :mod:`lmeeeg.core.marginal` computes the marginalized EEG by subtracting fitted random contributions.
- :mod:`lmeeeg.backends.ols.numpy_backend` fits OLS across all channel-time features in closed form.
- :mod:`lmeeeg.backends.correction` provides max-stat and optional MNE-based correction backends.

Public workflow map
-------------------

.. code-block:: text

   metadata + formula + eeg
            |
            v
   lmeeeg.api.fit.fit_lmm_mass_univariate
            |
            +--> lmeeeg.core.design.build_design_spec
            +--> lmeeeg.backends.lmm.statsmodels_backend
            +--> lmeeeg.core.marginal.compute_marginal_eeg
            +--> lmeeeg.backends.ols.numpy_backend
            |
            v
          FitResult
            |
            v
   lmeeeg.api.infer.permute_fixed_effect
            |
            +--> maxstat / cluster / tfce backend
            |
            v
       InferenceResult
