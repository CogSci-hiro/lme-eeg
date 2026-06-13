Source-Space Example
====================

Source-space data use the same model as sensor-space data. The second array
axis simply represents cortical vertices, parcels, or another spatial feature
axis instead of EEG channels.

Expected shape
--------------

Pass source-space data as:

.. code-block:: text

   (n_observations, n_sources, n_times)

The metadata still has one row per observation.

Minimal source-space fit
------------------------

.. code-block:: python

   import numpy as np

   from lmeeeg.api.fit import FitConfig, fit_lmm_mass_univariate
   from lmeeeg.api.infer import permute_fixed_effect

   # Replace this with source estimates or parcel time courses from your own
   # preprocessing workflow.
   source_data = source_data.astype("float32", copy=False)
   source_names = [f"src-{index:05d}" for index in range(source_data.shape[1])]

   fit_result = fit_lmm_mass_univariate(
       eeg=source_data,
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
           show_progress=False,
       ),
   )

   inference = permute_fixed_effect(
       fit_result=fit_result,
       effect="condition[T.B]",
       correction="maxstat",
       n_permutations=200,
       seed=13,
       spatial_chunk_size=512,
       time_chunk_size=50,
   )

   print(fit_result.space)
   print(fit_result.n_sources)
   print(inference.corrected_p_values.shape)

TFCE with source-space adjacency
--------------------------------

TFCE remains available through the optional MNE backend. The core package does
not require MNE objects, but you can pass a source-space adjacency matrix when
you have one.

.. code-block:: python

   inference = permute_fixed_effect(
       fit_result=fit_result,
       effect="condition[T.B]",
       correction="tfce",
       n_permutations=500,
       seed=13,
       threshold={"start": 0.0, "step": 0.2},
       adjacency=source_adjacency,
   )

For development runs, consider a coarser TFCE step such as ``0.5`` and a modest
permutation count. For final analyses, use the resolution and permutation count
required by your study design.

Simulation validation
---------------------

The repository includes a manual validation script for source-space support:

.. code-block:: bash

   PYTHONPATH=src python scripts/validate_source_lmeeeg_simulation.py \
       --scenario recovery \
       --n-subjects 8 \
       --trials-per-subject 12 \
       --n-sources 32 \
       --n-times 20 \
       --n-permutations 200 \
       --spatial-chunk-size 16 \
       --time-chunk-size 10 \
       --check-chunk-equivalence \
       --out-dir outputs/source_lmeeeg_validation

Use ``--scenario recovery`` to test whether LmeEEG recovers a planted smooth
source-time condition effect. Use ``--scenario null`` to estimate the
family-wise false-positive rate when no condition effect is present.

The script runs simulation replicates in batches and computes a Wilson
confidence interval after each batch. Once the detection-rate or false-positive
rate interval is narrower than ``--tolerance`` after ``--min-reps`` valid
replicates, it stops early. It always stops by ``--max-reps``.

Two kinds of convergence are reported:

- Model convergence is the per-replicate mixed-model convergence reported by
  LmeEEG across source-time features. Replicates with failed model convergence
  are written to the CSV but excluded from the detection/FPR denominator.
- Monte Carlo convergence is the stability of the simulation-level
  detection-rate or false-positive-rate estimate. This is controlled by the
  Wilson confidence-interval stopping rule.

The optional ``--check-chunk-equivalence`` check runs the first replicate both
with and without chunked processing and verifies that the observed t maps are
numerically close. Keep this check for small validation grids; it intentionally
does extra work.
