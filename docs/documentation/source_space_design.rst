Source-Space Design Notes
=========================

LmeEEG treats sensors, sources, parcels, and vertices as a generic spatial
feature axis. Source-space analysis therefore uses the same random-intercept
model as sensor-space analysis:

.. code-block:: text

   observations × locations × times

The public API keeps the argument name ``eeg`` for backward compatibility, but
``FitConfig(space="source")`` records that the second axis contains source
locations.

Memory guidance
---------------

Whole-brain source-space arrays can be large. The implementation avoids
observation × source × time long-format tables and correction backends store
compact max-statistic null distributions rather than full permutation null
maps.

Recommended practice:

- Use ``float32`` for large signal arrays when appropriate.
- Keep design matrices and statistical algebra in ``float64``.
- Use ``spatial_chunk_size`` and ``time_chunk_size`` for OLS and max-stat
  permutation inference.
- Prefer parcellated source spaces or MNE ico-4 style meshes during method
  development.
- Work at 50 to 100 Hz and restrict the time window to the scientific question.
- Use modest permutation counts during development, then increase them for
  final analyses.
- For TFCE, consider ``{"start": 0.0, "step": 0.5}`` during development and
  ``{"start": 0.0, "step": 0.2}`` for final analyses.

TFCE caveat
-----------

The TFCE backend uses optional MNE functionality. LmeEEG stores only one maximum
TFCE statistic per permutation, but MNE still processes complete statistic maps
internally. Large TFCE jobs may therefore need parcellation, shorter windows, or
coarser development settings even though null maps are not retained.
