Design Principles
=================

LmeEEG is intentionally narrow in scope. The implementation favors an explicit, inspectable pipeline over a large abstraction surface.

Core principles
---------------

1. Keep the public API small.
   The main workflow is exposed through a small number of user-facing entry points in :mod:`lmeeeg.api`.

2. Separate model fitting from correction.
   The mixed-model layer estimates feature-wise random intercepts. Inference backends operate afterward on marginalized data.

3. Prefer explicit data transformations.
   Design-matrix construction, random-effect subtraction, OLS fitting, and permutation correction each live in separate modules with clear inputs and outputs.

4. Make optional dependencies truly optional.
   The base package runs without MNE-Python. MNE-specific cluster and TFCE correction backends are enabled only when the optional extra is installed.

5. Optimize for scientific readability.
   The package structure mirrors the analysis stages: formula parsing, design construction, mixed-model fitting, marginalization, mass-univariate OLS, and corrected inference.

Workflow overview
-----------------

The current implementation follows this sequence:

1. Parse a mixed-model style formula such as ``"y ~ condition + latency + (1|subject)"``.
2. Build the fixed-effects design matrix and the grouping codes from observation-level metadata.
3. Fit a random-intercept mixed model at each channel and time sample.
4. Subtract the fitted random contribution to produce marginalized EEG.
5. Fit closed-form OLS to the marginalized data for every feature.
6. Run permutation-based correction on the selected fixed effect.

Design tradeoffs
----------------

- The current public API supports a single grouping factor and random intercepts only.
- The default max-stat backend uses a deliberately simple row-shuffle permutation scheme so the implementation stays easy to inspect.
- The MNE-based correction backends use the marginalized data representation rather than coupling permutation logic to the mixed-model solver itself.
