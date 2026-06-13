Mathematical Definition of the Algorithm
========================================

LmeEEG implements a minimal random-intercept version of the lmeEEG workflow for trial-wise epoched data.

Notation
--------

Let:

- :math:`n` be the number of observations.
- :math:`C` be the number of channels.
- :math:`T` be the number of sampled time points.
- :math:`p` be the number of fixed-effect regressors.
- :math:`G` be the number of groups in the random-intercept factor.

For each feature :math:`f = (c, t)` with channel :math:`c` and time :math:`t`, define:

- :math:`\mathbf{y}_f \in \mathbb{R}^{n}` as the observed EEG values across observations.
- :math:`\mathbf{X} \in \mathbb{R}^{n \times p}` as the fixed-effects design matrix.
- :math:`\mathbf{Z} \in \mathbb{R}^{n \times G}` as the random-intercept design matrix.
- :math:`\boldsymbol{\beta}_f \in \mathbb{R}^{p}` as the fixed-effect coefficients at feature :math:`f`.
- :math:`\mathbf{b}_f \in \mathbb{R}^{G}` as the random intercepts at feature :math:`f`.
- :math:`\boldsymbol{\varepsilon}_f \in \mathbb{R}^{n}` as the residual noise.

Feature-wise mixed model
------------------------

At every channel-time feature, the package fits the linear mixed model

.. math::

   \mathbf{y}_f = \mathbf{X}\boldsymbol{\beta}_f + \mathbf{Z}\mathbf{b}_f + \boldsymbol{\varepsilon}_f.

The current implementation assumes a random-intercept structure:

.. math::

   \mathbf{b}_f \sim \mathcal{N}\left(\mathbf{0}, \sigma_{b,f}^{2}\mathbf{I}_G\right),

.. math::

   \boldsymbol{\varepsilon}_f \sim \mathcal{N}\left(\mathbf{0}, \sigma_{\varepsilon,f}^{2}\mathbf{I}_n\right).

The grouping factor is extracted from the mixed-model formula and encoded into integer group codes during design construction.

Marginalization step
--------------------

After fitting the mixed model at each feature, the package reconstructs the fitted random contribution

.. math::

   \widehat{\mathbf{r}}_f = \mathbf{Z}\widehat{\mathbf{b}}_f

and subtracts it from the original EEG to obtain the marginalized data:

.. math::

   \widetilde{\mathbf{y}}_f = \mathbf{y}_f - \widehat{\mathbf{r}}_f.

This is the same object implemented in :func:`lmeeeg.core.marginal.compute_marginal_eeg`.

Mass-univariate OLS on marginalized EEG
---------------------------------------

The marginalized observations are then analyzed with ordinary least squares using the same fixed-effects design:

.. math::

   \widetilde{\mathbf{y}}_f = \mathbf{X}\widetilde{\boldsymbol{\beta}}_f + \widetilde{\boldsymbol{\eta}}_f.

The closed-form estimator is

.. math::

   \widehat{\widetilde{\boldsymbol{\beta}}}_f =
   \left(\mathbf{X}^{\top}\mathbf{X}\right)^{-1}\mathbf{X}^{\top}\widetilde{\mathbf{y}}_f.

The residual variance estimate is

.. math::

   \widehat{\sigma}^{2}_{f} =
   \frac{\left\lVert \widetilde{\mathbf{y}}_f - \mathbf{X}\widehat{\widetilde{\boldsymbol{\beta}}}_f \right\rVert_2^2}
   {n - p}.

For coefficient :math:`j`, the standard error and t-statistic are

.. math::

   \operatorname{SE}\!\left(\widehat{\widetilde{\beta}}_{f,j}\right) =
   \sqrt{\widehat{\sigma}^{2}_{f}\left[\left(\mathbf{X}^{\top}\mathbf{X}\right)^{-1}\right]_{jj}},

.. math::

   t_{f,j} = \frac{\widehat{\widetilde{\beta}}_{f,j}}
   {\operatorname{SE}\!\left(\widehat{\widetilde{\beta}}_{f,j}\right)}.

These t-maps are the statistics exposed in ``FitResult.ols_t_values``.

Permutation correction
----------------------

LmeEEG currently exposes three correction backends.

Max-stat backend
^^^^^^^^^^^^^^^^

For the built-in max-stat backend, the package permutes observation rows of the fixed-effects design matrix while keeping the marginalized data fixed:

.. math::

   \mathbf{X}^{(k)} = \mathbf{P}_k \mathbf{X},

where :math:`\mathbf{P}_k` is a permutation matrix for permutation :math:`k`.

For each permutation, the package recomputes the OLS t-statistic map for the selected effect and stores

.. math::

   M^{(k)} = \max_f \left| t_f^{(k)} \right|.

The corrected p-value at feature :math:`f` is then estimated by

.. math::

   p_f = \frac{1 + \sum_{k=1}^{K} \mathbb{I}\left(M^{(k)} \ge \left|t_f^{\mathrm{obs}}\right|\right)}
   {K + 1}.

MNE cluster and TFCE backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MNE-powered backends operate on observation-level marginalized data, but the permutation logic is implemented in LmeEEG itself rather than delegated to a high-level MNE permutation wrapper.

For a selected fixed effect with column :math:`\mathbf{x}_j`, define the reduced fixed-effects design matrix :math:`\mathbf{X}_{-j}` by removing that column from :math:`\mathbf{X}`. Let :math:`\mathbf{Y} \in \mathbb{R}^{n \times (CT)}` be the marginalized EEG reshaped so each column corresponds to one channel-time feature.

The backend first residualizes both the EEG and the selected effect with respect to the reduced model:

.. math::

   \mathbf{Y}^{\ast} = \mathbf{Y} - \mathbf{X}_{-j}\mathbf{X}_{-j}^{+}\mathbf{Y},

.. math::

   \mathbf{x}_j^{\ast} = \mathbf{x}_j - \mathbf{X}_{-j}\mathbf{X}_{-j}^{+}\mathbf{x}_j,

where :math:`(\cdot)^{+}` denotes the Moore-Penrose pseudoinverse.

For each feature :math:`f`, the package then computes a partial-regression coefficient and t-statistic:

.. math::

   \widehat{\beta}^{\ast}_{f,j} =
   \frac{\left(\mathbf{x}_j^{\ast}\right)^{\top}\mathbf{y}^{\ast}_f}
   {\left(\mathbf{x}_j^{\ast}\right)^{\top}\mathbf{x}_j^{\ast}},

.. math::

   \widehat{\sigma}^{2,\ast}_{f} =
   \frac{\left\lVert \mathbf{y}^{\ast}_f - \mathbf{x}_j^{\ast}\widehat{\beta}^{\ast}_{f,j} \right\rVert_2^2}
   {n - \operatorname{rank}(\mathbf{X})},

.. math::

   t^{\ast}_{f,j} =
   \frac{\widehat{\beta}^{\ast}_{f,j}}
   {\sqrt{\widehat{\sigma}^{2,\ast}_{f} / \left[\left(\mathbf{x}_j^{\ast}\right)^{\top}\mathbf{x}_j^{\ast}\right]}}.

These partial-regression t values are the observed statistics for cluster correction and the inputs to the TFCE transform.

Exchangeability scheme
+++++++++++++++++++++++++++++

Permutations are not performed across all observations. Instead, the backend shuffles rows of :math:`\mathbf{Y}^{\ast}` only within the exchangeability blocks defined by the grouping factor from the random-intercept model.

If :math:`g(i)` denotes the group label of observation :math:`i`, then each permutation :math:`\pi_k` satisfies

.. math::

   g(\pi_k(i)) = g(i) \quad \text{for all } i.

This preserves the within-group permutation scheme used by the implementation.

Cluster correction
++++++++++++++++++

For cluster correction, the observed partial-regression t map is thresholded at the user-supplied cluster-forming threshold (default ``2.0``) and connected components are formed using the supplied adjacency, if any. MNE is used only for the cluster-identification and cluster-statistic computation step.

For each observed cluster :math:`c`, let :math:`S_c^{\mathrm{obs}}` denote its cluster statistic. For each permutation :math:`k`, the backend recomputes the partial-regression t map from the permuted residualized data, re-forms clusters, and stores the maximum absolute cluster statistic:

.. math::

   M_{\mathrm{cluster}}^{(k)} = \max_c \left| S_c^{(k)} \right|,

with the convention that this value is zero if no cluster is found in permutation :math:`k`.

The cluster-level corrected p-value is then estimated as

.. math::

   p_c = \frac{1 + \sum_{k=1}^{K} \mathbb{I}\left(M_{\mathrm{cluster}}^{(k)} \ge \left|S_c^{\mathrm{obs}}\right|\right)}
   {K + 1}.

Each feature belonging to cluster :math:`c` receives that cluster's corrected p-value in the reported ``corrected_p_values`` map, while features outside all observed clusters retain value 1.

TFCE correction
+++++++++++++++

For TFCE correction, the observed partial-regression t map is transformed into a TFCE score map using MNE's TFCE implementation. The default TFCE control dictionary is ``{"start": 0.0, "step": 0.2}``.

For each permutation :math:`k`, the backend recomputes the partial-regression t map from the permuted residualized data, applies the TFCE transform, and stores the maximum absolute TFCE score over the full channel-time grid:

.. math::

   M_{\mathrm{TFCE}}^{(k)} = \max_f \left| \operatorname{TFCE}\!\left(t_f^{(k)}\right) \right|.

The feature-wise corrected p-value at feature :math:`f` is then estimated by

.. math::

   p_f = \frac{1 + \sum_{k=1}^{K} \mathbb{I}\left(M_{\mathrm{TFCE}}^{(k)} \ge \left|\operatorname{TFCE}\!\left(t_f^{\mathrm{obs}}\right)\right|\right)}
   {K + 1}.

Because the null distribution is built from the maximum absolute TFCE score in each permutation, these are family-wise-error-rate corrected p-values rather than uncorrected feature-wise permutation p-values.

Interpretation
--------------

The practical intent of this workflow is:

1. use the mixed model to estimate and remove group-specific intercept shifts,
2. recover a marginal EEG representation with reduced subject-level nuisance variance,
3. test fixed effects quickly across the full channel-time grid with classical mass-univariate tools.
