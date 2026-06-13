# Nested-Block Permutation Usage Note

Nested-block permutation inference is available in the working tree via:

```python
from lmeeeg.api.infer import permute_fixed_block
```

Use it after fitting a full random-intercept model with stored marginalized EEG:

```python
from lmeeeg.api.fit import fit_lmm_mass_univariate
from lmeeeg.api.infer import permute_fixed_block

fit_result = fit_lmm_mass_univariate(
    eeg=eeg,
    metadata=metadata,
    formula="y ~ condition + latency + condition:latency + (1|subject)",
    variable_types={
        "condition": "categorical",
        "latency": "numeric",
        "subject": "group",
    },
)

result = permute_fixed_block(
    fit_result=fit_result,
    reduced_formula="y ~ latency + (1|subject)",
    correction="cluster",
    n_permutations=1000,
    seed=123,
    tail=1,
    threshold=4.0,
    verbose="info",
)
```

The tested block is the set of fixed-design columns present in the full model but absent from the reduced model. The reduced model is rebuilt with the same design-matrix builder, so factor codings and interactions are handled by design-column names rather than raw metadata column names. The reduced formula must be nested in the full formula; otherwise the code raises an explicit `ValueError`.

Supported corrections:

- `correction="cluster"`: uses raw non-negative partial-F maps with one-tailed cluster correction.
- `correction="tfce"`: uses TFCE on raw non-negative partial-F maps. Optional `tfce_h_power` and `tfce_e_power` are exposed because the inherited defaults were tuned for signed t maps, not validated F maps.

Short-form reduced specifications also work:

```python
permute_fixed_block(fit_result, reduced_formula=["latency"])
permute_fixed_block(fit_result, reduced_formula="latency")
```

Current caveat: this is implemented in the checkout, but the installed site-packages copy may be older. For local tests or another Codex thread, run with:

```bash
PYTHONPATH=src pytest
```

Acceptance check:

```bash
python dev/nested_block_equivalence.py
```

That script verifies the single-column equivalence `F == t**2` and matching corrected significant-feature sets against the existing single-effect cluster path.
