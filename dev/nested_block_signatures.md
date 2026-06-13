# Nested-block permutation discovery notes

Reused signatures discovered before implementation:

- `StatsModelsLMMBackend.fit_mass_univariate(self, eeg: np.ndarray, metadata: pd.DataFrame, design_spec: DesignSpec, show_progress: bool = True, store_fitted_random_effects: bool = False, store_marginal_eeg: bool = True, output_dtype: np.dtype | None = None) -> LMMBackendResult`
- `compute_marginal_eeg(eeg: np.ndarray, fitted_random_effects: np.ndarray) -> np.ndarray`
- `prepare_effect_regression(fit_result: FitResult, effect: str) -> dict[str, np.ndarray | int]`
- `permute_within_groups(y_residualized: np.ndarray, group_codes: np.ndarray, rng: np.random.Generator) -> np.ndarray`
- `MNEClusterCorrectionBackend.run(self, fit_result: FitResult, effect: str, n_permutations: int, seed: int, tail: int, threshold: float | dict[str, float] | None, adjacency, verbose: bool | str | int | None = "info", spatial_chunk_size: int | None = None, time_chunk_size: int | None = None, store_null_maps: bool = False) -> InferenceResult`
- `MNETFCorrectionBackend.run(self, fit_result: FitResult, effect: str, n_permutations: int, seed: int, tail: int, threshold: float | dict[str, float] | None, adjacency, verbose: bool | str | int | None = "info", spatial_chunk_size: int | None = None, time_chunk_size: int | None = None, store_null_maps: bool = False) -> InferenceResult`

The single-effect path takes `effect` as an exact fixed-design column name and treats every other fixed column as nuisance. The nested-block path takes a reduced model, rebuilds it with `build_design_spec`, checks that reduced columns are present in the full fixed design with identical encoded values, and tests the full-minus-reduced column block with a partial F statistic.

Cluster and TFCE currently consume statistic maps shaped `(n_locations, n_times)` and transpose them to `(n_times, n_locations)` for MNE cluster internals. The existing single-effect cluster/TFCE paths consume signed partial-t maps and use absolute values for the two-sided max-stat null. The nested-block cluster/TFCE paths consume raw non-negative partial-F maps, require `tail=1`, and do not apply absolute-value handling to the F null or F p-value comparisons.

New public entry point:

- `permute_fixed_block(fit_result: FitResult, reduced_formula: str | list[str], correction: str = "cluster", n_permutations: int = 1000, seed: int = 0, tail: int = 1, threshold: float | dict[str, float] | None = None, adjacency=None, verbose: bool | str | int | None = "info", spatial_chunk_size: int | None = None, time_chunk_size: int | None = None, store_null_maps: bool = False, tfce_h_power: float = 2.0, tfce_e_power: float = 0.5) -> InferenceResult`
