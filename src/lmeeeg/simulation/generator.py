from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_CHANNEL_TOPOGRAPHY_WIDTH = 0.18
DEFAULT_CHANNEL_COVARIANCE_DECAY = 0.2
DEFAULT_CHANNEL_COVARIANCE_JITTER = 1e-6
DEFAULT_AR1_RHO = 0.6
DEFAULT_SAMPLING_RATE_HZ = 250.0
DEFAULT_TMIN_S = -0.2
DEFAULT_TMAX_S = 0.6
DEFAULT_NOISE_SD = 1.0


@dataclass(slots=True)
class SimulatedDataset:
    """Container holding the minimal random-intercept dataset."""

    eeg: np.ndarray
    metadata: pd.DataFrame
    ground_truth_effect: np.ndarray


@dataclass(slots=True)
class ERPComponentSpec:
    """Specification for one ERP component.

    Parameters
    ----------
    name : str
        Human-readable component name.
    peak_latency_ms : float
        Peak latency in milliseconds.
    width_ms : float
        Gaussian width parameter in milliseconds.
    polarity : float
        Component polarity. Use ``1.0`` for positive and ``-1.0`` for negative.
    amplitude_intercept : float
        Fixed intercept for component amplitude.
    amplitude_condition_effect : float
        Fixed effect of condition ``B`` relative to ``A`` on component amplitude.
    amplitude_latency_effect : float
        Fixed effect of the numeric ``latency`` covariate on component amplitude.
    trial_amplitude_sd : float
        Standard deviation of trial-to-trial amplitude noise.
    latency_jitter_sd_ms : float
        Standard deviation of trial-wise latency jitter in milliseconds.
    subject_amplitude_sd : float
        Standard deviation of subject-level amplitude deviations.
    topography_center : float
        Center of the smooth channel topography in normalized channel coordinates.
    topography_width : float
        Width of the smooth channel topography in normalized channel coordinates.

    Usage example
    -------------
    >>> ERPComponentSpec(
    ...     name="P100",
    ...     peak_latency_ms=100.0,
    ...     width_ms=25.0,
    ...     polarity=1.0,
    ...     amplitude_intercept=1.4,
    ...     amplitude_condition_effect=0.0,
    ...     topography_center=0.8,
    ... )
    """

    name: str
    peak_latency_ms: float
    width_ms: float
    polarity: float
    amplitude_intercept: float
    amplitude_condition_effect: float
    amplitude_latency_effect: float = 0.0
    trial_amplitude_sd: float = 0.25
    latency_jitter_sd_ms: float = 8.0
    subject_amplitude_sd: float = 0.35
    topography_center: float = 0.5
    topography_width: float = DEFAULT_CHANNEL_TOPOGRAPHY_WIDTH


@dataclass(slots=True)
class ERPSimulationConfig:
    """Configuration for the ERP-style random-intercept simulator."""

    n_subjects: int = 20
    n_trials_per_subject: int = 40
    n_channels: int = 32
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ
    tmin_s: float = DEFAULT_TMIN_S
    tmax_s: float = DEFAULT_TMAX_S
    time_ms: np.ndarray | None = None
    noise_sd: float = DEFAULT_NOISE_SD
    ar1_rho: float = DEFAULT_AR1_RHO
    include_channel_covariance: bool = False
    channel_covariance_decay: float = DEFAULT_CHANNEL_COVARIANCE_DECAY
    component_specs: tuple[ERPComponentSpec, ...] = ()
    seed: int = 0


@dataclass(slots=True)
class ERPSimulationMetadata:
    """Structured metadata returned with an ERP simulation result."""

    config: ERPSimulationConfig
    channel_positions: np.ndarray
    subject_component_amplitude_offsets: np.ndarray
    trial_component_amplitudes: np.ndarray
    trial_component_peak_latencies_ms: np.ndarray
    component_names: tuple[str, ...]
    channel_covariance: np.ndarray | None


@dataclass(slots=True)
class ERPSimulationResult:
    """Container holding the ERP-style random-intercept dataset."""

    eeg: np.ndarray
    metadata: pd.DataFrame
    time_ms: np.ndarray
    ground_truth_beta_condition: np.ndarray
    ground_truth_component_maps: dict[str, np.ndarray]
    simulation_metadata: ERPSimulationMetadata


# ==============================
# Minimal random-intercept simulation
# ==============================

def simulate_random_intercept_dataset(
    n_subjects: int = 12,
    n_trials_per_subject: int = 20,
    n_channels: int = 8,
    n_times: int = 40,
    effect_channels: Iterable[int] = (2, 3),
    effect_times: Iterable[int] = range(10, 18),
    beta: float = 0.8,
    random_intercept_sd: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 0,
) -> SimulatedDataset:
    """Simulate a minimal random-intercept dataset.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_trials_per_subject : int
        Number of trials per subject.
    n_channels : int
        Number of channels.
    n_times : int
        Number of timepoints.
    effect_channels : Iterable[int]
        Channels containing a true effect.
    effect_times : Iterable[int]
        Timepoints containing a true effect.
    beta : float
        Effect size for the condition regressor.
    random_intercept_sd : float
        Standard deviation of the subject random intercept.
    noise_sd : float
        Standard deviation of feature-wise noise.
    seed : int
        Random seed.

    Returns
    -------
    SimulatedDataset
        Simulated dataset.

    Usage example
    -------------
        simulated = simulate_random_intercept_dataset(seed=13)
    """
    rng = np.random.default_rng(seed)
    n_observations = n_subjects * n_trials_per_subject

    subject_ids = np.repeat(np.arange(n_subjects), n_trials_per_subject)
    condition_codes = np.tile(np.repeat([0, 1], repeats=n_trials_per_subject // 2), n_subjects)
    if len(condition_codes) < n_observations:
        condition_codes = np.concatenate(
            [condition_codes, rng.integers(0, 2, size=n_observations - len(condition_codes))]
        )
    condition_codes = condition_codes[:n_observations]
    latency = rng.normal(loc=0.0, scale=1.0, size=n_observations)

    metadata = pd.DataFrame(
        {
            "subject": [f"sub-{subject:03d}" for subject in subject_ids],
            "condition": np.where(condition_codes == 0, "A", "B"),
            "latency": latency,
        }
    )

    ground_truth_effect = np.zeros((n_channels, n_times), dtype=float)
    valid_effect_channels = [index for index in effect_channels if 0 <= index < n_channels]
    valid_effect_times = [index for index in effect_times if 0 <= index < n_times]
    for channel_index in valid_effect_channels:
        for time_index in valid_effect_times:
            ground_truth_effect[channel_index, time_index] = beta

    subject_intercepts = rng.normal(loc=0.0, scale=random_intercept_sd, size=n_subjects)
    eeg = np.zeros((n_observations, n_channels, n_times), dtype=float)
    for observation_index in range(n_observations):
        subject_index = subject_ids[observation_index]
        eeg[observation_index, :, :] += subject_intercepts[subject_index]
        eeg[observation_index, :, :] += condition_codes[observation_index] * ground_truth_effect
        eeg[observation_index, :, :] += 0.2 * latency[observation_index]
        eeg[observation_index, :, :] += rng.normal(loc=0.0, scale=noise_sd, size=(n_channels, n_times))

    return SimulatedDataset(
        eeg=eeg,
        metadata=metadata,
        ground_truth_effect=ground_truth_effect,
    )


# ==============================
# ERP-style random-intercept simulation
# ==============================

def simulate_erp_random_intercept_dataset(
    n_subjects: int = 20,
    n_trials_per_subject: int = 40,
    n_channels: int = 32,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    tmin_s: float = DEFAULT_TMIN_S,
    tmax_s: float = DEFAULT_TMAX_S,
    time_ms: np.ndarray | None = None,
    noise_sd: float = DEFAULT_NOISE_SD,
    ar1_rho: float = DEFAULT_AR1_RHO,
    include_channel_covariance: bool = False,
    channel_covariance_decay: float = DEFAULT_CHANNEL_COVARIANCE_DECAY,
    component_specs: Sequence[ERPComponentSpec] | None = None,
    seed: int = 0,
) -> ERPSimulationResult:
    """Simulate ERP-like trial-wise EEG for a random-intercept lmeEEG workflow.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_trials_per_subject : int
        Number of trials per subject.
    n_channels : int
        Number of channels.
    sampling_rate_hz : float
        Sampling rate used when ``time_ms`` is not provided.
    tmin_s : float
        Start time in seconds when ``time_ms`` is not provided.
    tmax_s : float
        End time in seconds when ``time_ms`` is not provided.
    time_ms : np.ndarray | None
        Optional explicit time vector in milliseconds.
    noise_sd : float
        Standard deviation of additive correlated noise.
    ar1_rho : float
        AR(1) coefficient controlling temporal autocorrelation.
    include_channel_covariance : bool
        Whether to impose smooth channel covariance on innovations.
    channel_covariance_decay : float
        Distance decay for the optional channel covariance.
    component_specs : Sequence[ERPComponentSpec] | None
        ERP components to include. If omitted, uses P100 and N200 defaults.
    seed : int
        Random seed.

    Returns
    -------
    ERPSimulationResult
        ERP-style simulation result.

    Notes
    -----
    The returned ``metadata`` DataFrame contains one row per observation with columns:

    +-------------------+-----------+--------------------------------------+
    | column            | dtype     | meaning                              |
    +===================+===========+======================================+
    | subject           | object    | subject label such as ``sub-003``    |
    +-------------------+-----------+--------------------------------------+
    | condition         | object    | categorical condition ``A`` or ``B`` |
    +-------------------+-----------+--------------------------------------+
    | latency           | float     | numeric trial covariate              |
    +-------------------+-----------+--------------------------------------+
    | trial_index       | int       | within-subject trial index           |
    +-------------------+-----------+--------------------------------------+
    | observation_index | int       | global trial index                   |
    +-------------------+-----------+--------------------------------------+

    Usage example
    -------------
        simulation = simulate_erp_random_intercept_dataset(
            n_subjects=20,
            n_trials_per_subject=40,
            n_channels=32,
            sampling_rate_hz=250.0,
            tmin_s=-0.2,
            tmax_s=0.6,
            include_channel_covariance=True,
            noise_sd=1.0,
            ar1_rho=0.6,
            seed=7,
        )
    """
    from lmeeeg.simulation.scenarios import build_default_erp_component_specs

    if n_subjects <= 0:
        raise ValueError("n_subjects must be positive.")
    if n_trials_per_subject <= 0:
        raise ValueError("n_trials_per_subject must be positive.")
    if n_channels <= 0:
        raise ValueError("n_channels must be positive.")
    if not (-0.99 < ar1_rho < 0.99):
        raise ValueError("ar1_rho must be between -0.99 and 0.99.")
    if channel_covariance_decay <= 0.0:
        raise ValueError("channel_covariance_decay must be positive.")

    resolved_component_specs = tuple(component_specs or build_default_erp_component_specs())
    if len(resolved_component_specs) == 0:
        raise ValueError("At least one ERP component specification is required.")

    config = ERPSimulationConfig(
        n_subjects=n_subjects,
        n_trials_per_subject=n_trials_per_subject,
        n_channels=n_channels,
        sampling_rate_hz=sampling_rate_hz,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        time_ms=None if time_ms is None else np.asarray(time_ms, dtype=float),
        noise_sd=noise_sd,
        ar1_rho=ar1_rho,
        include_channel_covariance=include_channel_covariance,
        channel_covariance_decay=channel_covariance_decay,
        component_specs=resolved_component_specs,
        seed=seed,
    )
    return _simulate_erp_random_intercept_from_config(config=config)


def _simulate_erp_random_intercept_from_config(config: ERPSimulationConfig) -> ERPSimulationResult:
    """Internal implementation for the ERP-style random-intercept simulator."""
    rng = np.random.default_rng(config.seed)

    time_ms = _build_time_vector_ms(
        time_ms=config.time_ms,
        sampling_rate_hz=config.sampling_rate_hz,
        tmin_s=config.tmin_s,
        tmax_s=config.tmax_s,
    )
    channel_positions = _build_channel_positions(n_channels=config.n_channels)
    channel_covariance = None
    if config.include_channel_covariance:
        channel_covariance = _build_channel_covariance(
            channel_positions=channel_positions,
            decay=config.channel_covariance_decay,
        )

    metadata = _build_metadata(
        n_subjects=config.n_subjects,
        n_trials_per_subject=config.n_trials_per_subject,
        rng=rng,
    )
    n_observations = len(metadata)
    n_components = len(config.component_specs)
    n_times = len(time_ms)

    component_topographies = np.stack(
        [
            _gaussian_channel_topography(
                channel_positions=channel_positions,
                center=component_spec.topography_center,
                width=component_spec.topography_width,
            )
            for component_spec in config.component_specs
        ],
        axis=0,
    )

    subject_component_offsets = np.stack(
        [
            rng.normal(loc=0.0, scale=component_spec.subject_amplitude_sd, size=config.n_subjects)
            for component_spec in config.component_specs
        ],
        axis=1,
    )

    trial_component_amplitudes = np.zeros((n_observations, n_components), dtype=float)
    trial_component_peak_latencies_ms = np.zeros((n_observations, n_components), dtype=float)
    fixed_signal = np.zeros((n_observations, config.n_channels, n_times), dtype=float)
    subject_signal = np.zeros_like(fixed_signal)
    trial_noise_signal = np.zeros_like(fixed_signal)
    for observation_index, row in metadata.iterrows():
        subject_index = int(row["subject_code"])
        condition_code = int(row["condition_code"])
        latency_value = float(row["latency"])

        for component_index, component_spec in enumerate(config.component_specs):
            jitter_ms = rng.normal(loc=0.0, scale=component_spec.latency_jitter_sd_ms)
            peak_latency_ms = component_spec.peak_latency_ms + jitter_ms
            temporal_waveform = _gaussian_erp_waveform(
                time_ms=time_ms,
                peak_latency_ms=peak_latency_ms,
                width_ms=component_spec.width_ms,
                polarity=component_spec.polarity,
            )
            component_pattern = np.outer(component_topographies[component_index], temporal_waveform)

            fixed_amplitude = (
                component_spec.amplitude_intercept
                + component_spec.amplitude_condition_effect * condition_code
                + component_spec.amplitude_latency_effect * latency_value
            )
            subject_amplitude = subject_component_offsets[subject_index, component_index]
            trial_amplitude_noise = rng.normal(loc=0.0, scale=component_spec.trial_amplitude_sd)
            total_amplitude = fixed_amplitude + subject_amplitude + trial_amplitude_noise

            fixed_contribution = fixed_amplitude * component_pattern
            subject_contribution = subject_amplitude * component_pattern
            trial_noise_contribution = trial_amplitude_noise * component_pattern

            fixed_signal[observation_index] += fixed_contribution
            subject_signal[observation_index] += subject_contribution
            trial_noise_signal[observation_index] += trial_noise_contribution

            trial_component_amplitudes[observation_index, component_index] = total_amplitude
            trial_component_peak_latencies_ms[observation_index, component_index] = peak_latency_ms

    noise = _sample_correlated_noise(
        n_observations=n_observations,
        n_channels=config.n_channels,
        n_times=n_times,
        noise_sd=config.noise_sd,
        ar1_rho=config.ar1_rho,
        rng=rng,
        channel_covariance=channel_covariance,
    )
    eeg = fixed_signal + subject_signal + trial_noise_signal + noise

    ground_truth_component_maps = {
        component_spec.name: _compute_component_condition_beta_map(
            component_spec=component_spec,
            channel_positions=channel_positions,
            time_ms=time_ms,
        )
        for component_spec in config.component_specs
    }
    ground_truth_beta_condition = np.sum(
        np.stack(list(ground_truth_component_maps.values()), axis=0),
        axis=0,
    )

    output_metadata = metadata.drop(columns=["subject_code", "condition_code"]).copy()
    simulation_metadata = ERPSimulationMetadata(
        config=config,
        channel_positions=channel_positions,
        subject_component_amplitude_offsets=subject_component_offsets,
        trial_component_amplitudes=trial_component_amplitudes,
        trial_component_peak_latencies_ms=trial_component_peak_latencies_ms,
        component_names=tuple(component_spec.name for component_spec in config.component_specs),
        channel_covariance=channel_covariance,
    )
    return ERPSimulationResult(
        eeg=eeg,
        metadata=output_metadata,
        time_ms=time_ms,
        ground_truth_beta_condition=ground_truth_beta_condition,
        ground_truth_component_maps=ground_truth_component_maps,
        simulation_metadata=simulation_metadata,
    )


# ==============================
# ERP helper functions
# ==============================

def _build_metadata(
    n_subjects: int,
    n_trials_per_subject: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create observation-level metadata for ERP simulations."""
    n_observations = n_subjects * n_trials_per_subject
    subject_codes = np.repeat(np.arange(n_subjects, dtype=int), n_trials_per_subject)
    trial_indices = np.tile(np.arange(n_trials_per_subject, dtype=int), n_subjects)

    condition_codes = np.concatenate(
        [
            _generate_balanced_condition_codes(n_trials=n_trials_per_subject, rng=rng)
            for _ in range(n_subjects)
        ]
    )
    latency_covariate = rng.normal(loc=0.0, scale=1.0, size=n_observations)

    return pd.DataFrame(
        {
            "subject": [f"sub-{subject_index:03d}" for subject_index in subject_codes],
            "condition": np.where(condition_codes == 0, "A", "B"),
            "latency": latency_covariate,
            "trial_index": trial_indices,
            "observation_index": np.arange(n_observations, dtype=int),
            "subject_code": subject_codes,
            "condition_code": condition_codes,
        }
    )


def _generate_balanced_condition_codes(n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Generate approximately balanced binary condition codes for one subject."""
    n_condition_b = n_trials // 2
    n_condition_a = n_trials - n_condition_b
    codes = np.concatenate(
        [
            np.zeros(n_condition_a, dtype=int),
            np.ones(n_condition_b, dtype=int),
        ]
    )
    rng.shuffle(codes)
    return codes


def _build_time_vector_ms(
    time_ms: np.ndarray | None,
    sampling_rate_hz: float,
    tmin_s: float,
    tmax_s: float,
) -> np.ndarray:
    """Create the ERP time vector in milliseconds."""
    if time_ms is not None:
        resolved_time_ms = np.asarray(time_ms, dtype=float)
        if resolved_time_ms.ndim != 1:
            raise ValueError("time_ms must be a one-dimensional array.")
        if len(resolved_time_ms) == 0:
            raise ValueError("time_ms must not be empty.")
        return resolved_time_ms

    if sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be positive.")
    if tmax_s <= tmin_s:
        raise ValueError("tmax_s must be greater than tmin_s.")

    sampling_interval_ms = 1000.0 / sampling_rate_hz
    n_times = int(np.round((tmax_s - tmin_s) * sampling_rate_hz)) + 1
    return np.arange(n_times, dtype=float) * sampling_interval_ms + tmin_s * 1000.0


def _build_channel_positions(n_channels: int) -> np.ndarray:
    """Create normalized one-dimensional channel positions."""
    if n_channels == 1:
        return np.array([0.5], dtype=float)
    return np.linspace(0.0, 1.0, n_channels, dtype=float)


def _gaussian_erp_waveform(
    time_ms: np.ndarray,
    peak_latency_ms: float,
    width_ms: float,
    polarity: float,
) -> np.ndarray:
    """Generate a smooth Gaussian ERP waveform."""
    if width_ms <= 0.0:
        raise ValueError("width_ms must be positive.")
    squared_distance = (time_ms - peak_latency_ms) ** 2
    return float(polarity) * np.exp(-0.5 * squared_distance / (width_ms ** 2))


def _gaussian_channel_topography(
    channel_positions: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """Generate a smooth Gaussian channel topography."""
    if width <= 0.0:
        raise ValueError("width must be positive.")
    squared_distance = (channel_positions - center) ** 2
    topography = np.exp(-0.5 * squared_distance / (width ** 2))
    return topography / np.max(topography)


def _build_channel_covariance(
    channel_positions: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Create a smooth channel covariance matrix from channel distance."""
    pairwise_distance = np.abs(channel_positions[:, None] - channel_positions[None, :])
    covariance = np.exp(-0.5 * (pairwise_distance / decay) ** 2)
    covariance += np.eye(len(channel_positions), dtype=float) * DEFAULT_CHANNEL_COVARIANCE_JITTER
    return covariance


def _sample_correlated_noise(
    n_observations: int,
    n_channels: int,
    n_times: int,
    noise_sd: float,
    ar1_rho: float,
    rng: np.random.Generator,
    channel_covariance: np.ndarray | None,
) -> np.ndarray:
    """Sample additive noise with temporal AR(1) correlation and optional channel covariance."""
    if noise_sd < 0.0:
        raise ValueError("noise_sd must be non-negative.")

    if channel_covariance is None:
        channel_transform = np.eye(n_channels, dtype=float)
    else:
        channel_transform = np.linalg.cholesky(channel_covariance)

    innovations = rng.normal(loc=0.0, scale=1.0, size=(n_observations, n_channels, n_times))
    innovations = np.einsum("ij,okj->oki", channel_transform, innovations.transpose(0, 2, 1)).transpose(0, 2, 1)

    noise = np.zeros_like(innovations)
    scale = np.sqrt(1.0 - ar1_rho ** 2)
    noise[:, :, 0] = innovations[:, :, 0]
    for time_index in range(1, n_times):
        noise[:, :, time_index] = ar1_rho * noise[:, :, time_index - 1] + scale * innovations[:, :, time_index]
    return noise_sd * noise


def _expected_gaussian_erp_waveform(
    time_ms: np.ndarray,
    peak_latency_ms: float,
    width_ms: float,
    jitter_sd_ms: float,
    polarity: float,
) -> np.ndarray:
    """Compute the expected ERP waveform under Gaussian latency jitter."""
    effective_variance = width_ms ** 2 + jitter_sd_ms ** 2
    amplitude_scale = width_ms / np.sqrt(effective_variance)
    squared_distance = (time_ms - peak_latency_ms) ** 2
    return float(polarity) * amplitude_scale * np.exp(-0.5 * squared_distance / effective_variance)


def _compute_component_condition_beta_map(
    component_spec: ERPComponentSpec,
    channel_positions: np.ndarray,
    time_ms: np.ndarray,
) -> np.ndarray:
    """Compute the analytic condition beta map contributed by one component."""
    topography = _gaussian_channel_topography(
        channel_positions=channel_positions,
        center=component_spec.topography_center,
        width=component_spec.topography_width,
    )
    temporal_waveform = _expected_gaussian_erp_waveform(
        time_ms=time_ms,
        peak_latency_ms=component_spec.peak_latency_ms,
        width_ms=component_spec.width_ms,
        jitter_sd_ms=component_spec.latency_jitter_sd_ms,
        polarity=component_spec.polarity,
    )
    return component_spec.amplitude_condition_effect * np.outer(topography, temporal_waveform)
