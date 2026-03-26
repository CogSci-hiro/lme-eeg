from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SimulatedDataset:
    """Container holding a simulated dataset."""

    eeg: np.ndarray
    metadata: pd.DataFrame
    ground_truth_effect: np.ndarray


# ==============================
# Random-intercept simulation
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
        condition_codes = np.concatenate([condition_codes, rng.integers(0, 2, size=n_observations - len(condition_codes))])
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
