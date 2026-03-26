from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmeeeg.core.results import FitResult, InferenceResult


class BaseCorrectionBackend(ABC):
    """Abstract base class for correction backends."""

    @abstractmethod
    def run(
        self,
        fit_result: FitResult,
        effect: str,
        n_permutations: int,
        seed: int,
        tail: int,
        threshold: float | dict[str, float] | None,
        adjacency: Any,
    ) -> InferenceResult:
        """Run correction backend."""
