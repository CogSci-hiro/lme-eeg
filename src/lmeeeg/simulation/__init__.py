"""Simulation tools."""

from lmeeeg.simulation.generator import (
    ERPComponentSpec,
    ERPSimulationConfig,
    ERPSimulationMetadata,
    ERPSimulationResult,
    SimulatedDataset,
    simulate_erp_random_intercept_dataset,
    simulate_random_intercept_dataset,
)
from lmeeeg.simulation.scenarios import build_default_erp_component_specs

__all__ = [
    "ERPComponentSpec",
    "ERPSimulationConfig",
    "ERPSimulationMetadata",
    "ERPSimulationResult",
    "SimulatedDataset",
    "build_default_erp_component_specs",
    "simulate_erp_random_intercept_dataset",
    "simulate_random_intercept_dataset",
]
