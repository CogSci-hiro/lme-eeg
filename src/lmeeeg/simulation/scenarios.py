from __future__ import annotations

from lmeeeg.simulation.generator import ERPComponentSpec


def build_default_erp_component_specs() -> tuple[ERPComponentSpec, ERPComponentSpec]:
    """Return the default canonical ERP component specifications.

    Returns
    -------
    tuple[ERPComponentSpec, ERPComponentSpec]
        Default P100 and N200 component specifications.

    Usage example
    -------------
        component_specs = build_default_erp_component_specs()
    """
    return (
        ERPComponentSpec(
            name="P100",
            peak_latency_ms=100.0,
            width_ms=22.0,
            polarity=1.0,
            amplitude_intercept=1.6,
            amplitude_condition_effect=0.0,
            amplitude_latency_effect=0.05,
            trial_amplitude_sd=0.20,
            latency_jitter_sd_ms=8.0,
            subject_amplitude_sd=0.30,
            topography_center=0.80,
            topography_width=0.16,
        ),
        ERPComponentSpec(
            name="N200",
            peak_latency_ms=200.0,
            width_ms=32.0,
            polarity=-1.0,
            amplitude_intercept=1.2,
            amplitude_condition_effect=0.9,
            amplitude_latency_effect=-0.08,
            trial_amplitude_sd=0.25,
            latency_jitter_sd_ms=12.0,
            subject_amplitude_sd=0.35,
            topography_center=0.45,
            topography_width=0.18,
        ),
    )


__all__ = ["build_default_erp_component_specs"]
