"""PEST++ calibration interface for MOBIDICpy.

This package provides tools for model calibration, sensitivity analysis,
and uncertainty quantification using PEST++ via pyemu.

Requires calibration dependencies: pyemu, HydroErr
Install with: pip install mobidic[calibration]
"""

from mobidic.calibration.config import (
    CalibrationConfig,
    CalibrationParameter,
    CalibrationPeriod,
    MetricConfig,
    ObservationGroup,
    load_calibration_config,
)
from mobidic.calibration.metrics import (
    METRIC_REGISTRY,
    compute_metrics,
)
from mobidic.calibration.observation import (
    align_observations_to_simulation,
    load_observations,
)
from mobidic.calibration.parameter_mapping import apply_optimal_parameters, apply_parameters_to_yaml
from mobidic.calibration.pest_setup import PestSetup
from mobidic.calibration.results import CalibrationResults

__all__ = [
    # Config
    "CalibrationConfig",
    "CalibrationParameter",
    "CalibrationPeriod",
    "MetricConfig",
    "ObservationGroup",
    "load_calibration_config",
    # Setup and results
    "PestSetup",
    "CalibrationResults",
    # Observations
    "load_observations",
    "align_observations_to_simulation",
    # Metrics
    "compute_metrics",
    "METRIC_REGISTRY",
    # Parameter mapping and export
    "apply_optimal_parameters",
    "apply_parameters_to_yaml",
]
