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
    compute_metrics,
    kge,
    nse,
    nse_log,
    pbias,
    peak_error,
    rmse,
)
from mobidic.calibration.observation import (
    align_observations_to_simulation,
    load_observations,
)
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
    "nse",
    "nse_log",
    "pbias",
    "peak_error",
    "rmse",
    "kge",
    "compute_metrics",
]
