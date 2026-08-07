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
    DAConfig,
    DAStateConfig,
    MetricConfig,
    ObservationGroup,
    load_calibration_config,
)
from mobidic.calibration.da_cycles import (
    CycleSchedule,
    build_cycle_schedule,
    build_observation_cycle_tables,
    build_parameter_cycle_table,
    write_cycle_metadata,
)
from mobidic.calibration.da_states import (
    StateBlock,
    StateSpec,
    build_discharge_state_spec,
    build_reach_zone_map,
    build_soil_state_spec,
    build_state_mask,
    build_surface_state_spec,
    insert_state_vector,
    extract_state_vector,
    new_state_id,
    read_state_file,
    remove_old_state_files,
    rescale_to_zone_saturation,
    rescale_zone_field,
    resolve_estimate_kinds,
    soil_capacities,
    upstream_reaches,
    write_state_file,
    zone_mean,
    zone_saturation,
)
from mobidic.calibration.metrics import (
    METRIC_REGISTRY,
    compute_metrics,
)
from mobidic.calibration.forward_model import prepare_simulation
from mobidic.calibration.observation import (
    align_observations_to_simulation,
    load_observations,
    observation_weights,
)
from mobidic.calibration.parameter_mapping import apply_optimal_parameters, apply_parameters_to_yaml
from mobidic.calibration.pest_setup import PestSetup
from mobidic.calibration.results import CalibrationResults

__all__ = [
    # Config
    "CalibrationConfig",
    "CalibrationParameter",
    "CalibrationPeriod",
    "DAConfig",
    "DAStateConfig",
    "MetricConfig",
    "ObservationGroup",
    "load_calibration_config",
    # Setup and results
    "PestSetup",
    "CalibrationResults",
    # Data assimilation: cycles
    "CycleSchedule",
    "build_cycle_schedule",
    "build_observation_cycle_tables",
    "build_parameter_cycle_table",
    "write_cycle_metadata",
    # Data assimilation: states
    "StateBlock",
    "StateSpec",
    "build_discharge_state_spec",
    "build_reach_zone_map",
    "build_soil_state_spec",
    "build_state_mask",
    "build_surface_state_spec",
    "insert_state_vector",
    "extract_state_vector",
    "new_state_id",
    "read_state_file",
    "remove_old_state_files",
    "rescale_to_zone_saturation",
    "rescale_zone_field",
    "resolve_estimate_kinds",
    "soil_capacities",
    "upstream_reaches",
    "write_state_file",
    "zone_mean",
    "zone_saturation",
    # Observations
    "load_observations",
    "align_observations_to_simulation",
    "observation_weights",
    "prepare_simulation",
    # Metrics
    "compute_metrics",
    "METRIC_REGISTRY",
    # Parameter mapping and export
    "apply_optimal_parameters",
    "apply_parameters_to_yaml",
]
