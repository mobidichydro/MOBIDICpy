"""MOBIDIC - Distributed and continuous hydrological balance model."""

import importlib
from typing import TYPE_CHECKING

from mobidic.utils import configure_logger as _configure_logger

__version__ = "0.3.0"

# Lazy imports (PEP 562)
_LAZY_IMPORTS = {
    # Configuration
    "MOBIDICConfig": "mobidic.config",
    "load_config": "mobidic.config",
    # Logging utilities
    "configure_logger": "mobidic.utils",
    "configure_logger_from_config": "mobidic.utils",
    # GIS preprocessing
    "GISData": "mobidic.preprocessing",
    "grid_to_matrix": "mobidic.preprocessing",
    "decimate_raster": "mobidic.preprocessing",
    "decimate_flow_direction": "mobidic.preprocessing",
    "read_shapefile": "mobidic.preprocessing",
    "process_river_network": "mobidic.preprocessing",
    "compute_hillslope_cells": "mobidic.preprocessing",
    "map_hillslope_to_reach": "mobidic.preprocessing",
    "run_preprocessing": "mobidic.preprocessing",
    "load_network": "mobidic.preprocessing",
    "save_network": "mobidic.preprocessing",
    "load_gisdata": "mobidic.preprocessing",
    "save_gisdata": "mobidic.preprocessing",
    "load_reservoirs": "mobidic.preprocessing",
    "save_reservoirs": "mobidic.preprocessing",
    "MeteoData": "mobidic.preprocessing",
    "MeteoRaster": "mobidic.preprocessing",
    "convert_mat_to_netcdf": "mobidic.preprocessing",
    "Reservoir": "mobidic.preprocessing",
    "Reservoirs": "mobidic.preprocessing",
    "process_reservoirs": "mobidic.preprocessing",
    "IDFParameters": "mobidic.preprocessing",
    "HyetographGenerator": "mobidic.preprocessing",
    "read_idf_parameters": "mobidic.preprocessing",
    "read_idf_parameters_resampled": "mobidic.preprocessing",
    "resample_raster_to_grid": "mobidic.preprocessing",
    "idf_depth": "mobidic.preprocessing",
    # Core simulation
    "constants": "mobidic.core",
    "soil_mass_balance": "mobidic.core",
    "capillary_rise": "mobidic.core",
    "hillslope_routing": "mobidic.core",
    "linear_channel_routing": "mobidic.core",
    "groundwater_linear": "mobidic.core",
    "compute_energy_balance_1l": "mobidic.core",
    "diurnal_radiation_cycle": "mobidic.core",
    "energy_balance_1l": "mobidic.core",
    "saturation_specific_humidity": "mobidic.core",
    "solar_hours": "mobidic.core",
    "solar_position": "mobidic.core",
    "Simulation": "mobidic.core",
    "SimulationState": "mobidic.core",
    "SimulationResults": "mobidic.core",
    # I/O
    "load_state": "mobidic.io",
    "StateWriter": "mobidic.io",
    "MeteoWriter": "mobidic.io",
    "save_discharge_report": "mobidic.io",
    "save_lateral_inflow_report": "mobidic.io",
    "load_discharge_report": "mobidic.io",
    # Calibration (optional dependency: pyemu)
    "CalibrationConfig": "mobidic.calibration",
    "CalibrationResults": "mobidic.calibration",
    "PestSetup": "mobidic.calibration",
    "load_calibration_config": "mobidic.calibration",
}

# Names backed by the optional ``calibration`` extra. Accessing one without the
# extra installed yields a helpful AttributeError instead of a raw ImportError.
_CALIBRATION_NAMES = {name for name, mod in _LAZY_IMPORTS.items() if mod == "mobidic.calibration"}

if TYPE_CHECKING:
    # Make static analysers and IDEs aware of the lazily exported symbols.
    from mobidic.calibration import CalibrationConfig
    from mobidic.calibration import CalibrationResults
    from mobidic.calibration import PestSetup
    from mobidic.calibration import load_calibration_config
    from mobidic.config import MOBIDICConfig
    from mobidic.config import load_config
    from mobidic.core import Simulation
    from mobidic.core import SimulationResults
    from mobidic.core import SimulationState
    from mobidic.core import capillary_rise
    from mobidic.core import compute_energy_balance_1l
    from mobidic.core import constants
    from mobidic.core import diurnal_radiation_cycle
    from mobidic.core import energy_balance_1l
    from mobidic.core import groundwater_linear
    from mobidic.core import hillslope_routing
    from mobidic.core import linear_channel_routing
    from mobidic.core import saturation_specific_humidity
    from mobidic.core import soil_mass_balance
    from mobidic.core import solar_hours
    from mobidic.core import solar_position
    from mobidic.io import MeteoWriter
    from mobidic.io import StateWriter
    from mobidic.io import load_discharge_report
    from mobidic.io import load_state
    from mobidic.io import save_discharge_report
    from mobidic.io import save_lateral_inflow_report
    from mobidic.preprocessing import GISData
    from mobidic.preprocessing import HyetographGenerator
    from mobidic.preprocessing import IDFParameters
    from mobidic.preprocessing import MeteoData
    from mobidic.preprocessing import MeteoRaster
    from mobidic.preprocessing import Reservoir
    from mobidic.preprocessing import Reservoirs
    from mobidic.preprocessing import compute_hillslope_cells
    from mobidic.preprocessing import convert_mat_to_netcdf
    from mobidic.preprocessing import decimate_flow_direction
    from mobidic.preprocessing import decimate_raster
    from mobidic.preprocessing import grid_to_matrix
    from mobidic.preprocessing import idf_depth
    from mobidic.preprocessing import load_gisdata
    from mobidic.preprocessing import load_network
    from mobidic.preprocessing import load_reservoirs
    from mobidic.preprocessing import map_hillslope_to_reach
    from mobidic.preprocessing import process_reservoirs
    from mobidic.preprocessing import process_river_network
    from mobidic.preprocessing import read_idf_parameters
    from mobidic.preprocessing import read_idf_parameters_resampled
    from mobidic.preprocessing import read_shapefile
    from mobidic.preprocessing import resample_raster_to_grid
    from mobidic.preprocessing import run_preprocessing
    from mobidic.preprocessing import save_gisdata
    from mobidic.preprocessing import save_network
    from mobidic.preprocessing import save_reservoirs
    from mobidic.utils import configure_logger
    from mobidic.utils import configure_logger_from_config


def __getattr__(name: str):
    """Lazily import and cache a public symbol on first access (PEP 562)."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'mobidic' has no attribute '{name}'")

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if name in _CALIBRATION_NAMES:
            raise AttributeError(
                f"'{name}' requires the calibration extra. Install it with: pip install mobidicpy[calibration]"
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__():
    """Include lazily exported names in ``dir(mobidic)`` for completion."""
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = ["__version__", *sorted(_LAZY_IMPORTS)]

# Configure default logging on import. Users can reconfigure via configure_logger().
_configure_logger(level="INFO")
