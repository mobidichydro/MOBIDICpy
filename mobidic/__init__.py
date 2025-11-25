"""MOBIDIC - Distributed and continuous hydrological balance model."""

__version__ = "0.0.1"

from mobidic.config import MOBIDICConfig
from mobidic.config import load_config
from mobidic.utils import configure_logger, configure_logger_from_config
from mobidic.preprocessing import GISData

from mobidic.preprocessing import (
    grid_to_matrix,
    decimate_raster,
    decimate_flow_direction,
    read_shapefile,
    process_river_network,
    compute_hillslope_cells,
    map_hillslope_to_reach,
    run_preprocessing,
    load_network,
    save_network,
    load_gisdata,
    save_gisdata,
    MeteoData,
    convert_mat_to_netcdf,
)
from mobidic.core import (
    constants,
    soil_mass_balance,
    capillary_rise,
    hillslope_routing,
    linear_channel_routing,
    Simulation,
    SimulationState,
    SimulationResults,
)
from mobidic.io import load_state, StateWriter, save_discharge_report, save_lateral_inflow_report, load_discharge_report

# Configure default logging behavior on package import
# Users can reconfigure by calling configure_logger() with custom settings
configure_logger(level="INFO")

__all__ = [
    "__version__",
    "constants",
    "MOBIDICConfig",
    "load_config",
    "configure_logger",
    "configure_logger_from_config",
    "GISData",
    "grid_to_matrix",
    "decimate_raster",
    "decimate_flow_direction",
    "read_shapefile",
    "process_river_network",
    "compute_hillslope_cells",
    "map_hillslope_to_reach",
    "run_preprocessing",
    "load_network",
    "save_network",
    "load_gisdata",
    "save_gisdata",
    "MeteoData",
    "convert_mat_to_netcdf",
    "soil_mass_balance",
    "capillary_rise",
    "hillslope_routing",
    "linear_channel_routing",
    "Simulation",
    "SimulationState",
    "SimulationResults",
    "load_state",
    "StateWriter",
    "save_discharge_report",
    "save_lateral_inflow_report",
    "load_discharge_report",
]
