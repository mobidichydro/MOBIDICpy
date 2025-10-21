"""MOBIDIC - Distributed and continuous hydrological balance model."""

__version__ = "0.0.1"

from mobidic.config import MOBIDICConfig
from mobidic.config import load_config
from mobidic.utils import configure_logger
from mobidic.preprocessing import GISData

from mobidic.preprocessing import (
    grid_to_matrix,
    degrade_raster,
    degrade_flow_direction,
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
from mobidic.core import soil_mass_balance, capillary_rise

# Configure default logging behavior on package import
# Users can reconfigure by calling configure_logger() with custom settings
configure_logger(level="INFO")

__all__ = [
    "__version__",
    "MOBIDICConfig",
    "load_config",
    "configure_logger",
    "GISData",
    "grid_to_matrix",
    "degrade_raster",
    "degrade_flow_direction",
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
]
