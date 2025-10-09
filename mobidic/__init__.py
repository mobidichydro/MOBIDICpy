"""MOBIDIC - Distributed and continuous hydrological balance model."""

__version__ = "0.0.1"

from mobidic.config import MOBIDICConfig
from mobidic.config import load_config
from mobidic.utils import configure_logger
from mobidic.preprocessing import (
    read_raster,
    grid_to_matrix,
    degrade_raster,
    degrade_flow_direction,
    read_shapefile,
    load_network,
    process_river_network,
    export_network,
    compute_hillslope_cells,
    map_hillslope_to_reach,
)

__all__ = [
    "__version__",
    "load_config",
    "MOBIDICConfig",
    "configure_logger",
    "read_raster",
    "grid_to_matrix",
    "degrade_raster",
    "degrade_flow_direction",
    "read_shapefile",
    "load_network",
    "process_river_network",
    "export_network",
    "compute_hillslope_cells",
    "map_hillslope_to_reach",
]
