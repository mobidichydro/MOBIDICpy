"""MOBIDIC - Distributed and continuous hydrological balance model."""

__version__ = "0.0.1"

from mobidic.config import MOBIDICConfig
from mobidic.config import load_config
from mobidic.preprocessing import (
    read_raster,
    degrade_raster,
    degrade_flow_direction,
    convert_flow_direction,
    read_shapefile,
    load_network,
    process_river_network,
    export_network,
)

__all__ = [
    "__version__",
    "load_config",
    "MOBIDICConfig",
    "read_raster",
    "degrade_raster",
    "degrade_flow_direction",
    "convert_flow_direction",
    "read_shapefile",
    "load_network",
    "process_river_network",
    "export_network",
]
