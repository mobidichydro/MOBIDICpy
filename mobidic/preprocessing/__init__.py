"""MOBIDIC preprocessing module for GIS data processing."""

from mobidic.preprocessing.gis_reader import (
    read_shapefile,
    read_raster,
    grid_to_matrix,
)
from mobidic.preprocessing.grid_operations import (
    degrade_flow_direction,
    degrade_raster,
    convert_to_mobidic_notation,
)
from mobidic.preprocessing.river_network import (
    export_network,
    load_network,
    process_river_network,
)
from mobidic.preprocessing.hillslope_reach_mapping import (
    compute_hillslope_cells,
    map_hillslope_to_reach,
)

__all__ = [
    "read_shapefile",
    "read_raster",
    "grid_to_matrix",
    "degrade_raster",
    "degrade_flow_direction",
    "convert_to_mobidic_notation",
    "process_river_network",
    "export_network",
    "load_network",
    "compute_hillslope_cells",
    "map_hillslope_to_reach",
]
