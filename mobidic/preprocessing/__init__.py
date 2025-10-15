"""MOBIDIC preprocessing module for GIS data processing."""

from mobidic.preprocessing.gis_reader import (
    read_shapefile,
    grid_to_matrix,
)
from mobidic.preprocessing.grid_operations import (
    degrade_flow_direction,
    degrade_raster,
    convert_to_mobidic_notation,
)
from mobidic.preprocessing.river_network import (
    process_river_network,
)
from mobidic.preprocessing.hillslope_reach_mapping import (
    compute_hillslope_cells,
    map_hillslope_to_reach,
)
from mobidic.preprocessing.preprocessor import (
    GISData,
    run_preprocessing,
)
from mobidic.preprocessing.io import (
    load_network,
    save_network,
    load_gisdata,
    save_gisdata,
)

__all__ = [
    "read_shapefile",
    "grid_to_matrix",
    "degrade_flow_direction",
    "degrade_raster",
    "convert_to_mobidic_notation",
    "process_river_network",
    "compute_hillslope_cells",
    "map_hillslope_to_reach",
    "GISData",
    "run_preprocessing",
    "load_network",
    "save_network",
    "load_gisdata",
    "save_gisdata",
]
