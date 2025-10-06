"""MOBIDIC - Distributed and continuous hydrological balance model."""

__version__ = "0.0.1"

from mobidic.config import MOBIDICConfig
from mobidic.config import load_config
from mobidic.preprocessing import read_raster
from mobidic.preprocessing import read_shapefile

__all__ = ["__version__", "load_config", "MOBIDICConfig", "read_raster", "read_shapefile"]
