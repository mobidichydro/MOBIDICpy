"""MOBIDIC - Distributed and continuous hydrological balance model."""

__version__ = "0.0.1"

from mobidic.config import MOBIDICConfig, load_config

__all__ = ["__version__", "load_config", "MOBIDICConfig"]
