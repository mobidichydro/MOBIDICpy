"""Meteorological raster data handling.

This module handles loading and accessing gridded meteorological forcing data
from NetCDF raster files for use in MOBIDIC simulations.

The module provides:
- Raster reader for CF-compliant NetCDF files with dimensions (time, y, x)
- Grid alignment validation against model grid
- Preloading option: loads all data into memory for fast access (default)
- Lazy loading option: reads data on-demand for lower memory usage
- Single-timestep caching to avoid redundant reads
- Time indexing using nearest neighbor sampling

Performance:
- Default behavior (preload=True) loads entire NetCDF into memory at initialization (faster)
- Use preload=False for very large datasets that don't fit in memory (slower)
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mobidic.utils.crs import crs_equals, get_epsg_code


class MeteoRaster:
    """Container for gridded meteorological data from rasters.

    Loads CF-1.12 compliant NetCDF files with dimensions (time, y, x).
    By default, preloads all data into memory for optimal performance.

    Attributes:
        nc_path: Path to NetCDF file
        ds: xarray Dataset (preloaded by default, or lazy-loaded if preload=False)
        variables: List of available meteorological variables
        time_range: Tuple of (start_time, end_time)
        start_date: First timestamp in the dataset (property)
        end_date: Last timestamp in the dataset (property)
        grid_metadata: Dictionary with shape, resolution, crs, origin

    Performance:
        Default preload=True: Loads entire dataset into memory at initialization.
        This provides fast access during simulation (comparable to station interpolation).
        Use preload=False for very large datasets to use lazy loading (slower but less memory).
    """

    def __init__(self, nc_path: str | Path, preload: bool = True):
        """Initialize from NetCDF file.

        Args:
            nc_path: Path to NetCDF file containing raster meteorological data
            preload: If True, load all data into memory immediately (default: True).
                If False, use lazy loading (slower but uses less memory).

        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the NetCDF file has invalid structure
        """
        self.nc_path = Path(nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"Meteorological raster file not found: {self.nc_path}")

        logger.info(f"Loading meteorological rasters from: {self.nc_path}")

        # Load NetCDF file with xarray
        if preload:
            # Load all data into memory for fast access
            logger.debug("Preloading all meteorological data into memory...")
            self.ds = xr.open_dataset(self.nc_path).load()
            logger.debug("Preloading complete")
        else:
            # Use lazy loading (slower but uses less memory)
            logger.debug("Using lazy loading (data read on-demand from disk)")
            self.ds = xr.open_dataset(self.nc_path)

        # Validate structure
        self._validate_structure()

        # Extract metadata
        self._extract_metadata()

        # Initialize cache for current timestep (less useful when preloaded, but kept for compatibility)
        self._cache: dict[tuple[str, str], np.ndarray] = {}

        logger.success(
            f"Loaded meteorological rasters: {len(self.variables)} variables, "
            f"{len(self.ds.time)} timesteps, grid={self.grid_metadata['shape']}"
        )

    def _validate_structure(self) -> None:
        """Validate that NetCDF has required dimensions and structure.

        Raises:
            ValueError: If structure is invalid
        """
        # Check required dimensions
        if "time" not in self.ds.dims:
            raise ValueError("NetCDF file must have 'time' dimension")
        if "y" not in self.ds.dims:
            raise ValueError("NetCDF file must have 'y' dimension")
        if "x" not in self.ds.dims:
            raise ValueError("NetCDF file must have 'x' dimension")

        # Check that we have at least one data variable (besides crs)
        data_vars = [v for v in self.ds.data_vars if v != "crs"]
        if len(data_vars) == 0:
            raise ValueError("NetCDF file must have at least one meteorological variable")

        logger.debug(f"NetCDF structure validated: {self.ds.dims}")

    def _extract_metadata(self) -> None:
        """Extract grid metadata from NetCDF file."""
        # Extract variables (exclude 'crs' coordinate variable)
        self.variables = [v for v in self.ds.data_vars if v != "crs"]

        # Extract time range
        times = self.ds.time.values
        self.time_range = (times[0], times[-1])

        # Extract grid metadata
        nrows = len(self.ds.y)
        ncols = len(self.ds.x)

        # Calculate resolution (assumes regular grid)
        if nrows > 1:
            y_res = abs(float(self.ds.y[1] - self.ds.y[0]))
        else:
            y_res = None

        if ncols > 1:
            x_res = abs(float(self.ds.x[1] - self.ds.x[0]))
        else:
            x_res = None

        # Check that resolution is consistent
        if y_res is not None and x_res is not None and not np.isclose(y_res, x_res, rtol=1e-6):
            logger.warning(f"Grid has different x and y resolution: x={x_res}, y={y_res}")

        resolution = x_res if x_res is not None else y_res

        # Extract origin (lower-left corner)
        # y coordinate is typically in descending order (north to south)
        y_values = self.ds.y.values
        x_values = self.ds.x.values
        yllcorner = float(y_values.min())
        xllcorner = float(x_values.min())

        # Extract CRS if available
        crs = None
        if "crs" in self.ds:
            crs_var = self.ds["crs"]
            if hasattr(crs_var, "spatial_ref"):
                crs = crs_var.attrs.get("spatial_ref")
            elif hasattr(crs_var, "crs_wkt"):
                crs = crs_var.attrs.get("crs_wkt")

        # Convert CRS to EPSG code for cleaner logging output
        epsg = get_epsg_code(crs)
        crs_display = f"EPSG:{epsg}" if epsg else crs

        self.grid_metadata = {
            "shape": (nrows, ncols),
            "resolution": resolution,
            "xllcorner": xllcorner,
            "yllcorner": yllcorner,
            "crs": crs_display,
        }

        logger.debug(f"Grid metadata: {self.grid_metadata}")

    @property
    def start_date(self):
        """First timestamp in the dataset.

        Returns the start date as pandas Timestamp for consistency with MeteoData API.
        """
        return pd.Timestamp(self.time_range[0])

    @property
    def end_date(self):
        """Last timestamp in the dataset.

        Returns the end date as pandas Timestamp for consistency with MeteoData API.
        """
        return pd.Timestamp(self.time_range[1])

    @classmethod
    def from_netcdf(cls, nc_path: str | Path, preload: bool = True) -> "MeteoRaster":
        """Load meteorological raster data from NetCDF file.

        This is an alias for __init__ to match the MeteoData API.

        Args:
            nc_path: Path to NetCDF file containing raster meteorological data
            preload: If True, load all data into memory immediately (default: True).
                If False, use lazy loading (slower but uses less memory).

        Returns:
            MeteoRaster object

        Examples:
            >>> # Fast loading (preload into memory)
            >>> meteo = MeteoRaster.from_netcdf("Arno_meteoraster.nc")
            >>> print(meteo)
            >>>
            >>> # Lazy loading (less memory, slower)
            >>> meteo = MeteoRaster.from_netcdf("Arno_meteoraster.nc", preload=False)
        """
        return cls(nc_path, preload=preload)

    def get_timestep(self, time: datetime, variable: str) -> np.ndarray:
        """Extract 2D grid for a variable at a specific time.

        Uses nearest neighbor time sampling. Results are cached to avoid
        repeated disk reads for the same timestep.

        Args:
            time: Datetime to extract (uses nearest neighbor)
            variable: Variable name (e.g., 'precipitation', 'pet')

        Returns:
            2D numpy array (nrows, ncols) in file units (mm/h for precip/pet)

        Raises:
            KeyError: If variable not found in dataset
        """
        # Check if variable exists
        if variable not in self.variables:
            raise KeyError(f"Variable '{variable}' not found in raster data. Available variables: {self.variables}")

        # Check cache first
        cache_key = (variable, time.isoformat())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract data using xarray's nearest neighbor time selection
        # Note: When data is preloaded, this operation is fast (in-memory)
        # When using lazy loading, this triggers disk I/O
        data = self.ds[variable].sel(time=time, method="nearest").values

        # Cache the result
        self._cache[cache_key] = data

        return data

    def validate_grid_alignment(self, model_metadata: dict) -> None:
        """Validate that raster grid matches model grid.

        Checks shape, resolution, and origin alignment. Raises detailed
        error if grids don't match.

        Args:
            model_metadata: Dictionary with model grid metadata containing:
                - shape: (nrows, ncols)
                - resolution: grid resolution in meters
                - xllcorner: x coordinate of lower-left corner
                - yllcorner: y coordinate of lower-left corner
                - crs: coordinate reference system (optional)

        Raises:
            ValueError: If grids are not aligned with detailed mismatch info
        """
        logger.info("Validating grid alignment between raster and model")

        errors = []

        # Check shape
        if self.grid_metadata["shape"] != model_metadata["shape"]:
            errors.append(f"Shape mismatch: raster={self.grid_metadata['shape']}, model={model_metadata['shape']}")

        # Check resolution
        raster_res = self.grid_metadata["resolution"]
        model_res = model_metadata["resolution"]
        if raster_res is not None and model_res is not None:
            # Handle model_res as tuple (x_res, y_res) or scalar
            if isinstance(model_res, (tuple, list, np.ndarray)):
                model_res_scalar = model_res[0]  # Use x resolution
            else:
                model_res_scalar = model_res

            if not np.isclose(raster_res, model_res_scalar, rtol=1e-6):
                errors.append(f"Resolution mismatch: raster={raster_res:.6f}m, model={model_res_scalar:.6f}m")

        # Check origin (allow 1mm tolerance for floating point precision)
        if not np.isclose(self.grid_metadata["xllcorner"], model_metadata["xllcorner"], atol=1e-3):
            errors.append(
                f"X origin mismatch: raster={self.grid_metadata['xllcorner']:.6f}, "
                f"model={model_metadata['xllcorner']:.6f}"
            )

        if not np.isclose(self.grid_metadata["yllcorner"], model_metadata["yllcorner"], atol=1e-3):
            errors.append(
                f"Y origin mismatch: raster={self.grid_metadata['yllcorner']:.6f}, "
                f"model={model_metadata['yllcorner']:.6f}"
            )

        # Check CRS (warning only, not an error)
        raster_crs = self.grid_metadata["crs"]
        model_crs = model_metadata.get("crs")
        if raster_crs and model_crs:
            if not crs_equals(raster_crs, model_crs):
                logger.warning(
                    "CRS mismatch (proceeding anyway): "
                    "raster CRS != model CRS. "
                    "Ensure grids are in same coordinate system."
                )

        # Raise error if any mismatches found
        if errors:
            error_msg = "Meteorological raster grid does not match model grid:\n  - " + "\n  - ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.success("Grid alignment validated: raster matches model grid")

    def close(self) -> None:
        """Close NetCDF file and clear cache."""
        logger.debug("Closing meteorological raster dataset")
        self.ds.close()
        self._cache.clear()

    def __repr__(self):
        return (
            f"MeteoRaster(variables={self.variables}, "
            f"n_times={len(self.ds.time)}, "
            f"shape={self.grid_metadata['shape']}, "
            f"period={self.start_date} to {self.end_date})"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
