"""Meteorological data I/O.

This module provides functionality to save interpolated meteorological data
(precipitation, temperature, etc.) from simulation to NetCDF format.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
from loguru import logger
from mobidic import __version__


class MeteoWriter:
    """
    Writer for interpolated meteorological data grids.

    This class stores interpolated meteorological grids during simulation
    and writes them to a single CF-compliant NetCDF file at the end.

    Unlike StateWriter which uses incremental writing with chunking, this writer
    buffers all data in memory and writes once. This is more efficient for meteo
    data which is typically much smaller than state data.

    Args:
        output_path: Path to output NetCDF file (will be created/overwritten)
        grid_metadata: Dictionary with grid metadata (shape, resolution, crs, etc.)
        variables: List of meteorological variable names to save (e.g., ['precipitation', 'temperature'])
        add_metadata: Additional global attributes (optional)

    Examples:
        >>> # Create writer for precipitation and temperature
        >>> writer = MeteoWriter(
        ...     "meteo_interpolated.nc",
        ...     metadata,
        ...     variables=['precipitation', 'temperature']
        ... )
        >>> for step in range(num_steps):
        ...     writer.append(current_time, precipitation=precip_grid, temperature=temp_grid)
        >>> writer.close()  # Writes all data to file
    """

    def __init__(
        self,
        output_path: str | Path,
        grid_metadata: dict,
        variables: list[str],
        add_metadata: dict | None = None,
    ):
        """Initialize the meteo writer."""
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.grid_metadata = grid_metadata
        self.variables = variables
        self.add_metadata = add_metadata or {}

        # Get grid dimensions
        self.nrows, self.ncols = grid_metadata["shape"]
        resolution = grid_metadata["resolution"]
        xllcorner = grid_metadata["xllcorner"]
        yllcorner = grid_metadata["yllcorner"]

        # Create coordinate arrays
        self.x = xllcorner + np.arange(self.ncols) * resolution[0]
        self.y = yllcorner + np.arange(self.nrows) * resolution[1]

        # Initialize buffers for data
        self.data_buffer = {var: [] for var in variables}
        self.time_buffer = []

        logger.debug(f"MeteoWriter initialized: {self.output_path}, variables={variables}")

    def append(self, time: datetime, **grids: np.ndarray) -> None:
        """
        Append meteorological grids for a timestep.

        Args:
            time: Timestamp for this data
            **grids: Keyword arguments for each variable grid (e.g., precipitation=precip_grid).
                Each grid should be a 2D numpy array with shape (nrows, ncols).

        Raises:
            ValueError: If a required variable is missing or has wrong shape

        Examples:
            >>> writer.append(current_time, precipitation=precip_grid, temperature=temp_grid)
        """
        # Validate that all required variables are provided
        for var in self.variables:
            if var not in grids:
                raise ValueError(f"Missing required variable '{var}' in append call")

            grid = grids[var]
            if grid.shape != (self.nrows, self.ncols):
                raise ValueError(
                    f"Variable '{var}' has incorrect shape {grid.shape}, expected {(self.nrows, self.ncols)}"
                )

        # Append data to buffers
        self.time_buffer.append(time)
        for var in self.variables:
            self.data_buffer[var].append(grids[var].copy())

    def close(self) -> None:
        """
        Write all buffered data to NetCDF file and close writer.

        This creates a CF-1.12 compliant NetCDF file with all meteorological
        variables and metadata.
        """
        if len(self.time_buffer) == 0:
            logger.warning("No data to write (buffer is empty)")
            return

        logger.info(f"Writing interpolated meteorological data to NetCDF: {self.output_path}")

        # Convert buffers to numpy arrays
        n_times = len(self.time_buffer)
        data_vars = {}

        for var in self.variables:
            # Stack list of 2D arrays into 3D array (time, y, x)
            data_array = np.stack(self.data_buffer[var], axis=0)

            # Convert units from m/s to mm/h for precipitation and PET
            if var in ["precipitation", "pet"]:
                data_array = data_array * 1000.0 * 3600.0

            # Create DataArray with metadata
            data_vars[var] = (
                ["time", "y", "x"],
                data_array,
                {
                    "long_name": _get_variable_longname(var),
                    "units": _get_variable_units(var),
                    "grid_mapping": "crs",
                },
            )

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": (["time"], self.time_buffer, {"long_name": "time", "axis": "T"}),
                "x": (
                    ["x"],
                    self.x,
                    {
                        "standard_name": "projection_x_coordinate",
                        "long_name": "x coordinate of projection",
                        "units": "m",
                        "axis": "X",
                    },
                ),
                "y": (
                    ["y"],
                    self.y,
                    {
                        "standard_name": "projection_y_coordinate",
                        "long_name": "y coordinate of projection",
                        "units": "m",
                        "axis": "Y",
                    },
                ),
            },
        )

        # Add grid mapping variable for CRS (CF-1.12 compliance)
        crs_string = str(self.grid_metadata.get("crs", ""))
        ds["crs"] = ([], 0)
        ds["crs"].attrs = {
            "grid_mapping_name": "spatial_ref",
            "crs_wkt": crs_string,
            "spatial_ref": crs_string,
        }

        # Add global attributes
        ds.attrs["title"] = "MOBIDIC interpolated meteorological data"
        ds.attrs["source"] = "Spatial interpolation from station observations"
        ds.attrs["Conventions"] = "CF-1.12"
        ds.attrs["history"] = f"Created by MOBIDICpy v{__version__}"
        ds.attrs["date_created"] = datetime.now().isoformat()

        # Add user-provided metadata
        ds.attrs.update(self.add_metadata)

        # Set encoding for compression
        encoding = {}
        for var in self.variables:
            encoding[var] = {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.nan,
            }

        # Write to NetCDF
        ds.to_netcdf(
            self.output_path,
            encoding=encoding,
            format="NETCDF4",
        )

        logger.success(
            f"Saved interpolated meteorological data: {len(self.variables)} variables, "
            f"{n_times} timesteps, {self.output_path}"
        )

        # Clear buffers to free memory
        self.data_buffer.clear()
        self.time_buffer.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close and write data."""
        if exc_type is None:
            # No exception - write data
            self.close()
        else:
            # Exception occurred - log and don't write
            logger.error(f"MeteoWriter exiting due to exception: {exc_val}")
        return False


def _get_variable_longname(var_name: str) -> str:
    """Get CF-compliant long name for variable.

    Args:
        var_name: Variable name

    Returns:
        Long descriptive name
    """
    longnames = {
        "precipitation": "precipitation rate",
        "pet": "potential evapotranspiration rate",
        "temperature": "air temperature",
        "temperature_min": "minimum air temperature",
        "temperature_max": "maximum air temperature",
        "humidity": "relative humidity",
        "wind_speed": "wind speed",
        "radiation": "surface downwelling shortwave flux",
    }
    return longnames.get(var_name, var_name)


def _get_variable_units(var_name: str) -> str:
    """Get CF-compliant units for variable.

    Args:
        var_name: Variable name

    Returns:
        Units string
    """
    units = {
        "precipitation": "mm h-1",
        "pet": "mm h-1",
        "temperature": "degC",
        "temperature_min": "degC",
        "temperature_max": "degC",
        "humidity": "%",
        "wind_speed": "m s-1",
        "radiation": "W m-2",
    }
    return units.get(var_name, "unknown")
