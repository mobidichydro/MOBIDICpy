"""Meteorological data preprocessing - station-based forcing data.

This module handles conversion of meteorological forcing data from various formats
(MATLAB .mat, NetCDF, CSV, etc.) to CF-compliant NetCDF format for use in MOBIDIC simulations.

The module provides:
- MAT file reader for MATLAB meteodata.mat files
- NetCDF reader/writer for CF-compliant input/output
- Validation and gap-filling capabilities (not yet implemented)
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mobidic import __version__


class MeteoData:
    """Container for meteorological station data.

    This class holds meteorological observations from multiple stations,
    organized by variable type (precipitation, temperature, wind, etc.).

    Attributes:
        stations: Dictionary mapping variable names to lists of station dictionaries
        variables: List of available meteorological variables
        start_date: First timestamp in the dataset
        end_date: Last timestamp in the dataset
    """

    def __init__(self, stations: dict[str, list[dict[str, Any]]]):
        """Initialize MeteoData container.

        Args:
            stations: Dictionary with variable names as keys ('precipitation',
                'temperature_min', 'temperature_max', 'humidity', 'wind_speed',
                'radiation') and lists of station dictionaries as values.
                Each station dict contains: 'code', 'x', 'y', 'elevation',
                'time', 'data', 'name' (optional).
        """
        self.stations = stations
        self.variables = list(stations.keys())

        # Extract date range from all stations
        all_times = []
        for var_stations in stations.values():
            for station in var_stations:
                if len(station["time"]) > 0:
                    all_times.extend(station["time"])

        if all_times:
            self.start_date = pd.Timestamp(min(all_times))
            self.end_date = pd.Timestamp(max(all_times))
        else:
            self.start_date = None
            self.end_date = None

    def __repr__(self):
        n_stations = {var: len(stations) for var, stations in self.stations.items()}
        return (
            f"MeteoData(variables={self.variables}, "
            f"n_stations={n_stations}, "
            f"period={self.start_date} to {self.end_date})"
        )

    def to_netcdf(
        self,
        output_path: str | Path,
        compression_level: int = 4,
        add_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save meteorological data to CF-compliant NetCDF file.

        Args:
            output_path: Path to output NetCDF file
            compression_level: Compression level for NetCDF (0-9, default 4)
            add_metadata: Additional global attributes to add to NetCDF file

        Examples:
            >>> meteo = MeteoData.from_mat("meteodata.mat")
            >>> meteo.to_netcdf("meteodata.nc", add_metadata={"basin": "Arno"})
        """
        output_path = Path(output_path)
        logger.info(f"Saving meteorological data to NetCDF: {output_path}")

        # Create xarray Dataset with station data organized by variable
        # Group related variables that should share station dimensions
        # temperature_min and temperature_max share the same stations
        var_groups = {
            "temperature": ["temperature_min", "temperature_max"],
            "precipitation": ["precipitation"],
            "humidity": ["humidity"],
            "wind_speed": ["wind_speed"],
            "radiation": ["radiation"],
        }

        datasets = {}

        for group_name, var_list in var_groups.items():
            # Check which variables in this group are available
            available_vars = [v for v in var_list if v in self.stations and len(self.stations[v]) > 0]

            if len(available_vars) == 0:
                continue

            # Use the first available variable to define the station dimension
            # For temperature, this ensures min and max share the same stations
            primary_var = available_vars[0]
            var_stations = self.stations[primary_var]

            # Extract station metadata from primary variable
            n_stations = len(var_stations)
            station_codes = np.array([s["code"] for s in var_stations])
            station_x = np.array([s["x"] for s in var_stations])
            station_y = np.array([s["y"] for s in var_stations])
            station_elevation = np.array([s["elevation"] for s in var_stations])
            station_names = np.array([s.get("name", "") for s in var_stations], dtype=str)

            # Collect all unique timestamps across all variables in this group
            all_times = []
            for var_name in available_vars:
                for station in self.stations[var_name]:
                    all_times.extend(station["time"])
            unique_times = pd.DatetimeIndex(sorted(set(all_times)))
            n_times = len(unique_times)

            # Station dimension name based on group (e.g., "station_temperature")
            station_dim = f"station_{group_name}"

            # Create data variables for all variables in this group
            data_vars = {}
            for var_name in available_vars:
                var_stations = self.stations[var_name]

                # Create data array (time x station)
                data_array = np.full((n_times, n_stations), np.nan, dtype=np.float32)

                # Fill data array
                for i, station in enumerate(var_stations):
                    if len(station["time"]) > 0:
                        # Find indices of this station's times in the unified time array
                        time_indices = unique_times.get_indexer(pd.DatetimeIndex(station["time"]))
                        valid_mask = time_indices >= 0
                        data_array[time_indices[valid_mask], i] = station["data"][valid_mask]

                data_vars[var_name] = (
                    ["time", station_dim],
                    data_array,
                    {
                        "long_name": _get_variable_longname(var_name),
                        "units": _get_variable_units(var_name),
                        "missing_value": np.nan,
                    },
                )

            # Create xarray Dataset for this group
            ds = xr.Dataset(
                data_vars,
                coords={
                    "time": (["time"], unique_times, {"long_name": "time", "axis": "T"}),
                    station_dim: ([station_dim], np.arange(n_stations), {"long_name": "station index"}),
                    f"station_code_{group_name}": (
                        [station_dim],
                        station_codes,
                        {"long_name": f"{group_name} station code"},
                    ),
                    f"x_{group_name}": (
                        [station_dim],
                        station_x,
                        {"long_name": f"{group_name} station x coordinate (easting)", "units": "m"},
                    ),
                    f"y_{group_name}": (
                        [station_dim],
                        station_y,
                        {"long_name": f"{group_name} station y coordinate (northing)", "units": "m"},
                    ),
                    f"elevation_{group_name}": (
                        [station_dim],
                        station_elevation,
                        {"long_name": f"{group_name} station elevation", "units": "m"},
                    ),
                    f"station_name_{group_name}": (
                        [station_dim],
                        station_names,
                        {"long_name": f"{group_name} station name"},
                    ),
                },
            )

            datasets[group_name] = ds

        # Merge all variable datasets
        if len(datasets) == 0:
            raise ValueError("No valid meteorological data to save")

        merged_ds = xr.merge(datasets.values(), compat="no_conflicts")

        # Add global attributes
        merged_ds.attrs["title"] = "MOBIDIC meteorological forcing data"
        merged_ds.attrs["source"] = "Station observations"
        merged_ds.attrs["Conventions"] = "CF-1.12"
        merged_ds.attrs["history"] = f"Created by MOBIDICpy v{__version__}"

        if add_metadata:
            merged_ds.attrs.update(add_metadata)

        # Set encoding for compression
        encoding = {}
        for var in merged_ds.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
                "dtype": "float32",
                "_FillValue": np.nan,
            }

        # Save to NetCDF
        merged_ds.to_netcdf(
            output_path,
            encoding=encoding,
            format="NETCDF4",
        )

        logger.success(f"Saved meteorological data: {len(datasets)} variables, {output_path}")

    @classmethod
    def from_mat(cls, mat_path: str | Path) -> "MeteoData":
        """Load meteorological data from MATLAB .mat file (legacy MOBIDIC format).

        Args:
            mat_path: Path to MATLAB .mat file containing meteodata

        Returns:
            MeteoData object with station data

        Examples:
            >>> meteo = MeteoData.from_mat("examples/Arno/meteodata/meteodata.mat")
            >>> print(meteo)
            >>> meteo.to_netcdf("meteodata.nc")
        """
        reader = MATMeteoReader(mat_path)
        return reader.read()

    @classmethod
    def from_netcdf(cls, nc_path: str | Path) -> "MeteoData":
        """Load meteorological data from NetCDF file.

        Args:
            nc_path: Path to NetCDF file containing meteodata

        Returns:
            MeteoData object with station data

        Examples:
            >>> meteo = MeteoData.from_netcdf("meteodata.nc")
            >>> print(meteo)
        """
        reader = NetCDFMeteoReader(nc_path)
        return reader.read()

    @classmethod
    def from_csv(cls, csv_path: str | Path, config: dict[str, Any]) -> "MeteoData":
        """Load meteorological data from CSV file(s).

        Args:
            csv_path: Path to CSV file or directory containing CSV files
            config: Configuration dictionary specifying CSV structure

        Returns:
            MeteoData object with station data

        Note:
            This method is planned for future implementation.
        """
        raise NotImplementedError("CSV reader not yet implemented")


class MATMeteoReader:
    """Reader for MATLAB .mat meteorological data files.

    This reader handles the specific MATLAB struct format used by MOBIDIC,
    which contains arrays of station structs for different variables.
    Expected variables in .mat file:
    - sp: precipitation stations
    - s_ta_min: minimum temperature stations
    - s_ta_max: maximum temperature stations
    - s_ua: relative humidity stations
    - s_vv: wind speed stations
    - s_ra: solar radiation stations

    Each station struct contains:
    - code: station identifier
    - est: x coordinate (easting) [m]
    - nord: y coordinate (northing) [m]
    - quota: elevation [m a.s.l.]
    - name: station name (optional)
    - time: array of timestamps (MATLAB datenum format)
    - dati: array of observed values
    """

    # Mapping from MATLAB variable names to standard names
    VAR_MAP = {
        "sp": "precipitation",
        "s_ta_min": "temperature_min",
        "s_ta_max": "temperature_max",
        "s_ua": "humidity",
        "s_vv": "wind_speed",
        "s_ra": "radiation",
    }

    def __init__(self, file_path: str | Path):
        """Initialize MAT file reader.

        Args:
            file_path: Path to MATLAB .mat file containing meteodata

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Meteorological data file not found: {self.file_path}")

    def read(self) -> MeteoData:
        """Read MATLAB .mat file and extract meteorological station data.

        Returns:
            MeteoData object with station data organized by variable
        """
        logger.info(f"Reading MATLAB meteo data from: {self.file_path}")

        try:
            import scipy.io
        except ImportError as e:
            raise ImportError("scipy is required to read .mat files. Install with: pip install scipy") from e

        # Load MATLAB file
        mat_data = scipy.io.loadmat(str(self.file_path), squeeze_me=False, struct_as_record=True)

        stations = {}

        # Process each variable
        for mat_var, standard_var in self.VAR_MAP.items():
            if mat_var not in mat_data:
                logger.warning(f"Variable {mat_var} not found in .mat file, skipping")
                continue

            var_data = mat_data[mat_var]

            # Extract station array (shape is typically (1, n_stations))
            if var_data.shape[0] == 1:
                station_array = var_data[0]
            else:
                station_array = var_data.flatten()

            var_stations = []

            # Process each station
            for i, station_struct in enumerate(station_array):
                try:
                    station_dict = self._parse_station_struct(station_struct)
                    var_stations.append(station_dict)
                except Exception as e:
                    logger.warning(f"Failed to parse station {i} for {standard_var}: {e}")
                    continue

            # Convert precipitation from mm (cumulated over sampling interval) to mm/h
            # Infer timestep from the first station's time array: dt = time[1] - time[0]
            if standard_var == "precipitation" and len(var_stations) > 0:
                # Find a station with at least 2 timesteps to infer dt
                dt_hours = None
                for station in var_stations:
                    if len(station["time"]) >= 2:
                        dt_seconds = (station["time"][1] - station["time"][0]).total_seconds()
                        dt_hours = dt_seconds / 3600.0
                        break

                if dt_hours is not None and dt_hours > 0:
                    for station in var_stations:
                        station["data"] = station["data"] / dt_hours
                    logger.info(f"Converted precipitation from mm (cumulated over {dt_hours:.2f}h) to mm/h")
                else:
                    logger.warning("Could not infer timestep for precipitation conversion (need at least 2 timesteps)")

            stations[standard_var] = var_stations
            logger.debug(f"Loaded {len(var_stations)} stations for {standard_var}")

        logger.success(f"Loaded meteorological data: {len(stations)} variables")

        return MeteoData(stations)

    def _parse_station_struct(self, station_struct: np.ndarray) -> dict[str, Any]:
        """Parse MATLAB station struct into Python dictionary.

        Args:
            station_struct: NumPy structured array representing a station

        Returns:
            Dictionary with station metadata and data
        """
        # Extract fields from MATLAB struct
        code = int(station_struct["code"][0][0]) if station_struct["code"].size > 0 else 0
        x = float(station_struct["est"][0][0]) if station_struct["est"].size > 0 else 0.0
        y = float(station_struct["nord"][0][0]) if station_struct["nord"].size > 0 else 0.0
        elevation = float(station_struct["quota"][0][0]) if station_struct["quota"].size > 0 else 0.0

        # Name field might be empty
        if station_struct["name"].size > 0 and len(station_struct["name"][0]) > 0:
            name = str(station_struct["name"][0])
        else:
            name = f"Station_{code}"

        # Time array (MATLAB datenum format: days since January 0, 0000)
        # Shape is typically (n_times, 1), need to flatten
        time_matlab_raw = station_struct["time"] if station_struct["time"].size > 0 else np.array([])
        time_matlab = time_matlab_raw.flatten() if time_matlab_raw.size > 0 else np.array([])

        # Convert MATLAB datenum to pandas datetime (vectorized for speed)
        # MATLAB datenum uses the proleptic Gregorian calendar with day 1 = Jan 1, year 0
        # MATLAB datenum 367 = Jan 1, year 1 (year 0 has 366 days as a leap year)
        # Python datetime64("0001-01-01") represents Jan 1, year 1
        # When adding timedelta64[D](n), n=0 gives Jan 1, n=1 gives Jan 2, etc.
        # Therefore: adjusted_days = matlab_datenum - 367
        if len(time_matlab) > 0:
            # Vectorized conversion: extract integer days and fractional days
            days_int = time_matlab.astype(int)
            days_frac = time_matlab - days_int

            # Subtract 367 days offset (MATLAB day 367 = Python day 0 from epoch 0001-01-01)
            adjusted_days = days_int - 367

            # Vectorized datetime conversion using numpy datetime64
            # Convert to timedelta64[D] and add to epoch
            epoch = np.datetime64("0001-01-01")
            days_since_epoch = adjusted_days.astype("timedelta64[D]")
            time_np = epoch + days_since_epoch

            # Convert to pandas DatetimeIndex
            time_pd = pd.DatetimeIndex(time_np)

            # Add fractional days (sub-daily component)
            time_pd = time_pd + pd.to_timedelta(days_frac, unit="D")

            # Round to seconds to avoid floating point precision issues
            time_pd = time_pd.round("s")
        else:
            time_pd = pd.DatetimeIndex([])

        # Data array (shape is typically (n_times, 1), need to flatten)
        data_raw = station_struct["dati"] if station_struct["dati"].size > 0 else np.array([])
        data = data_raw.flatten() if data_raw.size > 0 else np.array([])

        return {
            "code": code,
            "x": x,
            "y": y,
            "elevation": elevation,
            "name": name,
            "time": time_pd,
            "data": data,
        }


class NetCDFMeteoReader:
    """Reader for NetCDF meteorological data files.

    This reader handles NetCDF files created by the to_netcdf() method,
    which follow the CF-1.12 convention and contain meteorological station data.

    Expected structure in NetCDF file:
    - Data variables: precipitation, temperature_min, temperature_max, humidity, wind_speed, radiation
    - Coordinates: time, station, station_code, x, y, elevation, station_name
    - Each variable has dimensions (time, station)
    """

    def __init__(self, file_path: str | Path):
        """Initialize NetCDF file reader.

        Args:
            file_path: Path to NetCDF file containing meteodata

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Meteorological data file not found: {self.file_path}")

    def read(self) -> MeteoData:
        """Read NetCDF file and extract meteorological station data.

        Returns:
            MeteoData object with station data organized by variable
        """
        logger.info(f"Reading NetCDF meteo data from: {self.file_path}")

        # Load NetCDF file using xarray
        ds = xr.open_dataset(self.file_path)

        stations = {}

        # Define variable groups and their corresponding station dimensions
        var_to_group = {
            "temperature_min": "temperature",
            "temperature_max": "temperature",
            "precipitation": "precipitation",
            "humidity": "humidity",
            "wind_speed": "wind_speed",
            "radiation": "radiation",
        }

        # List of expected meteorological variables
        expected_vars = ["precipitation", "temperature_min", "temperature_max", "humidity", "wind_speed", "radiation"]

        # Process each variable
        for var_name in expected_vars:
            if var_name not in ds.data_vars:
                logger.debug(f"Variable {var_name} not found in NetCDF file, skipping")
                continue

            # Extract data array and coordinates
            data_var = ds[var_name]

            # Get the group name for this variable (e.g., temperature_min -> temperature)
            group_name = var_to_group.get(var_name, var_name)

            # Station dimension name based on group (e.g., "station_temperature")
            station_dim = f"station_{group_name}"
            if station_dim not in ds.dims:
                logger.warning(f"Station dimension {station_dim} not found for {var_name}, skipping")
                continue

            n_stations = len(ds[station_dim])

            var_stations = []

            # Process each station
            for i in range(n_stations):
                station_code = int(ds[f"station_code_{group_name}"][i].values)
                station_x = float(ds[f"x_{group_name}"][i].values)
                station_y = float(ds[f"y_{group_name}"][i].values)
                station_elevation = float(ds[f"elevation_{group_name}"][i].values)
                station_name = str(ds[f"station_name_{group_name}"][i].values)

                # Extract time series for this station
                station_data = data_var[:, i].values
                station_times = pd.DatetimeIndex(ds["time"].values)
                valid_times = station_times
                valid_data = station_data

                station_dict = {
                    "code": station_code,
                    "x": station_x,
                    "y": station_y,
                    "elevation": station_elevation,
                    "name": station_name,
                    "time": valid_times,
                    "data": valid_data,
                }

                var_stations.append(station_dict)

            stations[var_name] = var_stations
            logger.debug(f"Loaded {len(var_stations)} stations for {var_name}")

        ds.close()

        logger.success(f"Loaded meteorological data: {len(stations)} variables")

        return MeteoData(stations)


def _get_variable_longname(var_name: str) -> str:
    """Get CF-compliant long name for variable.

    Args:
        var_name: Standard variable name

    Returns:
        Long descriptive name
    """
    longnames = {
        "precipitation": "precipitation",
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
        var_name: Standard variable name

    Returns:
        Units string
    """
    units = {
        "precipitation": "mm h-1",
        "temperature_min": "degC",
        "temperature_max": "degC",
        "humidity": "%",
        "wind_speed": "m s-1",
        "radiation": "W m-2",
    }
    return units.get(var_name, "unknown")


def convert_mat_to_netcdf(
    mat_path: str | Path,
    output_path: str | Path,
    compression_level: int = 4,
    add_metadata: dict[str, Any] | None = None,
) -> None:
    """Convert MATLAB .mat meteorological data to NetCDF format.

    This is a convenience function that combines loading from .mat and
    saving to NetCDF in a single call.

    Args:
        mat_path: Path to input MATLAB .mat file
        output_path: Path to output NetCDF file
        compression_level: Compression level (0-9, default 4)
        add_metadata: Additional global attributes for NetCDF file

    Examples:
        >>> convert_mat_to_netcdf(
        ...     "examples/Arno/meteodata/meteodata.mat",
        ...     "examples/Arno/meteodata/meteodata.nc",
        ...     add_metadata={"basin": "Arno", "basin_id": "1234"}
        ... )
    """
    logger.info("=" * 80)
    logger.info("MOBIDIC METEO PREPROCESSING")
    logger.info("=" * 80)

    # Load from MAT
    meteo_data = MeteoData.from_mat(mat_path)
    logger.info(f"Loaded {len(meteo_data.variables)} meteorological variables")
    logger.info(f"Date range: {meteo_data.start_date} to {meteo_data.end_date}")

    # Save to NetCDF
    meteo_data.to_netcdf(output_path, compression_level, add_metadata)

    logger.info("=" * 80)
    logger.info("METEO PREPROCESSING COMPLETE")
    logger.info("=" * 80)
