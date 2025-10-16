# Meteorological Preprocessing

The meteorological preprocessing module handles conversion of meteorological forcing data from various formats into CF-compliant NetCDF files for use in MOBIDIC simulations.

## Overview

MOBIDICpy uses a modular architecture for meteorological data preprocessing:

- **Abstract interface**: `MeteoReader` base class for extensibility
- **Format-specific readers**: Currently supports MATLAB .mat files
- **Unified data container**: `MeteoData` class organizes station data by variable type
- **CF-compliant output**: NetCDF files following Climate and Forecast conventions

This design allows easy extension to additional input formats (CSV, NetCDF, database connections) while maintaining a consistent internal representation.

## Classes

### MeteoData

::: mobidic.preprocessing.meteo_preprocessing.MeteoData

## Functions

### Convenience Functions

::: mobidic.preprocessing.meteo_preprocessing.convert_mat_to_netcdf

## Supported Variables

The meteorological preprocessing module handles six standard variables:

| Variable | Description | Units | MATLAB field |
|----------|-------------|-------|--------------|
| `precipitation` | Rainfall and snowfall | mm | `sp` |
| `temperature_min` | Minimum air temperature | °C | `s_ta_min` |
| `temperature_max` | Maximum air temperature | °C | `s_ta_max` |
| `humidity` | Relative humidity | % | `s_ua` |
| `wind_speed` | Wind speed | m/s | `s_vv` |
| `radiation` | Solar radiation | W/m² | `s_ra` |

## Usage Examples

### Example 1: Direct Conversion

Convert MATLAB .mat file directly to NetCDF:

```python
from mobidic import convert_mat_to_netcdf

convert_mat_to_netcdf(
    "input/meteodata.mat",
    "output/meteodata.nc",
    compression_level=4,
    add_metadata={
        "basin": "Arno",
        "description": "Meteorological forcing for flood event",
    }
)
```

### Example 2: Load and Inspect

Load data, inspect it, and then save:

```python
from mobidic import MeteoData

# Load from MATLAB file
meteo = MeteoData.from_mat("input/meteodata.mat")

# Inspect loaded data
print(meteo)
print(f"Date range: {meteo.start_date} to {meteo.end_date}")
print(f"Variables: {meteo.variables}")

# Check station counts
for var_name, stations in meteo.stations.items():
    print(f"{var_name}: {len(stations)} stations")

# Save to NetCDF with metadata
meteo.to_netcdf(
    "output/meteodata.nc",
    compression_level=4,
    add_metadata={"basin": "Arno"}
)
```

### Example 3: Read Back from NetCDF

Load previously saved NetCDF file:

```python
from mobidic import MeteoData

# Load from NetCDF
meteo = MeteoData.from_netcdf("output/meteodata.nc")

# Access station data
precip_stations = meteo.stations["precipitation"]
first_station = precip_stations[0]

print(f"Station code: {first_station['code']}")
print(f"Location: ({first_station['x']:.2f}, {first_station['y']:.2f})")
print(f"Elevation: {first_station['elevation']:.1f} m")
print(f"Data shape: {first_station['data'].shape}")
```

## Data Structure

### MATLAB Input Format

MATLAB .mat files should contain struct arrays with the following fields:

```matlab
% Example structure for precipitation
sp(i).code      % Station identifier (string or number)
sp(i).est       % X coordinate (easting)
sp(i).nord      % Y coordinate (northing)
sp(i).quota     % Elevation (m)
sp(i).name      % Station name (optional)
sp(i).time      % Time vector (MATLAB datenum)
sp(i).dati      % Data vector (values)
```

**Important**: MATLAB datenum values are automatically converted to pandas datetime, accounting for the MATLAB epoch (January 1, 0000) vs Unix epoch (January 1, 1970) offset.

### Internal Representation

`MeteoData` stores data in a dictionary structure:

```python
meteo.stations = {
    "precipitation": [
        {
            "code": "STATION_001",
            "x": 671234.5,
            "y": 4821098.3,
            "elevation": 342.0,
            "name": "Firenze",
            "time": pd.DatetimeIndex([...]),
            "data": np.array([...])
        },
        # ... more stations
    ],
    "temperature_min": [...],
    # ... other variables
}
```

### NetCDF Output Format

NetCDF files follow CF-1.12 conventions with:

**Dimensions:**
- `time`: Temporal dimension
- `station_{var}`: Station dimension for each variable

**Coordinates:**
- `time`: Datetime values (CF-compliant with calendar and units)
- `x_{var}`, `y_{var}`: Station coordinates
- `elevation_{var}`: Station elevations
- `station_code_{var}`: Station identifiers

**Data variables:**
- `{var}`: 2D arrays (time × station) for each meteorological variable

**Attributes:**
- Global attributes: CF conventions, creation date, custom metadata
- Variable attributes: standard_name, long_name, units, _FillValue

**Compression:**
- Uses zlib compression (default level 4)
- Significantly reduces file size for sparse station networks

## Architecture

### Extensibility

The module is designed for easy extension to new input formats:

```python
from mobidic.preprocessing.meteo_preprocessing import MeteoReader

class CSVMeteoReader(MeteoReader):
    """Reader for CSV-formatted meteorological data."""

    def read(self, file_path):
        # Implement CSV reading logic
        # Return dictionary of stations by variable
        pass

# Then use it:
meteo = MeteoData()
reader = CSVMeteoReader()
meteo.stations = reader.read("data.csv")
meteo.to_netcdf("output.nc")
```

### Planned Extensions

Future enhancements include:

- CSV and Excel readers
- Database connectivity (PostgreSQL, InfluxDB)
- Quality control and gap filling
- Temporal aggregation and interpolation
- Spatial interpolation onto model grid
- Real-time data ingestion

## Performance Considerations

### Memory Usage

MeteoData loads all station time series into memory. For very large datasets:

- Consider processing in chunks by variable or time period
- Use generators for streaming processing
- Leverage Dask for out-of-core computation

### File Sizes

NetCDF compression significantly reduces file sizes:

| Compression Level | Compression Ratio | Write Speed |
|-------------------|-------------------|-------------|
| 0 (none) | 1.0× | Fastest |
| 4 (default) | ~3-5× | Fast |
| 9 (maximum) | ~4-6× | Slow |

Level 4 provides good compression with minimal performance impact.

## MATLAB Translation

This module replaces MATLAB's manual .mat file handling and provides:

- Automatic datenum conversion
- CF-compliant NetCDF output (not available in MATLAB version)
- Type safety and validation
- Extensible architecture for multiple input formats

## Error Handling

The module provides comprehensive error checking:

- **File not found**: Clear error messages with file paths
- **Invalid format**: Validation of expected MATLAB struct fields
- **Missing data**: Warnings for stations with no valid data
- **Coordinate issues**: Validation of coordinate ranges
- **Time issues**: Detection of non-monotonic or duplicate timestamps

All operations use loguru for structured logging at appropriate levels (DEBUG, INFO, WARNING, ERROR).
