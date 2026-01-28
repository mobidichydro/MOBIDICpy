# Meteorological data

The meteorological data module handles both station-based and gridded (raster) meteorological forcing for MOBIDIC simulations. It also includes tools for generating synthetic design storm hyetographs from IDF (Intensity-Duration-Frequency) parameters.

## Overview

MOBIDICpy supports three modes of meteorological forcing, each suited to different workflows:

### Station-based forcing (MeteoData)

Use when you have time-series from weather stations that need spatial interpolation.

- **Station time series**: Time-series from weather stations
- **Spatial interpolation**: Performed during simulation (IDW or Nearest Neighbor)
- **Format support**: MATLAB .mat files and CF-compliant NetCDF
- **Optional grid export**: Can save interpolated grids as NetCDF for reuse


### Raster-based forcing (MeteoRaster)

Use when you have pre-interpolated gridded data or want to reuse previously interpolated results.

- **Gridded data**: Reads pre-interpolated NetCDF files with dimensions (time, y, x)
- **Performance**: Faster simulation by skipping real-time interpolation
- **Grid validation**: Automatically validates that rasters match the model grid
- **Best for**: Production runs, scenario analysis, or when using external gridded datasets (e.g., reanalysis products)


### Design storm hyetographs (HyetographGenerator)

Use for design storm simulations with synthetic precipitation from IDF curves.

- **IDF-based generation**: Constructs precipitation from spatially distributed IDF parameters (a, n, k)
- **Chicago method**: Currently implements the Chicago decreasing hyetograph (after-peak curve)
- **Areal reduction**: Applies areal reduction factor (ARF) coefficient for spatial averaging
- **Auto-resampling**: IDF parameter rasters are automatically resampled to match the model grid
- **Best for**: Design flood simulations, flood mapping, infrastructure sizing

!!! info "IDF formula"
    The total precipitation depth for a given duration is computed using the Depth-Duration-Frequency (DDF) relationship:

    $$DDF(t) = k_a \cdot k \cdot a \cdot t^n$$

    where:

    - $DDF(t)$ is the cumulative precipitation depth (mm) for duration $t$
    - $k_a$ is the areal reduction factor (ARF) coefficient
    - $k$ is the return period factor (spatially distributed raster)
    - $a$ is the IDF scale parameter (spatially distributed raster)
    - $n$ is the IDF exponent parameter (spatially distributed raster)
    - $t$ is duration in hours


**Workflow recommendation:**

1. **Initial run**: Use station-based forcing with `output_forcing_data.meteo_data = True`
2. **Subsequent runs**: Use the exported raster forcing for faster performance
3. **Different interpolation**: Re-run with station forcing if you need to change interpolation method

## API Reference

### Input data classes

#### MeteoData
Container for station-based meteorological data with spatial interpolation capability.

::: mobidic.preprocessing.meteo_preprocessing.MeteoData

#### MeteoRaster
Container for pre-interpolated raster meteorological forcing.

::: mobidic.preprocessing.meteo_raster.MeteoRaster

#### HyetographGenerator
Generator for synthetic hyetographs from IDF parameters.

::: mobidic.preprocessing.hyetograph.HyetographGenerator

### Output classes

#### MeteoWriter
Writer for saving interpolated meteorological grids from station-based simulations.

::: mobidic.io.meteo.MeteoWriter

### Utility functions

#### convert_mat_to_netcdf
Direct conversion from MATLAB .mat format to CF-compliant NetCDF.

::: mobidic.preprocessing.meteo_preprocessing.convert_mat_to_netcdf

## Supported variables

The meteorological preprocessing module handles six standard variables:

| Variable | Description | Units (NetCDF) | MATLAB field | MATLAB units |
|----------|-------------|----------------|--------------|--------------|
| `precipitation` | Rainfall and snowfall | mm/h | `sp` | mm (cumulated) |
| `temperature_min` | Minimum air temperature | °C | `s_ta_min` | °C |
| `temperature_max` | Maximum air temperature | °C | `s_ta_max` | °C |
| `humidity` | Relative humidity | % | `s_ua` | % |
| `wind_speed` | Wind speed | m/s | `s_vv` | m/s |
| `radiation` | Solar radiation | W/m² | `s_ra` | W/m² |

!!! note "Automatic precipitation unit conversion"
    When loading from MATLAB .mat files, precipitation is automatically converted from mm (cumulated over the sampling interval) to mm/h. The conversion infers the timestep from the time array (`dt = time[1] - time[0]`) and divides precipitation values by the timestep in hours. This ensures consistency with raster forcing, which also uses mm/h.

## Usage examples

### Example 1: Quick conversion from MATLAB format

The fastest way to convert legacy MATLAB meteorological data:

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

### Example 2: Load, inspect, and save station data

When you need to examine data before using it in simulations:

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

### Example 3: Work with existing NetCDF station data

Load and access station data from CF-compliant NetCDF files:

```python
from mobidic import MeteoData

# Load from NetCDF
meteo = MeteoData.from_netcdf("output/meteodata.nc")

# Access station data for a specific variable
precip_stations = meteo.stations["precipitation"]
first_station = precip_stations[0]

print(f"Station code: {first_station['code']}")
print(f"Location: ({first_station['x']:.2f}, {first_station['y']:.2f})")
print(f"Elevation: {first_station['elevation']:.1f} m")
print(f"Data shape: {first_station['data'].shape}")
```

### Example 4: Use pre-interpolated raster forcing

Load and run simulations with gridded meteorological data:

```python
from mobidic import MeteoRaster, Simulation, load_gisdata, load_config

# Load raster forcing (default: preload into memory for best performance)
forcing = MeteoRaster.from_netcdf("meteo_raster.nc")

# Inspect raster data
print(f"Variables: {forcing.variables}")
print(f"Time range: {forcing.start_date} to {forcing.end_date}")
print(f"Grid shape: {forcing.grid_metadata['shape']}")
print(f"Resolution: {forcing.grid_metadata['resolution']:.1f} m")

# For very large datasets (>several GB), use lazy loading to save memory
forcing_lazy = MeteoRaster.from_netcdf("meteo_raster.nc", preload=False)

# Use in simulation (automatically detects raster mode)
config = load_config("basin.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")
sim = Simulation(gisdata, forcing, config)
results = sim.run("2023-01-01", "2023-12-31")
```

### Example 5: Export interpolated data for reuse

Create reusable raster forcing from station-based simulations:

```python
from mobidic import MeteoData, Simulation, load_gisdata, load_config

# Load station-based forcing
forcing = MeteoData.from_netcdf("meteo_stations.nc")

# Enable meteo forcing output in configuration
config = load_config("basin.yaml")
config.output_forcing_data.meteo_data = True

# Run simulation (exports forcing grids automatically)
gisdata = load_gisdata("gisdata.nc", "network.parquet")
sim = Simulation(gisdata, forcing, config)
results = sim.run("2023-01-01", "2023-12-31")

# Forcing data saved to: {output_dir}/meteo_forcing.nc
# Use this file as raster forcing in subsequent runs for faster performance
```

### Example 6: Generate design storm hyetograph from configuration

The simplest workflow for design storm simulations reads all parameters from the YAML configuration:

```python
from datetime import datetime
from pathlib import Path
from mobidic import load_config, load_gisdata, Simulation
from mobidic.preprocessing.hyetograph import HyetographGenerator

# Load configuration with hyetograph section
config_file = Path("basin_hyetograph.yaml")
config = load_config(config_file)
gisdata = load_gisdata(config.paths.gisdata, config.paths.network)

# Generate forcing from config - only start_time needed!
# All parameters (IDF rasters, duration, timestep, output path) read from config
forcing = HyetographGenerator.from_config(
    config=config,
    base_path=config_file.parent,
    start_time=datetime(2000, 1, 1)
)

# Run simulation
sim = Simulation(gisdata, forcing, config)
results = sim.run(forcing.start_date, forcing.end_date)
```

**Required configuration** (`basin_hyetograph.yaml`):

```yaml
paths:
  hyetograph: output/design_storm.nc  # Output NetCDF path

hyetograph:
  a_raster: idf/a.tif        # IDF 'a' parameter raster
  n_raster: idf/n.tif        # IDF 'n' parameter raster
  k_raster: idf/k30.tif      # Return period factor raster (e.g., 30-year)
  duration_hours: 48         # Storm duration
  timestep_hours: 1          # Time step
  hyetograph_type: chicago_decreasing  # Hyetograph method
  ka: 0.8                    # Areal reduction factor
```

### Example 7: Generate hyetograph with manual parameters

For more control over the hyetograph generation process:

```python
from datetime import datetime
from mobidic import Simulation, load_gisdata, load_config
from mobidic.preprocessing.hyetograph import HyetographGenerator

# Create generator from IDF rasters
generator = HyetographGenerator.from_rasters(
    a_raster="idf/a.tif",
    n_raster="idf/n.tif",
    k_raster="idf/k30.tif",
    ka=0.8,                    # Areal reduction factor
    ref_raster="dem.tif"       # Reference grid for resampling
)

# Generate forcing and get MeteoRaster in one call
forcing = generator.generate_forcing(
    duration_hours=48,
    start_time=datetime(2023, 11, 1),
    output_path="design_storm.nc",
    method="chicago_decreasing",
    timestep_hours=1,
    add_metadata={"return_period": "30 years"}
)

# Use directly in simulation
config = load_config("basin.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")
sim = Simulation(gisdata, forcing, config)
results = sim.run(forcing.start_date, forcing.end_date)
```

### Example 8: Advanced hyetograph workflow (full control)

For complete control over generation and export:

```python
from datetime import datetime
from mobidic import MeteoRaster
from mobidic.preprocessing.hyetograph import HyetographGenerator

# Create generator
generator = HyetographGenerator.from_rasters(
    a_raster="idf/a.tif",
    n_raster="idf/n.tif",
    k_raster="idf/k30.tif",
    ka=0.8,
    ref_raster="dem.tif"
)

# Generate time series (returns times and precipitation arrays)
times, precipitation = generator.generate(
    duration_hours=48,
    start_time=datetime(2023, 11, 1),
    method="chicago_decreasing",
    timestep_hours=1,
)

# Inspect generated data
print(f"Time steps: {len(times)}")
print(f"Precipitation shape: {precipitation.shape}")  # (time, y, x)
print(f"Peak intensity: {precipitation[0].max():.2f} mm/h")
print(f"Total depth: {precipitation.sum(axis=0).max():.1f} mm")

# Save to NetCDF with custom metadata
generator.to_netcdf(
    "design_storm.nc",
    times=times,
    precipitation=precipitation,
    add_metadata={
        "return_period": "30 years",
        "event_type": "design_storm",
        "basin": "Arno"
    }
)

# Load as MeteoRaster for simulation
forcing = MeteoRaster.from_netcdf("design_storm.nc")
```


## Data formats

### Station data (MeteoData)

#### MATLAB format

Legacy MATLAB .mat files should contain struct arrays with these fields:

```matlab
% Example structure for precipitation (variable name: sp)
sp(i).code      % Station identifier (string or number)
sp(i).est       % X coordinate (easting)
sp(i).nord      % Y coordinate (northing)
sp(i).quota     % Elevation (m)
sp(i).name      % Station name (optional)
sp(i).time      % Time vector (MATLAB datenum)
sp(i).dati      % Data vector (values)
```

Variable names in MATLAB: `sp` (precipitation), `s_ta_min` (temperature min), `s_ta_max` (temperature max), `s_ua` (humidity), `s_vv` (wind speed), `s_ra` (radiation).

!!! note "Unit conversions during import"
    - **MATLAB datenum**: Automatically converted to pandas datetime, accounting for the epoch difference (MATLAB: January 1, 0000; Unix: January 1, 1970).
    - **Precipitation**: Automatically converted from mm (cumulated over sampling interval) to mm/h by inferring the timestep from the time array.

#### Internal Python representation

`MeteoData` stores station data as a nested dictionary:

```python
meteo.stations = {
    "precipitation": [
        {
            "code": "STATION_001",
            "x": 671234.5,              # Easting
            "y": 4821098.3,             # Northing
            "elevation": 342.0,         # meters
            "name": "Firenze",
            "time": pd.DatetimeIndex([...]),
            "data": np.array([...])     # Time series values
        },
        # ... additional stations
    ],
    "temperature_min": [...],
    "temperature_max": [...],
    # ... other variables
}
```

#### NetCDF format (CF-1.12 compliant)

Station data exported to NetCDF follows Climate and Forecast conventions:

**Structure:**
```
Dimensions:
  - time: N timesteps
  - station_precipitation: M stations for precipitation
  - station_temperature_min: K stations for temperature_min
  - ... (one dimension per variable)

Coordinates:
  - time(time): datetime64[ns]
  - x_precipitation(station_precipitation): float64
  - y_precipitation(station_precipitation): float64
  - elevation_precipitation(station_precipitation): float32
  - station_code_precipitation(station_precipitation): str

Data variables:
  - precipitation(time, station_precipitation): float32 [mm/h]
  - temperature_min(time, station_temperature_min): float32 [°C]
  - ... (one variable per meteorological parameter)
```

**Features:**

- CF-1.12 compliant metadata (standard_name, long_name, units)
- Zlib compression (default level 4) for efficient storage
- Custom global attributes supported via `add_metadata`
- Precipitation stored in mm/h (consistent with raster forcing)

### Raster data (MeteoRaster)

#### NetCDF format (CF-1.12 compliant)

Gridded forcing data follows standard CF conventions:

**Required structure:**
```
Dimensions:
  - time: unlimited (temporal)
  - y: grid rows (north-south)
  - x: grid columns (east-west)

Coordinates:
  - time(time): datetime64[ns] with CF-compliant calendar
  - x(x): float64 (easting or longitude)
  - y(y): float64 (northing or latitude)

Data variables:
  - precipitation(time, y, x): float32 [mm/h]
  - pet(time, y, x): float32 [mm/h]
  - temperature(time, y, x): float32 [°C]
  - ... (additional variables as needed)

Grid mapping:
  - crs(): int32 with spatial_ref or crs_wkt attribute
```

**Example inspection:**
```python
import xarray as xr

ds = xr.open_dataset("meteo_raster.nc")
print(ds)
# <xarray.Dataset>
# Dimensions:         (time: 8761, y: 200, x: 300)
# Coordinates:
#   * time            (time) datetime64[ns] 2023-01-01 ... 2023-12-31
#   * x               (x) float64 600000.0 ... 629900.0
#   * y               (y) float64 4800000.0 ... 4820000.0
# Data variables:
#     precipitation   (time, y, x) float32 ...
#     pet             (time, y, x) float32 ...
#     temperature     (time, y, x) float32 ...
#     crs             () int32 ...
```

**Requirements:**

- Grid must align with model grid (validated automatically)
- Units: mm/h for precipitation, °C for temperature
- Missing values: represented as NaN

