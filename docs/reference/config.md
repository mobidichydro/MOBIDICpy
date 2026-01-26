# Configuration

The configuration module provides a schema-driven approach to loading and validating MOBIDIC model configurations using YAML.

## Overview

Configuration files define all aspects of a MOBIDIC simulation, including:

- **Basin metadata**: Optional basin ID, optional parameter set ID, baricenter coordinates
- **Input/output paths**: Meteodata, GIS data, network, states, output directories
- **Raster and vector data sources**: DTM, flow direction/accumulation, soil/energy parameters, river network
- **Model parameters**: Organized into subsections:
    - `soil`: Hydraulic conductivity, water holding capacity, flow coefficients
    - `energy`: Thermal properties, turbulent exchange, albedo
    - `routing`: Channel routing method, wave celerity, Manning coefficient
    - `groundwater`: Model type, global loss
    - `reservoirs`: Reservoir shapefiles, stage-storage curves, regulation curves/schedules (optional)
    - `multipliers`: Multiplying factors for calibration
- **Initial conditions**: Initial state (hillslope runoff, soil saturation, reservoir volumes)
- **Simulation settings**: Time step, resampling, soil/energy balance schemes, precipitation interpolation
- **Output options**:
    - `output_states`: Boolean flags for state variables to save
    - `output_states_settings`: State output format (NetCDF) and intervals
    - `output_report`: Report variables to save (discharge, lateral inflow)
    - `output_report_settings`: Report format (CSV/Parquet) and reach selection
    - `output_forcing_data`: Meteorological forcing data output options
- **Advanced settings**: Logging level and log file

The configuration system ensures all required fields are present, validates ranges and consistency, and provides sensible defaults for optional parameters.

## Functions

::: mobidic.config.parser.load_config

::: mobidic.config.parser.save_config

## Classes

### Main configuration

::: mobidic.config.schema.MOBIDICConfig
    options:
      members:
        - basin
        - paths
        - vector_files
        - raster_files
        - raster_settings
        - parameters
        - initial_conditions
        - simulation
        - output_states
        - output_states_settings
        - output_report
        - output_report_settings
        - advanced

### Nested models

The configuration is organized hierarchically (using nested Pydantic models):

::: mobidic.config.schema.Basin

::: mobidic.config.schema.Paths

::: mobidic.config.schema.VectorFiles

::: mobidic.config.schema.RasterFiles

::: mobidic.config.schema.RasterSettings

::: mobidic.config.schema.Parameters

::: mobidic.config.schema.SoilParameters

::: mobidic.config.schema.EnergyParameters

::: mobidic.config.schema.RoutingParameters

::: mobidic.config.schema.GroundwaterParameters

::: mobidic.config.schema.ReservoirParameters

::: mobidic.config.schema.Multipliers

::: mobidic.config.schema.InitialConditions

::: mobidic.config.schema.Simulation

::: mobidic.config.schema.OutputStates

::: mobidic.config.schema.OutputStatesSettings

::: mobidic.config.schema.OutputReport

::: mobidic.config.schema.OutputReportSettings

::: mobidic.config.schema.OutputInterpolatedData

::: mobidic.config.schema.Advanced

## Configuration structure

See the [sample configuration file](https://github.com/mobidichydro/mobidicpy/blob/main/examples/sample_config.yaml) for a complete example with all available options and their descriptions.

## Validation

The configuration system performs validation of the .yaml files with the following checks:

- **Type checking**: All fields are type-checked by Pydantic
- **Range validation**: Numeric parameters are validated against physical constraints (e.g., albedo 0-1, saturation 0-1)
- **Required fields**: Missing required fields raise validation errors
- **Consistency checks**: Inter-field dependencies are validated (e.g., if `reach_selection='file'`, then `sel_file` must be provided)
- **Path validation**: File paths can optionally be validated for existence

Invalid configurations will raise `pydantic.ValidationError` with detailed error messages.

## Auxiliary files

### Reach selection file

When using `reach_selection='file'` in the output report settings, you need to provide a JSON file containing the reach IDs to include in the output reports. The file should contain a simple JSON array of integer reach IDs (corresponding to `mobidic_id` values in the processed network).

**Example `reach_ids.json`:**

```json
[
  1000,
  1050,
  1100,
  1200,
  1234
]
```

The reach IDs correspond to the `mobidic_id` field in the processed river network. 

**Configuration example:**

```yaml
output_report_settings:
  output_format: Parquet
  reach_selection: file
  sel_file: data/reach_ids.json  # Path to JSON file
```

### Reservoir CSV files

When using reservoirs (`parameters.reservoirs.res_shape` is set), you need to provide CSV files for stage-storage curves, regulation curves, and regulation schedules.

**Stage-storage CSV (`stage_storage.csv`):**

Defines the relationship between water stage (elevation) and reservoir volume.

```csv
reservoir_id,stage_m,volume_m3
1,219.9,0.0
1,230.0,5000000.0
1,240.0,15000000.0
1,250.0,30000000.0
1,254.9,45000000.0
```

**Regulation curves CSV (`regulation_curves.csv`):**

Defines stage-discharge relationships for different regulation periods (e.g., winter vs summer operations).

```csv
reservoir_id,regulation_name,stage_m,discharge_m3s
1,winter,219.9,0.0
1,winter,230.0,5.0
1,winter,240.0,20.0
1,winter,250.0,50.0
1,summer,219.9,0.0
1,summer,230.0,2.0
1,summer,240.0,10.0
1,summer,250.0,30.0
```

**Regulation schedule CSV (`regulation_schedule.csv`):**

Defines which regulation curve to use during different time periods.

```csv
reservoir_id,start_date,end_date,regulation_name
1,2000-01-01,2000-05-31,winter
1,2000-06-01,2000-09-30,summer
1,2000-10-01,2000-12-31,winter
1,2001-01-01,2001-05-31,winter
1,2001-06-01,2001-09-30,summer
```

**Initial volumes CSV (`initial_conditions.csv`)** (optional):

Defines initial reservoir volumes. If not provided, volumes are auto-calculated from `z_max` field in the shapefile.

```csv
reservoir_id,volume_m3
1,20000000.0
```

**Configuration example:**

```yaml
parameters:
  reservoirs:
    res_shape: reservoirs/reservoirs.shp
    stage_storage: reservoirs/stage_storage.csv
    regulation_curves: reservoirs/regulation_curves.csv
    regulation_schedule: reservoirs/regulation_schedule.csv

initial_conditions:
  reservoir_volumes: reservoirs/initial_volumes.csv  # Optional

paths:
  reservoirs: output/reservoirs.parquet  # Consolidated output

output_states:
  reservoir_states: true  # Enable reservoir state output
```

## Logging

MOBIDICpy uses [loguru](https://github.com/Delgan/loguru) for logging throughout the package. Logging is automatically configured with INFO level when the package is imported, but can be customized using either programmatic configuration or YAML configuration.

### Functions

::: mobidic.utils.logging.configure_logger

::: mobidic.utils.logging.configure_logger_from_config

### Programmatic configuration

Configure logging directly in your Python code:

```python
from mobidic import configure_logger

# Set logging level
configure_logger(level="DEBUG")  # or INFO, WARNING, ERROR, CRITICAL

# Configure with log file
configure_logger(level="INFO", log_file="mobidic.log")

# Disable colorization (e.g., for CI environments)
configure_logger(level="INFO", colorize=False)
```

### YAML configuration

Configure logging via the `advanced` section in your configuration file:

```yaml
advanced:
  log_level: DEBUG  # or INFO, WARNING, ERROR, CRITICAL
  log_file: logs/mobidic.log  # Optional
```

Then load configuration and apply logging settings:

```python
from mobidic import load_config, configure_logger_from_config

# Load configuration
config = load_config("config.yaml")

# Apply logging settings from config
configure_logger_from_config(config)

# Proceed with preprocessing or simulation
gisdata = run_preprocessing(config)
```

### Log levels

- **DEBUG**: Detailed information for troubleshooting (includes function names and line numbers)
- **INFO**: Progress updates and general information (default)
- **WARNING**: Non-critical issues (e.g., missing optional files, automatic fallbacks)
- **ERROR**: Critical failures that stop execution
- **CRITICAL**: Severe errors that may lead to program termination

### Log output

**Console (stdout):**
- Always enabled with colorized output (unless `colorize=False`)
- Automatically adjusts format based on log level (DEBUG shows more detail)

**File (optional):**
- Specify via `log_file` parameter or `config.advanced.log_file`
- Automatic log rotation: rotates when file reaches 10 MB
- Log retention: keeps logs for 30 days
- Compression: rotated logs are compressed to `.zip` format
- Format adapts based on log level (DEBUG includes module/function/line)

### Usage example

```python
from mobidic import configure_logger, run_preprocessing, load_config

# Configure detailed logging for debugging
configure_logger(level="DEBUG", log_file="debug.log")

# Load configuration
config = load_config("basin.yaml")

# Run preprocessing (all operations will be logged)
gisdata = run_preprocessing(config)
# Output:
# 2026-01-21 10:30:45 | INFO     | Starting preprocessing workflow
# 2026-01-21 10:30:45 | DEBUG    | mobidic.config.parser:load_config:123 | Loading configuration from basin.yaml
# 2026-01-21 10:30:46 | INFO     | Stage 1/5: Loading configuration
# 2026-01-21 10:30:46 | INFO     | Stage 2/5: Reading raster data
# 2026-01-21 10:30:47 | DEBUG    | mobidic.preprocessing.gis_reader:grid_to_matrix:45 | Reading raster: dtm.tif
# ...
```
