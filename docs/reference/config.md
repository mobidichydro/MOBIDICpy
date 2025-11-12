# Configuration

The configuration module provides a schema-driven approach to loading and validating MOBIDIC model configurations using Pydantic and YAML.

## Overview

Configuration files define all aspects of a MOBIDIC simulation, including:

- **Basin metadata**: ID, parameter set, baricenter coordinates
- **Input/output paths**: Meteodata, GIS data, network, states, output directories
- **Raster and vector data sources**: DTM, flow direction/accumulation, soil/energy parameters, river network
- **Model parameters**: Organized into four subsections:
    - `soil`: Hydraulic conductivity, water holding capacity, flow coefficients
    - `energy`: Thermal properties, turbulent exchange, albedo
    - `routing`: Channel routing method, wave celerity, Manning coefficient
    - `groundwater`: Model type, global loss
    - `multipliers`: Calibration factors
- **Initial conditions**: Initial state (hillslope runoff, soil saturation)
- **Simulation settings**: Time step, resampling, soil/energy balance schemes
- **Output options**: State outputs (NetCDF), report outputs (CSV/Parquet)

The configuration system ensures all required fields are present, validates ranges and consistency, and provides sensible defaults for optional parameters.

## Functions

::: mobidic.config.parser.load_config

::: mobidic.config.parser.save_config

## Classes

### Main Configuration

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

### Nested Models

The configuration is organized hierarchically using nested Pydantic models:

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

::: mobidic.config.schema.Multipliers

::: mobidic.config.schema.InitialConditions

::: mobidic.config.schema.Simulation

::: mobidic.config.schema.OutputStates

::: mobidic.config.schema.OutputStatesSettings

::: mobidic.config.schema.OutputReport

::: mobidic.config.schema.OutputReportSettings

::: mobidic.config.schema.Advanced

## Configuration Structure

See the [sample configuration file](https://github.com/mobidichydro/mobidicpy/blob/main/examples/sample_config.yaml) for a complete annotated example with all available options and their descriptions.

## Validation

The configuration system performs extensive validation:

- **Type checking**: All fields are type-checked by Pydantic
- **Range validation**: Numeric parameters are validated against physical constraints (e.g., albedo 0-1, saturation 0-1)
- **Required fields**: Missing required fields raise validation errors
- **Consistency checks**: Inter-field dependencies are validated (e.g., if `reach_selection='file'`, then `sel_file` must be provided)
- **Path validation**: File paths can optionally be validated for existence

Invalid configurations will raise `pydantic.ValidationError` with detailed error messages.
