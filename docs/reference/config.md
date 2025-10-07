# Configuration

The configuration module provides a schema-driven approach to loading and validating MOBIDIC model configurations using Pydantic and YAML.

## Overview

Configuration files define all aspects of a MOBIDIC simulation, including:

- Basin metadata
- Input/output file paths
- Raster and vector data sources
- Model parameters (soil, energy, routing, groundwater)
- Initial conditions
- Simulation settings
- Output options

The configuration system ensures all required fields are present, validates ranges and consistency, and provides sensible defaults for optional parameters.

## Functions

::: mobidic.config.parser.load_config

## Classes

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

## Example

```python
from mobidic import load_config

# Load and validate configuration
config = load_config("config.yaml")

# Access configuration sections
print(f"Basin ID: {config.basin.id}")
print(f"DTM path: {config.raster_files.dtm}")
print(f"Soil scheme: {config.simulation.soil_scheme}")
print(f"Time step: {config.simulation.dt} seconds")

# Access parameter values
ksat = config.parameters.soil.ksat
print(f"Hydraulic conductivity: {ksat} mm/h")
```

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
