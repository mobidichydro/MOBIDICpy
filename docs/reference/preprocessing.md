# Preprocessing Workflow

The preprocessing workflow module provides a high-level function that orchestrates the complete MOBIDIC preprocessing pipeline, from raw GIS data to simulation-ready inputs.

## Overview

The preprocessing workflow handles:

1. Loading and validating configuration
2. Reading all raster data (DTM, flow direction, soil parameters, etc.)
3. Processing the river network (topology, ordering, routing parameters)
4. Computing hillslope-reach mapping
5. Organizing all data into a consolidated `GISData` object

This is the recommended entry point for most users, as it handles all preprocessing steps automatically based on the configuration file.

## Functions

### Main Preprocessing Function

::: mobidic.preprocessing.preprocessor.run_preprocessing

## Workflow Stages

The preprocessing pipeline consists of five stages:

### Stage 1: Configuration Loading

Loads and validates the YAML configuration file:

```python
config = load_config(config_path)
```

All subsequent steps are driven by paths and parameters in this configuration.

### Stage 2: Raster Data Loading

Reads all raster files specified in the configuration:

- **Required rasters**: DTM, flow direction, flow accumulation
- **Soil parameters**: Wc0, Wg0, ks, and optional kf
- **Energy parameters**: CH, Alb (if energy balance enabled)
- **Flow coefficients**: gamma, kappa, beta, alpha (if provided as rasters)

All rasters are:
- Validated for consistent shape, resolution, and CRS
- Converted to numpy arrays with NaN for nodata
- Stored in the `GISData.grids` dictionary

### Stage 3: River Network Processing

Processes the river network shapefile:

```python
network = process_river_network(
    shapefile_path=config.vector_files.river_network.shp,
    join_single_tributaries=True,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    }
)
```

This step:
- Builds network topology
- Enforces binary tree structure
- Computes Strahler ordering
- Calculates routing parameters
- Determines calculation order

### Stage 4: Hillslope-Reach Mapping

Maps hillslope grid cells to river reaches:

```python
network = compute_hillslope_cells(network, grid_path)
reach_map = map_hillslope_to_reach(network, flowdir_path, flow_dir_type)
```

This establishes the connection between the distributed grid and the river network for lateral inflow routing.

### Stage 5: Data Consolidation

Packages everything into a `GISData` object:

```python
gisdata = GISData()
gisdata.grids = all_grid_data
gisdata.network = processed_network
gisdata.metadata = spatial_reference_info
```

The `GISData` object can then be saved for reuse or passed directly to the simulation.

## Complete Example

```python
from mobidic import load_config, run_preprocessing

# Load configuration
config = load_config("config.yaml")

# Run complete preprocessing
gisdata = run_preprocessing(config)

# Inspect results
print(f"Basin: {config.basin.id}")
print(f"Grid shape: {gisdata.metadata['shape']}")
print(f"Resolution: {gisdata.metadata['resolution']} m")
print(f"CRS: {gisdata.metadata['crs']}")
print(f"Grid variables: {list(gisdata.grids.keys())}")
print(f"Network reaches: {len(gisdata.network)}")
print(f"Strahler orders: {sorted(gisdata.network['strahler_order'].unique())}")

# Save for later use
gisdata.save(
    gisdata_path=config.paths.gisdata,
    network_path=config.paths.network
)
```

## Configuration Requirements

The preprocessing workflow requires the following configuration sections:

### Required Paths

```yaml
paths:
  meteodata: path/to/meteo.nc
  gisdata: path/to/gisdata.nc      # Output path
  network: path/to/network.parquet # Output path
  states: path/to/states/          # For simulation states
  output: path/to/output/          # For simulation outputs
```

### Required Vector Files

```yaml
vector_files:
  river_network:
    shp: path/to/river_network.shp
    id_field: REACH_ID  # Optional, for tracking original IDs
```

### Required Raster Files

```yaml
raster_files:
  dtm: path/to/dtm.tif
  flow_dir: path/to/flowdir.tif
  flow_acc: path/to/flowacc.tif
  Wc0: path/to/wc0.tif  # Capillary capacity
  Wg0: path/to/wg0.tif  # Gravitational capacity
  ks: path/to/ks.tif    # Hydraulic conductivity
```

### Required Raster Settings

```yaml
raster_settings:
  flow_dir_type: Grass  # or Arc
```

### Required Routing Parameters

```yaml
parameters:
  routing:
    method: Linear
    wcel: 5.0      # Wave celerity (m/s)
    Br0: 1.0       # Base channel width (m)
    NBr: 1.5       # Channel width exponent
    n_Man: 0.03    # Manning's n (s/m^(1/3))
```

See the [sample configuration](https://github.com/mobidichydro/mobidicpy/blob/main/examples/sample_config.yaml) for a complete example.

## Performance Considerations

### Execution Time

Typical preprocessing times (varies by basin size):

| Basin Size | Grid Cells | Reaches | Time |
|------------|-----------|---------|------|
| Small | 100×100 | 100 | ~10 sec |
| Medium | 1000×1000 | 1000 | ~1-2 min |
| Large | 5000×5000 | 5000 | ~10-20 min |

Most time is spent in:
1. Reading raster files (I/O bound)
2. Hillslope-reach mapping (computation bound)

### Memory Usage

Memory usage scales with grid size:

- Small basin (100×100): ~100 MB
- Medium basin (1000×1000): ~1-2 GB
- Large basin (5000×5000): ~10-20 GB

To reduce memory usage:
- Use grid resolution degradation
- Process in tiles (future feature)
- Increase virtual memory/swap

### Caching Results

After preprocessing once, save the results:

```python
# First run: expensive preprocessing
gisdata = run_preprocessing(config)
gisdata.save("output/gisdata.nc", "output/network.parquet")

# Subsequent runs: fast loading
from mobidic import GISData
gisdata = GISData.load("output/gisdata.nc", "output/network.parquet")
```

Loading is ~100× faster than preprocessing from scratch.

## Error Handling

The preprocessing workflow performs comprehensive validation:

### Configuration Validation

- All required paths and parameters are present
- Numeric parameters are within valid ranges
- File paths exist (if validation enabled)

### Spatial Consistency

- All rasters have the same shape
- All rasters have the same resolution
- All rasters have compatible CRS
- Network CRS matches raster CRS (or is reprojected)

### Data Quality

- Rasters have valid nodata handling
- Network has no topological errors
- Flow direction is valid (no invalid direction codes)
- Flow accumulation is consistent with flow direction

### Error Messages

All errors provide:
- Clear description of the problem
- Location (file path, line number)
- Suggested fixes
- Links to documentation

## Logging

Preprocessing operations are logged using loguru:

```python
from mobidic import configure_logger, run_preprocessing

# Set logging level
configure_logger(level="INFO")  # or DEBUG, WARNING, ERROR

# Run preprocessing with logging
gisdata = run_preprocessing(config)
```

Log levels:
- **DEBUG**: Detailed information for troubleshooting
- **INFO**: Progress updates for each stage
- **WARNING**: Non-critical issues (e.g., missing optional files)
- **ERROR**: Critical failures that stop execution

Logs are written to:
- Console (stdout)
- File (if specified in `config.advanced.log_file`)

## Customization

For advanced users who need custom preprocessing:

### Custom Grid Processing

```python
from mobidic import load_config, GISData, grid_to_matrix

config = load_config("config.yaml")
gisdata = GISData()

# Custom raster processing
dtm_data = grid_to_matrix(config.raster_files.dtm)
dtm_processed = custom_processing(dtm_data['data'])
gisdata.grids['dtm'] = dtm_processed

# ... add other grids
```

### Custom Network Processing

```python
from mobidic import process_river_network

# Custom routing parameters
network = process_river_network(
    shapefile_path="river_network.shp",
    join_single_tributaries=False,  # Keep all reaches
    routing_params={
        "wcel": 7.0,  # Custom wave celerity
        "Br0": 2.0,   # Custom base width
        "NBr": 1.3,   # Custom width exponent
        "n_Man": 0.025,  # Custom Manning coefficient
    }
)
```

### Skip Stages

```python
# If you already have processed network, skip that stage
from mobidic import load_network, GISData

gisdata = GISData()
# ... load grids manually
gisdata.network = load_network("existing_network.parquet")
```

## Integration with Simulation

After preprocessing, the `GISData` object is ready for simulation:

```python
from mobidic import run_preprocessing, run_simulation

# Preprocessing
gisdata = run_preprocessing(config)

# Simulation (future implementation)
results = run_simulation(config, gisdata)
```

The simulation will access:
- `gisdata.grids`: Spatially-distributed parameters
- `gisdata.network`: River network topology and parameters
- `gisdata.metadata`: Spatial reference information

## MATLAB Translation

This module replaces several MATLAB scripts:

- `buildgis_mysql_include.m` → Configuration-driven preprocessing
- `main_preprocessing.m` → Complete workflow orchestration
- Various helper scripts → Unified Python functions

The Python implementation provides:
- Cleaner separation of concerns
- Better error handling
- Structured logging
- Type safety and validation
- Modular and testable code
