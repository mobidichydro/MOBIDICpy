# Preprocessing workflow

The preprocessing module provides a high-level function that runs the complete MOBIDIC preprocessing, from raw GIS data to inputs that can be directly used in the simulation.

## Overview

The preprocessing workflow handles:

1. Loading and validating configuration
2. Reading all raster data (DTM, flow direction, soil parameters, etc.)
3. Processing the river network (topology, ordering, routing parameters)
4. Computing hillslope-reach mapping
5. Processing reservoirs (optional: polygons, stage-storage curves, regulation curves/schedules)
6. Organizing all data into a consolidated `GISData` object

This is the recommended entry point for most users, as it handles all preprocessing steps automatically based on the configuration file.

## Functions

### Main preprocessing function

::: mobidic.preprocessing.preprocessor.run_preprocessing

### Reservoir preprocessing function

::: mobidic.preprocessing.reservoirs.process_reservoirs

## Workflow stages

The preprocessing pipeline consists of up to seven stages (stages 6-7 are optional if reservoirs are configured):

### Stage 1: configuration loading

Loads and validates the YAML configuration file:

```python
config = load_config(config_path)
```

All subsequent steps are driven by paths and parameters in this configuration.

### Stage 2: raster data loading

Reads all raster files specified in the configuration:

- **Required rasters**: DTM, flow direction, flow accumulation
- **Soil parameters**: Wc0, Wg0, ks, and optional kf
- **Energy parameters**: CH, Alb (if energy balance enabled)
- **Flow coefficients**: gamma, kappa, beta, alpha (if provided as rasters)

All rasters are:
- Validated for consistent shape, resolution, and CRS
- Converted to numpy arrays with NaN for nodata
- Stored in the `GISData.grids` dictionary

### Stage 3: river network processing

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

### Stage 4: hillslope-reach mapping

Maps hillslope grid cells to river reaches:

```python
network = compute_hillslope_cells(network, grid_path)
reach_map = map_hillslope_to_reach(network, flowdir_path, flow_dir_type)
```

This establishes the connection between the distributed grid and the river network for lateral inflow routing.

### Stage 5: data consolidation

Packages everything into a `GISData` object:

```python
gisdata = GISData()
gisdata.grids = all_grid_data
gisdata.network = processed_network
gisdata.metadata = spatial_reference_info
```

The `GISData` object can then be saved for reuse or passed directly to the simulation.

### Stage 6: reservoir preprocessing (optional)

If reservoirs are configured (`config.parameters.reservoirs.res_shape` is set), process reservoir data:

```python
reservoirs = process_reservoirs(
    res_shape_path=config.parameters.reservoirs.res_shape,
    stage_storage_path=config.parameters.reservoirs.stage_storage,
    regulation_curves_path=config.parameters.reservoirs.regulation_curves,
    regulation_schedule_path=config.parameters.reservoirs.regulation_schedule,
    initial_volumes_path=config.initial_conditions.reservoir_volumes,
    grid_transform=gisdata.metadata['transform'],
    grid_shape=gisdata.metadata['shape'],
    network=gisdata.network,
)
gisdata.reservoirs = reservoirs
```

This step:
- Reads reservoir polygon shapefile
- Rasterizes polygons to identify basin pixels
- Loads stage-storage curves from CSV
- Loads regulation curves and schedules from CSV
- Identifies inlet/outlet reaches by network topology
- Auto-calculates initial volumes from z_max if not provided
- Consolidates all reservoir data into Reservoirs container

### Stage 7: reservoir I/O (optional)

Save/load processed reservoir data:

```python
# Save reservoirs to GeoParquet
gisdata.save(
    gisdata_path=config.paths.gisdata,
    network_path=config.paths.network,
    reservoirs_path=config.paths.reservoirs,
)

# Load reservoirs from GeoParquet
gisdata = GISData.load(
    gisdata_path=config.paths.gisdata,
    network_path=config.paths.network,
    reservoirs_path=config.paths.reservoirs,
)
```

## Complete example

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

**Note:** To configure logging behavior (level, output file, etc.), see the [Logging](config.md#logging) section in the Configuration reference.

## Configuration requirements

The preprocessing workflow requires the following configuration sections:

### Required paths

```yaml
paths:
  meteodata: path/to/meteo.nc
  gisdata: path/to/gisdata.nc      # Output path
  network: path/to/network.parquet # Output path
  states: path/to/states/          # For simulation states
  output: path/to/output/          # For simulation outputs
```

### Required vector files

```yaml
vector_files:
  river_network:
    shp: path/to/river_network.shp
    id_field: REACH_ID  # Optional, for tracking original IDs
```

### Required raster files

```yaml
raster_files:
  dtm: path/to/dtm.tif
  flow_dir: path/to/flowdir.tif
  flow_acc: path/to/flowacc.tif
  Wc0: path/to/wc0.tif  # Capillary capacity
  Wg0: path/to/wg0.tif  # Gravitational capacity
  ks: path/to/ks.tif    # Hydraulic conductivity
```

### Required raster settings

```yaml
raster_settings:
  flow_dir_type: Grass  # or Arc
```

### Required routing parameters

```yaml
parameters:
  routing:
    method: Linear
    wcel: 5.0      # Wave celerity (m/s)
    Br0: 1.0       # Base channel width (m)
    NBr: 1.5       # Channel width exponent
    n_Man: 0.03    # Manning's n (s/m^(1/3))
```

### Optional reservoir parameters

```yaml
parameters:
  reservoirs:
    res_shape: path/to/reservoirs.shp                    # Reservoir polygon shapefile
    stage_storage: path/to/stage_storage.csv             # Stage-storage curves
    regulation_curves: path/to/regulation_curves.csv     # Stage-discharge curves
    regulation_schedule: path/to/regulation_schedule.csv # Regulation period schedule

initial_conditions:
  reservoir_volumes: path/to/initial_volumes.csv  # Optional (auto-calculated if omitted)

paths:
  reservoirs: path/to/reservoirs.parquet  # Output path for consolidated reservoir data

output_states:
  reservoir_states: true  # Enable reservoir state output
```

**CSV file formats:**

- **stage_storage.csv**: Columns: `reservoir_id`, `stage_m`, `volume_m3`
- **regulation_curves.csv**: Columns: `reservoir_id`, `regulation_name`, `stage_m`, `discharge_m3s`
- **regulation_schedule.csv**: Columns: `reservoir_id`, `start_date`, `end_date`, `regulation_name`
- **initial_volumes.csv**: Columns: `reservoir_id`, `volume_m3` (optional, defaults to auto-calculation from z_max)

See the [sample configuration](https://github.com/mobidichydro/mobidicpy/blob/main/examples/sample_config.yaml) for a complete example.

## Error handling

The preprocessing workflow performs comprehensive validation:

### Configuration validation

- All required paths and parameters are present
- Numeric parameters are within valid ranges
- File paths exist (if validation enabled)

### Spatial consistency

- All rasters have the same shape
- All rasters have the same resolution
- All rasters have compatible CRS
- Network CRS matches raster CRS (or is reprojected)

### Data quality

- Rasters have valid nodata handling
- Network has no topological errors
- Flow direction is valid (no invalid direction codes)
- Flow accumulation is consistent with flow direction

## Advanced usage

### Skipping preprocessing stages

If you already have some preprocessed data, you can skip certain stages and load existing files:

```python
from mobidic import load_network, load_gisdata, GISData

# Option 1: Load complete preprocessed data
gisdata = load_gisdata("existing_gisdata.nc", "existing_network.parquet")

# Option 2: Load network separately
gisdata = GISData()
# ... load grids manually
gisdata.network = load_network("existing_network.parquet")
```

## Integration with simulation

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


