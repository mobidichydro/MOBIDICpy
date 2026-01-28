# Examples

This page provides practical examples demonstrating the main features of MOBIDICpy. All example scripts are available in the `examples/` directory of the repository.

## Configuration parser

This example shows how to load and validate MOBIDIC configuration files.

**Script**: `examples/demo_config_parser.py`

```python
from mobidic import load_config, configure_logger

# Configure logger
configure_logger(level="INFO")

# Load and validate configuration
config = load_config("examples/sample_config.yaml")

# Access configuration values
print(f"Basin ID: {config.basin.id}")
print(f"Parameter Set: {config.basin.paramset_id}")
print(f"DTM path: {config.raster_files.dtm}")
print(f"Soil scheme: {config.simulation.soil_scheme}")
print(f"Time step: {config.simulation.timestep} seconds")

# Access parameter values with validation
print(f"Hydraulic conductivity: {config.parameters.soil.ks} mm/h")
print(f"Wave celerity: {config.parameters.routing.wcel} m/s")
```

**What it demonstrates:**

- Loading YAML configuration files
- Accessing nested configuration values
- Automatic validation using Pydantic
- Type-safe configuration access

---

## River network processing

Process a river network shapefile to create a routing network with topology and parameters.

**Script**: `examples/demo_process_network.py`

```python
from mobidic import load_config, process_river_network, save_network

# Load configuration
config = load_config("examples/Arno/Arno.yaml")

# Process river network
network = process_river_network(
    shapefile_path=config.vector_files.river_network.shp,
    join_single_tributaries=True,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    },
)

# Display network summary
print(f"Total reaches: {len(network)}")
print(f"Strahler orders: {sorted(network['strahler_order'].unique())}")
print(f"Max calculation order: {network['calc_order'].max()}")
print(f"Total network length: {network['length_m'].sum():.1f} m")
print(f"Mean channel width: {network['width_m'].mean():.2f} m")

# Export to GeoParquet
save_network(network, config.paths.network, format="parquet")
```

**What it demonstrates:**

- Reading and processing river network shapefiles
- Building network topology
- Computing Strahler ordering
- Calculating routing parameters (width, lag time, storage coefficient)
- Exporting to GeoParquet format

---

## Hillslope-reach mapping

Map hillslope grid cells to river reaches for lateral inflow routing.

**Script**: `examples/demo_hill_reach_map.py`

```python
import rasterio
import numpy as np
from mobidic import (
    load_config,
    configure_logger,
    process_river_network,
    compute_hillslope_cells,
    map_hillslope_to_reach,
)

# Load configuration
config = load_config("examples/Arno/Arno.yaml")
configure_logger(level="INFO")

# Step 1: Process river network
network = process_river_network(
    shapefile_path=config.vector_files.river_network.shp,
    join_single_tributaries=True,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    },
)

# Step 2: Rasterize reaches onto grid
network = compute_hillslope_cells(
    network=network,
    grid_path=config.raster_files.flow_dir,
)

# Step 3: Map hillslope cells to reaches
reach_map = map_hillslope_to_reach(
    network=network,
    flowdir_path=config.raster_files.flow_dir,
    flow_dir_type=config.raster_settings.flow_dir_type,
)

# Display results
print(f"Reach map shape: {reach_map.shape}")
print(f"Unique reaches: {len(set(reach_map[~np.isnan(reach_map)]))}")
print(f"Cells assigned to reaches: {np.sum(~np.isnan(reach_map))}")

# Export reach map as raster
with rasterio.open(config.raster_files.flow_dir) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open("output/reach_map.tif", "w", **profile) as dst:
        dst.write(reach_map.astype(rasterio.float32), 1)
```

**What it demonstrates:**

- Rasterizing river reaches onto the model grid
- Following flow paths from hillslope to reaches
- Creating a reach assignment map
- Exporting spatial results as GeoTIFF

---

## Meteorological data preprocessing

Convert meteorological data from MATLAB format to CF-compliant NetCDF.

**Script**: `examples/demo_meteo_mat_to_nc.py`

```python
from pathlib import Path
from mobidic import MeteoData, convert_mat_to_netcdf

# Example 1: Direct conversion
convert_mat_to_netcdf(
    mat_file="examples/Arno/meteodata/meteodata.mat",
    output_file="examples/Arno/meteodata/meteodata.nc",
    compression_level=4,
    add_metadata={
        "basin": "Arno",
        "basin_id": "Event_November_2023",
        "description": "Meteorological forcing data for Arno basin flood event",
    },
)

# Example 2: Load and inspect before saving
meteo_data = MeteoData.from_mat("examples/Arno/meteodata/meteodata.mat")

print(f"{meteo_data}")
print(f"Date range: {meteo_data.start_date} to {meteo_data.end_date}")

# Inspect station counts
for var_name, stations in meteo_data.stations.items():
    print(f"{var_name}: {len(stations)} stations")

# Examine first precipitation station
if len(meteo_data.stations["precipitation"]) > 0:
    first_station = meteo_data.stations["precipitation"][0]
    print(f"\nFirst precipitation station:")
    print(f"  Code: {first_station['code']}")
    print(f"  Location: ({first_station['x']:.2f}, {first_station['y']:.2f})")
    print(f"  Elevation: {first_station['elevation']:.1f} m")

# Example 3: Read back from NetCDF
meteo_from_nc = MeteoData.from_netcdf("examples/Arno/meteodata/meteodata.nc")
print(f"\nLoaded from NetCDF: {meteo_from_nc}")
```

**What it demonstrates:**

- Converting MATLAB .mat files to NetCDF
- Loading and inspecting meteorological data
- Reading CF-compliant NetCDF files
- Working with station-based time series data

---

## Complete preprocessing workflow

The complete preprocessing workflow demonstrates how to process all GIS data and prepare it for simulation.

**Script**: `examples/run_preprocessing.py`

```python
from pathlib import Path
from mobidic import load_config, run_preprocessing, GISData, configure_logger

# Configure logging
configure_logger(level="INFO")

# Load configuration
config = load_config("examples/Arno/Arno.yaml")

# Run complete preprocessing pipeline
gisdata = run_preprocessing(config)

# Save preprocessed data
gisdata.save(
    gisdata_path=config.paths.gisdata,
    network_path=config.paths.network
)

# Display summary
print(f"Basin: {config.basin.id}")
print(f"Grid shape: {gisdata.metadata['shape']}")
print(f"Network reaches: {len(gisdata.network)}")
print(f"Total network length: {gisdata.network['length_m'].sum() / 1000:.1f} km")

# Later: reload preprocessed data quickly
loaded_gisdata = GISData.load(
    gisdata_path=config.paths.gisdata,
    network_path=config.paths.network
)
```

**What it demonstrates:**

- Loading configuration from YAML
- Running the complete preprocessing pipeline
- Saving preprocessed data for reuse
- Loading preprocessed data quickly

---

## Complete simulation: Arno river catchment

The Arno river example demonstrates a complete MOBIDIC simulation workflow, from preprocessing through simulation to visualization.

**Script**: `examples/run_example_Arno.py`

```python
from mobidic import load_config, run_preprocessing, MeteoData, Simulation

# Load configuration
config = load_config("examples/Arno/Arno.yaml")

# Run or load preprocessing
gisdata = run_preprocessing(config)

# Convert and load meteorological data
meteo_data = MeteoData.from_mat("examples/Arno/meteodata/meteodata.mat")
meteo_data.to_netcdf(config.paths.meteodata)
forcing = MeteoData.from_netcdf(config.paths.meteodata)

# Run simulation
sim = Simulation(gisdata, forcing, config)
results = sim.run(start_date=forcing.start_date, end_date=forcing.end_date)
```

**What it demonstrates:**

- Complete MOBIDIC workflow from configuration to results
- GIS preprocessing with caching
- Meteorological data conversion (MATLAB .mat to NetCDF)
- Running hydrological simulation with the `Simulation` class
- Automatic output saving (discharge, lateral inflow, states)
- Result visualization with matplotlib

**Output files:**

- Preprocessed GIS data: `Arno/gisdata/gisdata.nc`
- Processed network: `Arno/gisdata/network.parquet`
- Discharge time series: `Arno/output/discharge.parquet`
- Final states: `Arno/output/states/states_*.nc`

---

## Station vs raster forcing comparison

Compare station-based forcing (with spatial interpolation) against pre-interpolated raster forcing.

**Script**: `examples/run_example_Arno_raster_forcing.py`

```python
from mobidic import MeteoData, MeteoRaster, Simulation

# Run 1: Station-based with forcing output
config.output_forcing_data.meteo_data = True
forcing_stations = MeteoData.from_netcdf(config.paths.meteodata)
sim1 = Simulation(gisdata, forcing_stations, config)
results1 = sim1.run(start_date, end_date)

# Run 2: Raster-based (using exported data)
forcing_raster = MeteoRaster.from_netcdf("output/meteo_forcing.nc")
sim2 = Simulation(gisdata, forcing_raster, config)
results2 = sim2.run(start_date, end_date)
```

**What it demonstrates:**

- Exporting interpolated meteorological data as raster forcing
- Using raster forcing for subsequent runs (faster, no interpolation)

**Use cases:** Calibration runs, scenario analysis, large domains

---

## Simulation restart capability

The restart example demonstrates MOBIDICpy's ability to save intermediate simulation states and continue the simulation from a saved state.

**Script**: `examples/run_example_Arno_restart.py`

```python
from pathlib import Path
from mobidic import load_config, load_gisdata, MeteoData, Simulation

# Load configuration and data
config = load_config("examples/Arno/Arno.yaml")
gisdata = load_gisdata(config.paths.gisdata, config.paths.network)
forcing = MeteoData.from_netcdf(config.paths.meteodata)

# First run: simulate to midpoint
sim1 = Simulation(gisdata, forcing, config)
results_1 = sim1.run(start_date=start_date, end_date=restart_point)
# States automatically saved to config.paths.states

# Second run: restart from saved state
sim2 = Simulation(gisdata, forcing, config)
state_file = Path(config.paths.states) / "states_001.nc"
sim2.set_initial_state(state_file=state_file, time_index=-1)
results_2 = sim2.run(start_date=restart_point, end_date=end_date)

# Validation: compare with continuous run
sim_continuous = Simulation(gisdata, forcing, config)
results_continuous = sim_continuous.run(start_date=start_date, end_date=end_date)
```

**What it demonstrates:**

- Saving intermediate simulation states to NetCDF files
- Loading states from file using `set_initial_state()`
- Continuing simulations from saved checkpoints
- Validating restart accuracy against continuous run

**Use cases:**

- **Long-term simulations**: Break simulations into manageable chunks
- **Checkpoint recovery**: Resume after system interruptions
- **Multi-stage modeling**: Apply different parameters or forcings in different periods
- **Real-time operations**: Save current state and restart with new forecast data

---

## Reservoir routing: Arno river basin with reservoirs

This example demonstrates how to configure and simulate reservoirs with time-varying regulation curves and stage-discharge relationships.

**Script**: `examples/run_example_Arno_reservoirs.py`

```python
from pathlib import Path
from mobidic import (
    load_config,
    run_preprocessing,
    MeteoData,
    Simulation,
    GISData,
    configure_logger,
)

# Configure logging
configure_logger(level="INFO")

# Load reservoir-enabled configuration
config = load_config("examples/Arno/Arno.reservoirs.yaml")

# Run preprocessing with reservoirs
gisdata = run_preprocessing(config)

# The preprocessing automatically:
# - Reads reservoir polygon shapefile
# - Rasterizes polygons to identify basin pixels
# - Loads stage-storage curves from CSV
# - Loads regulation curves and schedules from CSV
# - Identifies inlet/outlet reaches by topology
# - Auto-calculates initial volumes from z_max

# Display reservoir summary
if gisdata.reservoirs:
    print(f"\nProcessed {len(gisdata.reservoirs)} reservoirs:")
    for reservoir in gisdata.reservoirs:
        print(f"  Reservoir {reservoir.id}: {reservoir.name}")
        print(f"    Basin pixels: {len(reservoir.basin_pixels)}")
        print(f"    Inlet reaches: {reservoir.inlet_reaches}")
        print(f"    Outlet reach: {reservoir.outlet_reach}")
        print(f"    Initial volume: {reservoir.initial_volume:.0f} m³")
        print(f"    Stage-storage points: {len(reservoir.stage_storage_curve)}")
        print(f"    Regulation periods: {len(reservoir.period_times)}")

# Save preprocessed data including reservoirs
gisdata.save(
    gisdata_path=config.paths.gisdata,
    network_path=config.paths.network,
    reservoirs_path=config.paths.reservoirs,
)

# Load meteorological data
forcing = MeteoData.from_netcdf(config.paths.meteodata)

# Run simulation with reservoirs
sim = Simulation(gisdata, forcing, config)
results = sim.run(start_date=forcing.start_date, end_date=forcing.end_date)
```

**Configuration example** (`Arno.reservoirs.yaml`):

```yaml
parameters:
  reservoirs:
    res_shape: reservoirs/reservoirs.shp
    stage_storage: reservoirs/stage_storage.csv
    regulation_curves: reservoirs/regulation_curves.csv
    regulation_schedule: reservoirs/regulation_schedule.csv

initial_conditions:
  reservoir_volumes: reservoirs/initial_volumes.csv  # Optional, otherwise set to 100% volume

paths:
  reservoirs: gisdata/reservoirs.parquet

output_states:
  reservoir_states: true  # Save reservoir states (volume, stage, discharge)
```

**Required CSV files:**

1. **stage_storage.csv**: Stage-volume relationship
   ```csv
   reservoir_id,stage_m,volume_m3
   1,219.9,0.0
   1,230.0,5000000.0
   1,240.0,15000000.0
   1,250.0,30000000.0
   1,254.9,45000000.0
   ```

2. **regulation_curves.csv**: Stage-discharge relationships by period
   ```csv
   reservoir_id,regulation_name,stage_m,discharge_m3s
   1,winter,219.9,0.0
   1,winter,240.0,20.0
   1,winter,250.0,50.0
   1,summer,219.9,0.0
   1,summer,240.0,10.0
   1,summer,250.0,30.0
   ```

3. **regulation_schedule.csv**: Seasonal regulation switching
   ```csv
   reservoir_id,start_date,end_date,regulation_name
   1,2000-01-01,2000-05-31,winter
   1,2000-06-01,2000-09-30,summer
   1,2000-10-01,2000-12-31,winter
   ```

4. **initial_volumes.csv** (optional): Initial reservoir volumes
   ```csv
   reservoir_id,volume_m3
   1,20000000.0
   ```

**What it demonstrates:**

- Configuring reservoirs with shapefiles and CSV data files
- Processing reservoir data during preprocessing
- Time-varying regulation curves (e.g., seasonal winter/summer operations)
- Inlet/outlet reach detection by network topology
- Reservoir state output (volume, stage, discharge) to NetCDF

**Output files:**

- Preprocessed reservoirs: `Arno/gisdata/reservoirs.parquet`
- Reservoir states: `Arno/output/states/states_*.nc` (includes reservoir volume, stage, discharge)
- Discharge time series: `Arno/output/discharge.parquet` (includes reservoir effects)

**Visualization script**: `examples/run_example_Arno_reservoirs_plots.py`

---

## Design storm simulation with hyetograph generation

This example demonstrates how to generate synthetic design storm hyetographs from IDF (Intensity-Duration-Frequency) parameters and run a design flood simulation.

```python
from datetime import datetime
from pathlib import Path
from mobidic import load_config, load_gisdata, Simulation
from mobidic.preprocessing.hyetograph import HyetographGenerator

# Load hyetograph-enabled configuration
config_file = Path("Arno_hyetograph.yaml")
config = load_config(config_file)

# Load preprocessed GIS data
gisdata = load_gisdata(config.paths.gisdata, config.paths.network)

# Generate hyetograph forcing from configuration
# All parameters (IDF rasters, duration, timestep, method) are read from config
forcing = HyetographGenerator.from_config(
    config=config,
    base_path=config_file.parent,
    start_time=datetime(2000, 1, 1)  # Reference start time
)

# Inspect generated forcing
print(f"Forcing date range: {forcing.start_date} to {forcing.end_date}")
print(f"Variables: {forcing.variables}")
print(f"Grid shape: {forcing.grid_metadata['shape']}")

# Run simulation
sim = Simulation(gisdata, forcing, config)
results = sim.run(forcing.start_date, forcing.end_date)

# Results are saved automatically to output directory
```

**Configuration example** (`Arno_hyetograph.yaml`):

```yaml
paths:
  gisdata: gisdata/Arno_gisdata.nc
  network: gisdata/Arno_net.parquet
  hyetograph: output/design_storm.nc  # Output path for generated hyetograph
  states: states/
  output: output/

raster_files:
  dtm: raster/dtm.tif  # Reference grid for IDF resampling
  # ... other rasters

hyetograph:
  a_raster: idf/a.tif        # IDF 'a' parameter (scale)
  n_raster: idf/n.tif        # IDF 'n' parameter (exponent)
  k_raster: idf/k30.tif      # Return period factor (30-year event)
  duration_hours: 48         # Storm duration
  timestep_hours: 1          # Time step
  hyetograph_type: chicago_decreasing  # Chicago method (after-peak)
  ka: 0.8                    # Areal reduction factor
```

**What it demonstrates:**

- Generating synthetic design storm hyetographs from IDF parameters
- Using spatially distributed IDF rasters (a, n, k)
- Automatic resampling of IDF parameters to match model grid
- Chicago decreasing hyetograph method

**IDF formula:**

The Depth-Duration-Frequency (DDF) relationship is:

$$DDF(t) = k_a \cdot k \cdot a \cdot t^n$$

where:

- $DDF(t)$ is cumulative precipitation depth (mm) for duration $t$
- $k_a$ is areal reduction factor
- $k$ is return period factor (from raster)
- $a$ is IDF scale parameter (from raster)
- $n$ is IDF exponent (from raster)
- $t$ is duration in hours

---

## Sample Configuration

A sample configuration file is provided at `examples/sample_config.yaml`. This file includes:

- Basin metadata and identification
- Input/output file paths
- Raster and vector data sources
- Soil, energy, routing, and groundwater parameters
- Initial conditions
- Simulation settings (time step, schemes)
- Output configuration

Refer to this file as a template when creating new configurations.

---

## Additional resources

### How to run the examples

All example scripts can be run from the repository root:

```bash
# Complete Arno basin simulation
python examples/run_example_Arno.py

# Visualization of results
python examples/run_example_Arno_plots.py

# Station vs raster forcing comparison
python examples/run_example_Arno_raster_forcing.py

# Complete Arno basin with reservoir routing
python examples/run_example_Arno_reservoirs.py

# Visualization of results
python examples/run_example_Arno_reservoirs_plots.py

# Restart capability demonstration
python examples/run_example_Arno_restart.py

# Complete preprocessing
python examples/run_preprocessing.py

# Configuration parser demo
python examples/demo_config_parser.py

# Network processing
python examples/demo_process_network.py

# Hillslope-reach mapping
python examples/demo_hill_reach_map.py

# Meteorological preprocessing
python examples/demo_meteo_mat_to_nc.py
```

### Test data

Example data for the Arno river catchment is provided in `examples/Arno/`:

- Configuration files: `Arno.yaml`, `Arno.reservoirs.yaml`
- River network shapefile
- Raster files (DTM, flow direction, soil parameters)
- Meteorological data
- Reservoir data (shapefiles, stage-storage curves, regulation curves/schedules)

### More information

- See the [API Reference](reference/index.md) for detailed function documentation
- See the [Introduction](introduction.md) for model background
- See the [Development Guide](development.md) for contributing
