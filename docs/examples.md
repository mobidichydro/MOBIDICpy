# Examples

This page provides practical examples demonstrating the main features of MOBIDICpy. All example scripts are available in the `examples/` directory of the repository.

## Complete Preprocessing Workflow

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

## Configuration Parser

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

## River Network Processing

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

## Hillslope-Reach Mapping

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

## Meteorological Data Preprocessing

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

## Sample Configuration

A complete annotated configuration file is provided at `examples/sample_config.yaml`. This file includes:

- Basin metadata and identification
- Input/output file paths
- Raster and vector data sources
- Soil, energy, routing, and groundwater parameters
- Initial conditions
- Simulation settings (time step, schemes)
- Output configuration

Refer to this file as a template when creating your own configurations.

---

## Additional Resources

### Running Examples

All example scripts can be run from the repository root:

```bash
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

### Test Data

Example data for the Arno basin is provided in `examples/Arno/`:
- Configuration file: `Arno.yaml`
- River network shapefile
- Raster files (DTM, flow direction, soil parameters)
- Meteorological data

### More Information

- See the [API Reference](reference/index.md) for detailed function documentation
- See the [Introduction](introduction.md) for model background
- See the [Development Guide](development.md) for contributing
