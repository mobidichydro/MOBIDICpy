# API Reference

MOBIDICpy provides a comprehensive Python API for distributed hydrological modeling. The API is organized into functional modules for easy navigation and usage.

## Overview

The MOBIDICpy public API is designed to be simple and functional, exposing only the essential functions needed for preprocessing, configuration, and model setup.

## API Modules

### [Configuration](config.md)
Load and validate YAML configuration files using Pydantic schemas.

- [`load_config()`](config.md#mobidic.config.parser.load_config) - Load and validate configuration from YAML
- [`MOBIDICConfig`](config.md#mobidic.config.schema.MOBIDICConfig) - Complete configuration schema

### [GIS Data I/O](gis_io.md)
Read and write geospatial data (rasters and vectors).

- [`grid_to_matrix()`](gis_io.md#mobidic.preprocessing.gis_reader.grid_to_matrix) - Read raster files (GeoTIFF)
- [`read_shapefile()`](gis_io.md#mobidic.preprocessing.gis_reader.read_shapefile) - Read vector files (Shapefile)

### [Grid Operations](grid_ops.md)
Process and transform gridded data (resolution degradation, flow direction conversion).

- [`degrade_raster()`](grid_ops.md#mobidic.preprocessing.grid_operations.degrade_raster) - Coarsen raster resolution
- [`degrade_flow_direction()`](grid_ops.md#mobidic.preprocessing.grid_operations.degrade_flow_direction) - Coarsen flow direction grids
- [`convert_to_mobidic_notation()`](grid_ops.md#mobidic.preprocessing.grid_operations.convert_to_mobidic_notation) - Convert flow direction to MOBIDIC notation

### [River Network Processing](network.md)
Build and process river network topology with Strahler ordering and routing parameters.

- [`process_river_network()`](network.md#mobidic.preprocessing.river_network.process_river_network) - Complete network processing pipeline
- [`save_network()`](network.md#mobidic.preprocessing.io.save_network) - Save network to file
- [`load_network()`](network.md#mobidic.preprocessing.io.load_network) - Load processed network

### [Hillslope-Reach Mapping](hillslope_mapping.md)
Connect hillslope grid cells to river reaches for lateral flow routing.

- [`compute_hillslope_cells()`](hillslope_mapping.md#mobidic.preprocessing.hillslope_reach_mapping.compute_hillslope_cells) - Rasterize reaches onto grid
- [`map_hillslope_to_reach()`](hillslope_mapping.md#mobidic.preprocessing.hillslope_reach_mapping.map_hillslope_to_reach) - Map cells to reaches

### [Meteorological Preprocessing](meteo.md)
Convert meteorological data from various formats to CF-compliant NetCDF.

- [`MeteoData`](meteo.md#mobidic.preprocessing.meteo_preprocessing.MeteoData) - Container for meteorological station data
- [`convert_mat_to_netcdf()`](meteo.md#mobidic.preprocessing.meteo_preprocessing.convert_mat_to_netcdf) - Convert MATLAB to NetCDF

### [Data I/O](io.md)
Consolidated I/O for preprocessed MOBIDIC data.

- [`GISData`](io.md#mobidic.preprocessing.io.GISData) - Container for all preprocessed data
- [`save_gisdata()`](io.md#mobidic.preprocessing.io.save_gisdata) - Save preprocessed data
- [`load_gisdata()`](io.md#mobidic.preprocessing.io.load_gisdata) - Load preprocessed data

### [Preprocessing Workflow](preprocessing.md)
High-level workflow orchestrating the complete preprocessing pipeline.

- [`run_preprocessing()`](preprocessing.md#mobidic.preprocessing.preprocessor.run_preprocessing) - Complete preprocessing pipeline

## Quick Import

All public APIs are available from the top-level `mobidic` package:

```python
from mobidic import (
    # Configuration
    load_config,
    MOBIDICConfig,
    configure_logger,
    # GIS I/O
    grid_to_matrix,
    read_shapefile,
    # Grid Operations
    degrade_raster,
    degrade_flow_direction,
    convert_to_mobidic_notation,
    # River Network
    process_river_network,
    save_network,
    load_network,
    # Hillslope-Reach Mapping
    compute_hillslope_cells,
    map_hillslope_to_reach,
    # Meteorological Data
    MeteoData,
    convert_mat_to_netcdf,
    # Data I/O
    GISData,
    save_gisdata,
    load_gisdata,
    # Preprocessing Workflow
    run_preprocessing,
)
```

## Development Status

MOBIDICpy is in **pre-alpha** (v0.0.1). Currently implemented:

- ✅ Configuration system
- ✅ GIS data I/O
- ✅ Grid operations
- ✅ River network processing
- ✅ Hillslope-reach mapping
- ✅ Meteorological preprocessing
- ✅ Data I/O and consolidation
- ✅ Complete preprocessing workflow

Coming soon:

- ⏳ Soil water balance
- ⏳ Linear routing
- ⏳ Energy balance module
- ⏳ Groundwater models
- ⏳ Simulation engine
