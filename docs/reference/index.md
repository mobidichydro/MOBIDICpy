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

- [`read_raster()`](gis_io.md#mobidic.preprocessing.gis_reader.read_raster) - Read raster files (GeoTIFF)
- [`read_shapefile()`](gis_io.md#mobidic.preprocessing.gis_reader.read_shapefile) - Read vector files (Shapefile)

### [Grid Operations](grid_ops.md)
Process and transform gridded data (resolution degradation, flow direction conversion).

- [`degrade_raster()`](grid_ops.md#mobidic.preprocessing.grid_operations.degrade_raster) - Coarsen raster resolution
- [`degrade_flow_direction()`](grid_ops.md#mobidic.preprocessing.grid_operations.degrade_flow_direction) - Coarsen flow direction grids
- [`convert_flow_direction()`](grid_ops.md#mobidic.preprocessing.grid_operations.convert_flow_direction) - Convert between Grass and Arc notation

### [River Network Processing](network.md)
Build and process river network topology with Strahler ordering and routing parameters.

- [`process_river_network()`](network.md#mobidic.preprocessing.river_network.process_river_network) - Complete network processing pipeline
- [`export_network()`](network.md#mobidic.preprocessing.river_network.export_network) - Export network to file
- [`load_network()`](network.md#mobidic.preprocessing.river_network.load_network) - Load processed network

## Quick Import

All public APIs are available from the top-level `mobidic` package:

```python
from mobidic import (
    # Configuration
    load_config,
    MOBIDICConfig,
    # GIS I/O
    read_raster,
    read_shapefile,
    # Grid Operations
    degrade_raster,
    degrade_flow_direction,
    convert_flow_direction,
    # River Network
    process_river_network,
    export_network,
    load_network,
)
```

## Development Status

MOBIDICpy is in **pre-alpha** (v0.0.1). Currently implemented:

- ✅ Configuration system
- ✅ GIS data I/O
- ✅ Grid operations
- ✅ River network processing

Coming soon:

- ⏳ Soil water balance
- ⏳ Linear routing
- ⏳ Energy balance module
- ⏳ Groundwater models
