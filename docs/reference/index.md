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
Process and transform gridded data (resolution decimation, flow direction conversion).

- [`decimate_raster()`](grid_ops.md#mobidic.preprocessing.grid_operations.decimate_raster) - Coarsen raster resolution
- [`decimate_flow_direction()`](grid_ops.md#mobidic.preprocessing.grid_operations.decimate_flow_direction) - Coarsen flow direction grids
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

### [Soil Water Balance](soil_water_balance.md)
Hillslope water balance with four reservoir model (capillary, gravitational, plants, surface).

- [`soil_mass_balance()`](soil_water_balance.md#mobidic.core.soil_water_balance.soil_mass_balance) - Main hillslope water balance function
- [`capillary_rise()`](soil_water_balance.md#mobidic.core.soil_water_balance.capillary_rise) - Calculate capillary rise from groundwater

### [Routing](routing.md)
Hillslope and channel routing algorithms for water propagation.

- [`hillslope_routing()`](routing.md#mobidic.core.routing.hillslope_routing) - Accumulate lateral flow following flow directions
- [`linear_channel_routing()`](routing.md#mobidic.core.routing.linear_channel_routing) - Linear reservoir channel routing

### [Simulation](simulation.md)
Main simulation engine and time-stepping loop.

- [`Simulation`](simulation.md#mobidic.core.simulation.Simulation) - Main simulation class
- [`SimulationState`](simulation.md#mobidic.core.simulation.SimulationState) - Container for state variables
- [`SimulationResults`](simulation.md#mobidic.core.simulation.SimulationResults) - Container for simulation results

### [State I/O](state.md)
Save and load simulation state variables (NetCDF format).

- [`StateWriter`](state.md#mobidic.io.state.StateWriter) - Incremental state writer with buffering
- [`load_state()`](state.md#mobidic.io.state.load_state) - Load state from NetCDF

### [Report I/O](report.md)
Export discharge and lateral inflow time series.

- [`save_discharge_report()`](report.md#mobidic.io.report.save_discharge_report) - Export discharge time series
- [`save_lateral_inflow_report()`](report.md#mobidic.io.report.save_lateral_inflow_report) - Export lateral inflow time series
- [`load_discharge_report()`](report.md#mobidic.io.report.load_discharge_report) - Load discharge time series

## Quick Import

All public APIs are available from the top-level `mobidic` package:

```python
from mobidic import (
    # Configuration
    load_config,
    MOBIDICConfig,
    configure_logger,
    configure_logger_from_config,
    # GIS I/O
    grid_to_matrix,
    read_shapefile,
    # Grid Operations
    decimate_raster,
    decimate_flow_direction,
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
    # Soil Water Balance
    soil_mass_balance,
    capillary_rise,
    # Routing
    hillslope_routing,
    linear_channel_routing,
    # Simulation
    Simulation,
    SimulationState,
    SimulationResults,
    # State I/O
    StateWriter,
    load_state,
    # Report I/O
    save_discharge_report,
    save_lateral_inflow_report,
    load_discharge_report,
    # Constants
    constants,
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
- ✅ Soil water balance (4 reservoirs: capillary, gravitational, plants, surface)
- ✅ Linear routing (hillslope + channel)
- ✅ Simulation engine (basic)
- ✅ State I/O (NetCDF)
- ✅ Report I/O (CSV/Parquet discharge time series)

Coming soon:

- ⏳ Energy balance module (1L, 5L schemes, Snow)
- ⏳ Groundwater models (Linear, Linear_mult, Dupuit, MODFLOW)
- ⏳ Advanced routing (Muskingum-Cunge)
- ⏳ Reservoir module
- ⏳ Real-time capability
