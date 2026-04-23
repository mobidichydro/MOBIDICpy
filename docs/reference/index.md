# API reference

MOBIDICpy provides a comprehensive Python API for distributed hydrological modeling. The API is organized into functional modules for easy navigation and usage.

## Overview

The MOBIDICpy public API is designed to be simple and functional, exposing only the essential functions needed for preprocessing, configuration, and model setup.

## API modules

### [Configuration](config.md)
Load and validate YAML configuration files using Pydantic schemas, and configure logging behavior.

- [`load_config()`](config.md#mobidic.config.parser.load_config) - Load and validate configuration from YAML
- [`MOBIDICConfig`](config.md#mobidic.config.schema.MOBIDICConfig) - Complete configuration schema
- [`configure_logger()`](config.md#mobidic.utils.logging.configure_logger) - Configure logging programmatically
- [`configure_logger_from_config()`](config.md#mobidic.utils.logging.configure_logger_from_config) - Configure logging from YAML

### [GIS data I/O](gis_io.md)
Read and write geospatial data (rasters and vectors).

- [`grid_to_matrix()`](gis_io.md#mobidic.preprocessing.gis_reader.grid_to_matrix) - Read raster files (GeoTIFF)
- [`read_shapefile()`](gis_io.md#mobidic.preprocessing.gis_reader.read_shapefile) - Read vector files (Shapefile)

### [Grid operations](grid_ops.md)
Process and transform gridded data (resolution decimation, flow direction conversion).

- [`decimate_raster()`](grid_ops.md#mobidic.preprocessing.grid_operations.decimate_raster) - Coarsen raster resolution
- [`decimate_flow_direction()`](grid_ops.md#mobidic.preprocessing.grid_operations.decimate_flow_direction) - Coarsen flow direction grids

### [River network processing](network.md)
Build and process river network topology with Strahler ordering and routing parameters.

- [`process_river_network()`](network.md#mobidic.preprocessing.river_network.process_river_network) - Complete network processing pipeline
- [`save_network()`](network.md#mobidic.preprocessing.io.save_network) - Save network to file
- [`load_network()`](network.md#mobidic.preprocessing.io.load_network) - Load processed network

### [Hillslope-reach mapping](hillslope_mapping.md)
Connect hillslope grid cells to river reaches for lateral flow routing.

- [`compute_hillslope_cells()`](hillslope_mapping.md#mobidic.preprocessing.hillslope_reach_mapping.compute_hillslope_cells) - Rasterize reaches onto grid
- [`map_hillslope_to_reach()`](hillslope_mapping.md#mobidic.preprocessing.hillslope_reach_mapping.map_hillslope_to_reach) - Map cells to reaches

### [Meteorological preprocessing](meteo.md)
Convert meteorological data from various formats to CF-compliant NetCDF, and generate synthetic design storm hyetographs.

- [`MeteoData`](meteo.md#mobidic.preprocessing.meteo_preprocessing.MeteoData) - Container for meteorological station data
- [`MeteoRaster`](meteo.md#mobidic.preprocessing.meteo_raster.MeteoRaster) - Container for gridded (raster) meteorological forcing
- [`HyetographGenerator`](meteo.md#mobidic.preprocessing.hyetograph.HyetographGenerator) - Generate synthetic hyetographs from IDF parameters
- [`convert_mat_to_netcdf()`](meteo.md#mobidic.preprocessing.meteo_preprocessing.convert_mat_to_netcdf) - Convert MATLAB to NetCDF

### [Data I/O](io.md)
Consolidated I/O for preprocessed MOBIDIC data.

- [`GISData`](io.md#mobidic.preprocessing.preprocessor.GISData) - Container for all preprocessed data
- [`save_gisdata()`](io.md#mobidic.preprocessing.io.save_gisdata) - Save preprocessed data
- [`load_gisdata()`](io.md#mobidic.preprocessing.io.load_gisdata) - Load preprocessed data

### [Preprocessing workflow](preprocessing.md)
High-level workflow orchestrating the complete preprocessing pipeline.

- [`run_preprocessing()`](preprocessing.md#mobidic.preprocessing.preprocessor.run_preprocessing) - Complete preprocessing pipeline
- [`process_reservoirs()`](preprocessing.md#mobidic.preprocessing.reservoirs.process_reservoirs) - Process reservoir data (polygons, curves, schedules)

### [Soil water balance](soil_water_balance.md)
Hillslope water balance with four reservoir model (capillary, gravitational, plants, surface).

- [`soil_mass_balance()`](soil_water_balance.md#mobidic.core.soil_water_balance.soil_mass_balance) - Main hillslope water balance function

### [Routing](routing.md)
Hillslope and channel routing algorithms for water propagation.

- [`hillslope_routing()`](routing.md#mobidic.core.routing.hillslope_routing) - Accumulate lateral flow following flow directions
- [`linear_channel_routing()`](routing.md#mobidic.core.routing.linear_channel_routing) - Linear reservoir channel routing

### [Groundwater](groundwater.md)
Saturated-zone dynamics providing the baseflow contribution to surface runoff.

- [`groundwater_linear()`](groundwater.md#functions) - Linear-reservoir groundwater model (with optional multi-aquifer averaging via the `Mf` raster)

### [Energy Balance](energy_balance.md)
Surface energy budget computing potential evapotranspiration and tracking surface/deep-soil temperatures.

- [`compute_energy_balance_1l()`](energy_balance.md#functions) - Orchestrator for the 1-layer scheme over a model timestep (sunrise/sunset sub-stepping)
- [`energy_balance_1l()`](energy_balance.md#functions) - Analytical Fourier 1-layer solver
- [`solar_position()`](energy_balance.md#functions), [`solar_hours()`](energy_balance.md#functions) - Solar geometry helpers
- [`diurnal_radiation_cycle()`](energy_balance.md#functions) - Sinusoidal decomposition of daily radiation
- [`saturation_specific_humidity()`](energy_balance.md#functions) - Magnus formula

### [Simulation](simulation.md)
Main simulation engine and time-stepping loop.

- [`Simulation`](simulation.md#mobidic.core.simulation.Simulation) - Main simulation class
- [`SimulationState`](simulation.md#mobidic.core.simulation.SimulationState) - Container for state variables
- [`SimulationResults`](simulation.md#mobidic.core.simulation.SimulationResults) - Container for simulation results

### [State I/O](state.md)
Save and load simulation state variables (NetCDF format).

- [`StateWriter`](state.md#mobidic.io.state.StateWriter) - Incremental state writer with buffering
- [`load_state()`](state.md#mobidic.io.state.load_state) - Load state from NetCDF
- [`MeteoWriter`](meteo.md#mobidic.io.meteo.MeteoWriter) - Writer for interpolated meteorological data grids

### [Report I/O](report.md)
Export discharge and lateral inflow time series.

- [`save_discharge_report()`](report.md#mobidic.io.report.save_discharge_report) - Export discharge time series
- [`save_lateral_inflow_report()`](report.md#mobidic.io.report.save_lateral_inflow_report) - Export lateral inflow time series
- [`load_discharge_report()`](report.md#mobidic.io.report.load_discharge_report) - Load discharge time series

### [Calibration](calibration.md)
Model calibration, global sensitivity analysis, and uncertainty quantification using PEST++.

Requires optional dependencies: `pip install "mobidicpy[calibration]" && get-pestpp :pyemu`

- [`PestSetup`](calibration.md#mobidic.calibration.pest_setup.PestSetup) - Orchestrates the complete PEST++ workflow (setup, run, results)
- [`CalibrationResults`](calibration.md#mobidic.calibration.results.CalibrationResults) - Parses PEST++ output files (parameters, objective function, sensitivities)
- [`CalibrationConfig`](calibration.md#mobidic.calibration.config.CalibrationConfig) - Pydantic model for calibration configuration
- [`load_calibration_config()`](calibration.md#mobidic.calibration.config.load_calibration_config) - Load calibration configuration from YAML
- [`nse()`](calibration.md#mobidic.calibration.metrics.nse), [`kge()`](calibration.md#mobidic.calibration.metrics.kge), [`pbias()`](calibration.md#mobidic.calibration.metrics.pbias), [`rmse()`](calibration.md#mobidic.calibration.metrics.rmse) - Hydrological performance metrics
- [`load_observations()`](calibration.md#mobidic.calibration.observation.load_observations) - Load observation time series from CSV

## Quick import

All core APIs are available from the top-level `mobidic` package. The calibration module is under `mobidic.calibration`:

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
    # River Network
    process_river_network,
    save_network,
    load_network,
    # Hillslope-Reach Mapping
    compute_hillslope_cells,
    map_hillslope_to_reach,
    # Meteorological Data
    MeteoData,
    MeteoRaster,
    HyetographGenerator,
    convert_mat_to_netcdf,
    # Data I/O
    GISData,
    save_gisdata,
    load_gisdata,
    # Preprocessing Workflow
    run_preprocessing,
    # Soil Water Balance
    soil_mass_balance,
    # Routing
    hillslope_routing,
    linear_channel_routing,
    # Groundwater
    groundwater_linear,
    # Energy Balance
    compute_energy_balance_1l,
    energy_balance_1l,
    diurnal_radiation_cycle,
    saturation_specific_humidity,
    solar_hours,
    solar_position,
    # Reservoirs
    process_reservoirs,
    Reservoir,
    Reservoirs,
    load_reservoirs,
    save_reservoirs,
    # Simulation
    Simulation,
    SimulationState,
    SimulationResults,
    # State I/O
    StateWriter,
    load_state,
    MeteoWriter,
    # Report I/O
    save_discharge_report,
    save_lateral_inflow_report,
    load_discharge_report,
    # Constants
    constants,
)
```

## Development status

MOBIDICpy's currently implemented features (v0.2):

- Configuration system
- GIS data I/O
- Grid operations
- River network processing
- Hillslope-reach mapping
- Meteorological preprocessing (station data, raster forcing, hyetograph generation)
- Data I/O and consolidation
- Complete preprocessing workflow
- Soil water balance (4 reservoirs: capillary, gravitational, plants, surface)
- Linear routing (hillslope + channel)
- Linear reservoir groundwater model (with multi-aquifer capability via the `Mf` raster)
- Reservoir module (preprocessing, routing, time-varying regulation)
- Energy balance, 1-layer (1L) scheme
- Simulation engine (basic)
- State I/O (NetCDF, including surface and deep-soil temperatures)
- Report I/O (CSV/Parquet discharge time series)
- Calibration and sensitivity analysis (PEST++ coupling via pyemu)

Not yet implemented features:

- Energy balance: 5-layer scheme and Snow module
- Advanced groundwater models (Dupuit, MODFLOW)
- Advanced routing (Muskingum-Cunge)
- Meteorological data gap filling and quality control
- CLI interface
