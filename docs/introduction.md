# Introduction

## Model Overview

MOBIDIC (MOdello di Bilancio Idrologico DIstribuito e Continuo) is a raster-based distributed hydrological model that simulates the water and energy balance of the hydrological cycle at the cell level. The model was originally developed at the University of Florence by Prof. Fabio Castelli and colleagues.

**MOBIDICpy** represents a complete Python reimplementation of MOBIDIC, originally developed in MATLAB. This translation aims to improve accessibility, maintainability, and computational performance while preserving the scientific rigor of the original model.

## Model Structure

MOBIDIC operates on a distributed grid-based approach where each cell represents a computational unit. The model consists of several interconnected modules:

### Soil Water Balance

The soil module simulates vertical water movement through multiple reservoirs:

- **Capillary reservoir (Wc)**: Water retained by capillary forces
- **Gravitational reservoir (Wg)**: Water draining under gravity
- **Surface runoff**: Excess rainfall that cannot infiltrate
- **Subsurface flow**: Lateral water movement through soil layers

Two soil schemes are available:

- **Bucket model**: Simplified representation with threshold-based routing
- **Curve Number (CN)**: NRCS-based runoff generation

### Energy Balance

The energy module computes surface and ground temperatures, evapotranspiration, and snowmelt (when enabled). Three configurations are available:

- **1-Layer (1L)**: Single surface layer
- **5-Layer (5L)**: Multi-layer soil temperature profile
- **Snow module**: Includes snow accumulation and melt processes

### Routing

Water routing occurs at two levels:

1. **Hillslope routing**: Lateral flow from hillslope cells to river reaches using linear reservoir or lag methods
2. **Channel routing**: In-stream propagation using:
   - Linear reservoir
   - Lag method
   - Muskingum method
   - Muskingum-Cunge method (physically-based)

### Groundwater

Several groundwater models are available:

- **None**: No groundwater interaction
- **Linear**: Simple linear reservoir
- **Linear_mult**: Multiple parallel linear reservoirs
- **Dupuit**: Physics-based aquifer model
- **MODFLOW**: Coupling with USGS MODFLOW

## Key Features

- **Distributed approach**: Spatial heterogeneity in parameters and forcing
- **Continuous simulation**: Long-term water balance accounting
- **Multi-scale**: From hillslope (cell-level) to basin outlet
- **Flexible**: Modular structure allows selective activation of components
- **Physics-based**: Process representations grounded in hydrological theory

## Applications

MOBIDIC has been successfully applied to:

- Flood forecasting and early warning systems
- Water resource management and planning
- Climate change impact assessment
- Land use change evaluation
- Model intercomparison studies

## Translation from MATLAB

MOBIDICpy is an ongoing translation effort focusing on:

1. **Modernization**: Python 3.10+ with type hints and modern syntax
2. **Simplification**: Functional approach prioritizing clarity over abstraction
3. **Validation**: Regression testing against MATLAB reference outputs
4. **Performance**: Leveraging NumPy, pandas, and modern geospatial libraries
5. **Standards**: CF-compliant NetCDF, GeoParquet, Pydantic validation

The original MATLAB implementation (87 files, ~15,000 LOC) is being systematically translated with careful attention to scientific accuracy and numerical consistency.

## Current Status

**Version 0.0.1 (Pre-Alpha)** - Configuration and preprocessing are functional. Core simulation engine is under development.

See the [home page](index.md) for detailed implementation status.

## References

Castelli, F., Menduni, G., & Mazzanti, B. (2009). A distributed package for sustainable water management: A case study in the Arno basin. *Role of Hydrology in Water Resources Management*, 327, 52–61.

Yang, J., Castelli, F., & Chen, Y. (2014). Multiobjective sensitivity analysis and optimization of distributed hydrologic model MOBIDIC. *Hydrology and Earth System Sciences*, 18(10), 4101–4112. [https://doi.org/10.5194/HESS-18-4101-2014](https://doi.org/10.5194/HESS-18-4101-2014)
