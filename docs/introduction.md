# Introduction

## Model Overview

MOBIDIC (MOdello di Bilancio Idrologico DIstribuito e Continuo) is a raster-based distributed hydrological model that simulates the water and energy balance of the hydrological cycle at the cell level. The model was originally developed at the University of Florence by Prof. Fabio Castelli and colleagues.

**MOBIDICpy** represents a complete Python reimplementation of MOBIDIC, originally developed in MATLAB. This translation aims to improve usability, maintainability, and computational performance while preserving the scientific methods of the original model.

## Model Structure

MOBIDIC operates on a distributed grid-based approach where each cell represents a computational unit. The model consists of several interconnected modules:

### Soil Water Balance

The soil module simulates vertical water movement through multiple reservoirs:

- **Capillary reservoir (Wc)**: Water retained by capillary forces
- **Gravitational reservoir (Wg)**: Water draining under gravity
- **Plant canopy reservoir (Wp)**: Interception storage
- **Runoff reservoir (Ws)**: Surface runoff storage

One soil scheme is currently available:

- **Bucket model**: Simplified representation with threshold-based routing

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

The original MATLAB implementation is being systematically translated with careful attention to scientific accuracy and numerical consistency.

## Current Status

**Version 0.0.1 (Pre-Alpha)** - Configuration and preprocessing are functional. Core simulation engine is under development.

See the [home page](index.md) for implementation status.

## References

Campo, L., Caparrini, F., Castelli, F. (2006). Use of multi-platform, multi-temporal remote-sensing data for calibration of a 
distributed hydrological model: an application in the Arno basin, Italy. Hydrol. Process., 20: 2693-2712. 
DOI: [10.1002/hyp.6061](https://doi.org/10.1002/hyp.6061)

Castelli, F. (1996). A simplified stochastic model for infiltration into a heterogeneous soil forced by random precipitation. Advances in water resources, 19(3), 133-144. DOI: [10.1016/0309-1708(95)00041-0](https://doi.org/10.1016/0309-1708(95)00041-0)

Castelli, F., Menduni, G., and Mazzanti, B. (2009). A distributed package for sustainable water
management: A case study in the Arno basin. Role of Hydrology in Water Resources Management,
327, 52–61.

Castillo, A., Castelli, F., Entekhabi, D. (2015). Gravitational and capillary soil moisture dynamics for distributed
hydrologic models, Hydrol. Earth Syst. Sci., 19, 1857–1869, DOI: [10.5194/hess-19-1857-2015](https://doi.org/10.5194/hess-19-1857-2015).

Castelli, F., Ercolani, G. (2016). Improvement of operational flood forecasting through the assimilation of satellite observations and 
multiple river flow data, Proc. IAHS, 373, 167–173. DOI: [10.5194/piahs-373-167-2016](https://doi.org/10.5194/piahs-373-167-2016).

Ercolani, G., Castelli, F. (2017), Variational assimilation of streamflow data in distributed flood forecasting, Water Resour. Res., 53, 158–183. 
DOI: [10.1002/2016WR019208](https://doi.org/10.1002/2016WR019208).

Ercolani, G., Chiaradia, E. A., Gandolfi, C., Castelli, F., Masseroni, D. (2018). Evaluating performances of green roofs for stormwater runoff mitigation in a high flood risk urban catchment. Journal of Hydrology, 566, 830-845. DOI: [10.1016/j.jhydrol.2018.09.050](https://doi.org/10.1016/j.jhydrol.2018.09.050)

Masi, M., Masseroni, D., Castelli, F. (2025). Coupled hydrologic, hydraulic, and surface water quality models for pollution management in urban–rural areas. 
Journal of Hydrology, 657, 133172. DOI: [10.1016/j.jhydrol.2025.133172](https://doi.org/10.1016/j.jhydrol.2025.133172).

Yang, J., Castelli, F., Chen, Y. (2014). Multiobjective sensitivity analysis and optimization of
distributed hydrologic model MOBIDIC. Hydrology and Earth System Sciences, 18(10), 4101–4112.
DOI: [10.5194/HESS-18-4101-2014](https://doi.org/10.5194/HESS-18-4101-2014)

Yang, J., Entekhabi, D., Castelli, F., Chua, L. (2014). Hydrologic response of a tropical watershed to urbanization. 
Journal of Hydrology, 517, 538-546. DOI: [10.1016/j.jhydrol.2014.05.053](https://doi.org/10.1016/j.jhydrol.2014.05.053).
