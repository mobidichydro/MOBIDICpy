# Introduction

## Model overview

MOBIDIC (MOdello di Bilancio Idrologico DIstribuito e Continuo) is a physically-based distributed hydrological model that simulates the water and energy balance of the hydrological cycle at the cell level and run-off propagation in the river network.

**MOBIDICpy** represents a complete Python reimplementation of MOBIDIC, originally developed in MATLAB. This translation aims to improve usability, maintainability, and computational performance while preserving the scientific methods of the original model.

## Model structure

MOBIDIC operates on a distributed grid-based approach where each cell represents a computational element. The model consists of several interconnected modules:

### Soil-water balance

The soil module uses a **dual-pore storage model** that discretizes soil vertically with a single layer conceptually subdivided into two nonlinear reservoirs (Castillo et al., 2015):

- **Capillary reservoir ($W_c$)**: Smaller pores that hold water through capillary forces (field capacity to wilting point). Water leaves through evapotranspiration and absorption processes.
- **Gravitational reservoir ($W_g$)**: Larger pores that drain under gravity (maximum water content above field capacity). Water is supplied by infiltration and lateral flow from upslope, and leaves through percolation, downslope lateral flow, and absorption into the capillary reservoir.
- **Plant canopy reservoir ($W_p$)**: Rainfall interception by vegetation canopy, with throughfall occurring when capacity is exceeded.
- **Runoff reservoir ($W_s$)**: Surface depression storage with kinematic routing to downslope cells.

The hydrological processes in the soil-water balance include:

- **Infiltration**: MOBIDIC uses a **stochastic approach** to represent the intermittent nature of precipitation and spatial heterogeneity of soil properties (Castelli et al., 1996). The expected infiltration rate is computed assuming rainfall follows an exponential distribution, accounting for augmented infiltration before surface ponding. Both **Horton** (infiltration excess) and **Dunne** (saturation excess) runoff mechanisms are represented.

- **Percolation and hypodermic flow**: Modeled as linearly dependent on gravitational water content through empirical rate coefficients ($\beta$ for downhill hypodermic flow, $\gamma$ for percolation toward groundwater).

- **Absorption**: Transfer from gravitational to capillary reservoir dependent on capillary soil saturation deficit.

- **Runoff**: Outgoing runoff toward the downhill cell is evaluated as linearly dependent from surface water content through a kinematic parameter ($\alpha$).

### Energy balance

MOBIDIC solves water and energy balance simultaneously in the soil-vegetation subsystem. The energy module computes:

- Surface net radiation ($R_n$)
- Ground heat flux ($G$)
- Sensible heat flux ($H$)
- Latent heat flux ($LE$) for evapotranspiration

**Turbulent fluxes** are computed using bulk formulations that include both surface roughness and atmospheric stability effects (Van den Hurk and Holtslag, 1997). The energy balance is coupled with **1-D heat diffusion** into the soil.

**Evapotranspiration**: Potential ET is computed first from the energy balance, then actual ET is calculated considering water availability in canopy, surface, and capillary soil reservoirs. Soil moisture stress on ET follows an S-curve relationship based on capillary saturation.

Energy balance configurations:

- **None**: Energy balance disabled (a constant PET rate of 1 mm/day is used)
- **1-Layer (1L)**: Single surface layer for land surface temperature, solved analytically with a Fourier decomposition of the diurnal cycle and split at sunrise/sunset boundaries within each timestep. Two passes per timestep (saturated pre-pass to obtain PET, then re-entry with the actual ET/PET ratio to refine $T_s$ and $T_d$).
- **5-Layer (5L)**: Multi-layer soil temperature profile with depth (not yet implemented)
- **Snow module**: Includes snow accumulation and melt processes (not yet implemented)

See the [Energy Balance API reference](reference/energy_balance.md) for implementation details.

### Routing

Water routing occurs at two levels:

1. **Hillslope routing**: Lateral flow accumulates from upslope cells following D8 flow directions. Surface runoff ($W_s$) and subsurface lateral flow (from $W_g$) are routed separately through the hillslope network until reaching river channels.

2. **Channel routing**: The river network is represented as a **vector map** (polylines) with channels treated as cylindrical. Rivers are fed by surface runoff and baseflow from groundwater. Available routing methods include:
    - **Linear reservoir cascade**: Each reach is modeled as a simple reservoir with exponential recession (storage coefficient $K$). This method represents an optimal compromise between complexity and physical representativeness.
    - **Lag method (not yet implemented)**: Simple translation with no attenuation
    - **Muskingum method (not yet implemented)**: Hydraulic routing with wedge storage
    - **Muskingum-Cunge method (not yet implemented)**: Physically-based hydraulic routing

### Reservoir routing

When reservoirs are present in the river network, a dedicated **reservoir routing** module simulates their storage dynamics and regulation effects. Large reservoirs can be explicitly represented as regulation structures that store and release water according to operation rules. The reservoir module includes:

- **Volume-stage-discharge relationships**: Stage-storage curves define the relationship  between water level and storage volume (cubic spline interpolation). Stage-discharge curves  define regulated outflow as a function of water level (linear interpolation).
- **Time-varying regulation**: Multiple regulation periods allow for seasonal operation rules  (e.g., winter flood control vs summer water supply). The model automatically switches between  regulation curves based on a user-defined schedule.
- **Basin interactions**: Reservoir polygons are rasterized to identify contributing  hillslope cells. Surface runoff and lateral flow from reservoir basins are collected as  direct inflow. The model automatically identifies inlet reaches (upstream) and outlet reach  (downstream) from network topology.
- **Adaptive sub-stepping**: For numerical stability, the model can automatically subdivide  the simulation timestep when discharge variability is high.
- **Network integration**: Reservoir outflow is added to the lateral inflow of the outlet  reach. Inlet reach discharge is zeroed to prevent double-counting of water that enters the  reservoir.


### Groundwater

Several groundwater models are available:

- **None**: No groundwater interaction
- **Linear**: Linear-reservoir groundwater. Each cell is treated as an independent reservoir with linear reservoir equation, where the net recharge is calulated from the soil balance. The resulting baseflow is added to surface runoff before lateral inflow accumulation. With a single aquifer class (or no `Mf` mask in the configuration file), one reservoir per cell is used; with multiple positive classes in the raster file `Mf`, the head is averaged within each class (multiple linear reservoirs, with independent heads).
- **Dupuit**: 2-D physics-based aquifer model with explicit surface-subsurface interaction. Not yet implemented.
- **MODFLOW**: Coupling with USGS MODFLOW. Not yet implemented.

See the [Groundwater API reference](reference/groundwater.md) for implementation details.



## Calibration and sensitivity analysis

MOBIDICpy includes an integrated interface to [PEST++](https://github.com/usgs/pestpp) by USGS for automated parameter estimation, global sensitivity analysis, and uncertainty quantification without requiring manual PEST++ file management.

The integration is built on [pyEMU](https://github.com/pypest/pyemu) and (currently) supports the following [PEST++](https://github.com/usgs/pestpp) tools:

- **pestpp-glm**: Gradient-based calibration using the Gauss-Levenberg-Marquardt algorithm
- **pestpp-ies**: Ensemble-based calibration and uncertainty quantification using an Iterative Ensemble Smoother
- **pestpp-sen**: Global sensitivity analysis using the Morris one-at-a-time (OAT) method or Sobol method.

The calibration workflow is driven by a dedicated YAML configuration file that specifies the parameters to estimate, the observation time series, the objective function, the simulation and calibration periods, and options for parallel model execution. The `PestSetup` class generates all necessary PEST++ files, runs the forward model in parallel, and parses the results.

Metric-based pseudo-observations (NSE, KGE, PBIAS, RMSE, peak error) can be included directly in the objective function alongside raw discharge residuals, allowing multi-criteria calibration in a single PEST++ run.

See the [Calibration API reference](reference/calibration.md) and [Examples](examples.md) for usage.


## Applications

MOBIDIC has been successfully applied to:

- **Operational flood forecasting**: Real-time predictions with data assimilation (Arno basin, Tuscany)
- **Water resource management**: Long-term simulations for reservoir siting and operation (Masi et al., 2024)
- **Reservoir regulation**: Simulating reservoir effects on downstream flows with seasonal operation rules
- **Land use change evaluation**: Urbanization effects on hydrologic response (Yang et al., 2014)
- **Data assimilation**: Variational data assimilation for improved flood forecasting (Ercolani and Castelli, 2017)
- **Model intercomparison**: Benchmarking with other hydrological models (Castillo et al., 2015)
- **Research applications**: Green infrastructure performance (Ercolani et al., 2018), coupled water quality (Masi et al., 2025)


## Time convention

All datetimes in MOBIDICpy are **timezone-naive**. No timezone information is attached to any timestamp — input data, simulation times, state files, and output reports all use naive datetimes. Users must ensure that all input data (meteorological forcing, reservoir schedules, simulation start/end dates) refers to the same time reference (e.g., UTC or local time).

## Translation from MATLAB

MOBIDICpy is an ongoing translation effort focusing on:

1. **Modernization**: Python 3.11+ with modern syntax
2. **Simplification**: Functional approach prioritizing clarity over abstraction
3. **Validation**: Regression testing against MATLAB reference outputs
4. **Performance**: Leveraging NumPy, Numba, pandas, and modern geospatial libraries
5. **Standards**: CF-compliant NetCDF, GeoParquet, Pydantic validation

The original MATLAB implementation is being systematically translated with numerical consistency respect to the original model behavior.

## Current Status

**Version 0.2** - Basic hydrological processes (infiltration, percolation, runoff generation, hillslope and linear channel routing, linear groundwater (with multi-aquifer capability), reservoir routing, 1-layer energy balance) are implemented. Advanced energy balance schemes (5L, Snow), advanced groundwater models, and advanced routing methods are under development.

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

Masi, M., Arrighi, C., Piragino, F., Castelli, F. (2024). Participatory multi-criteria decision making for optimal siting of multipurpose artificial reservoirs. Journal of Environmental Management, 370, 122904. DOI: [10.1016/j.jenvman.2024.122904](https://doi.org/10.1016/j.jenvman.2024.122904)

Masi, M., Masseroni, D., Castelli, F. (2025). Coupled hydrologic, hydraulic, and surface water quality models for pollution management in urban–rural areas. 
Journal of Hydrology, 657, 133172. DOI: [10.1016/j.jhydrol.2025.133172](https://doi.org/10.1016/j.jhydrol.2025.133172).

Salvucci, G. D. (1993), An approximate solution for steady vertical flux of moisture through an unsaturated homogeneous soil, Water
Resour. Res., 29(11), 3749–3753, DOI: [10.1029/93WR02068](https://doi.org/10.1029/93WR02068).

Van den Hurk, B., and A. Holtslag (1997), On the bulk parameterization of surface fluxes for various conditions and parameter ranges, Boundary Layer Meteorol., 82(1), 119–133, DOI: [10.1023/A:1000245600901](https://doi.org/10.1023/A:1000245600901).

Yang, J., Castelli, F., Chen, Y. (2014). Multiobjective sensitivity analysis and optimization of
distributed hydrologic model MOBIDIC. Hydrology and Earth System Sciences, 18(10), 4101–4112.
DOI: [10.5194/HESS-18-4101-2014](https://doi.org/10.5194/HESS-18-4101-2014)

Yang, J., Entekhabi, D., Castelli, F., Chua, L. (2014). Hydrologic response of a tropical watershed to urbanization. 
Journal of Hydrology, 517, 538-546. DOI: [10.1016/j.jhydrol.2014.05.053](https://doi.org/10.1016/j.jhydrol.2014.05.053).
