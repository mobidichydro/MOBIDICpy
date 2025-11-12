# Simulation

The simulation module implements the main time-stepping loop of the MOBIDIC hydrological model, orchestrating water balance calculations, routing, and I/O operations.

## Overview

The simulation engine coordinates:

- **Input data loading**: GIS preprocessing and meteorological forcing
- **State initialization**: Initial conditions for soil, surface, and channel states
- **Time-stepping loop**: Sequential water balance and routing calculations
- **Meteorological interpolation**: Station data → grid interpolation (IDW, nearest neighbor)
- **PET calculation**: Hargreaves-Samani method
- **Results storage**: Time series collection and state snapshots
- **Output generation**: NetCDF states and Parquet reports

**Current implementation**: Simplified version without energy balance, groundwater models, or reservoirs.

## Classes

::: mobidic.core.simulation.Simulation

::: mobidic.core.simulation.SimulationState

::: mobidic.core.simulation.SimulationResults

## Simulation loop

The main simulation loop performs the following operations for each time step:

1. **Interpolate forcing**: Precipitation from station data to grid
2. **Calculate PET**: Hargreaves-Samani method using temperature and solar radiation
3. **Soil water balance**: Four-reservoir hillslope water balance
4. **Hillslope routing**: Route surface runoff and lateral flow on grid
5. **Reach mapping**: Accumulate hillslope contributions to river reaches
6. **Channel routing**: Linear reservoir routing through river network
7. **Store results**: Save discharge and lateral inflow time series
8. **Output states**: Optionally save states

## Performance 

- **Meteorological interpolation caching**: Pre-computes time indices and spatial weights for all timesteps
- **Numba JIT compilation**: Hillslope and channel routing use compiled kernels
- **Memory efficiency**: State variables use NumPy arrays with F-contiguous memory layout
- **Progress logging**: Adaptive logging interval (max 20 logs or every 30s)

## Implementated modules

- [Soil Water Balance](soil_water_balance.md) - Hillslope water balance
- [Routing](routing.md) - Hillslope and channel routing
- [State I/O](state.md) - NetCDF state export/import
- [Report I/O](report.md) - Time series export/import
