# Simulation

The simulation module implements the main time-stepping loop of the MOBIDIC hydrological model, orchestrating water balance calculations, routing, and I/O operations.

## Overview

The simulation engine coordinates:

- **Input data loading**: GIS preprocessing and meteorological forcing (station-based or raster-based)
- **Meteorological forcing**: Automatically detects station data (with spatial interpolation) or pre-interpolated raster data
- **State initialization**: Initial conditions for soil, surface, and channel states (supports warm start)
- **Time-stepping loop**: Sequential water balance and routing calculations
- **Station-based interpolation**: Grid interpolation using IDW or nearest neighbor with pre-computed weights
- **Raster-based forcing**: Direct sampling from pre-interpolated grids with grid alignment validation
- **Interpolated meteo output**: Optional export of interpolated grids for subsequent raster-based runs
- **PET calculation**: 1-layer analytical energy balance (when `simulation.energy_balance == "1L"`) or a constant 1 mm/day fallback. See the [Energy Balance reference](energy_balance.md) for details.
- **Precomputed-PET fast path**: Automatically skips the energy balance when the input `MeteoRaster` already contains a `pet` variable
- **Results storage**: Time series collection and state snapshots with automatic file chunking
- **Output generation**: NetCDF states (with chunking) and Parquet/CSV reports
- **Restart capability**: Load and resume from previously saved states

**Current implementation**: Includes soil water balance, routing (hillslope, channel, reservoir), linear-reservoir groundwater (with multi-aquifer averaging), the 1-layer analytical energy balance, and state/report I/O. The 5L and Snow energy schemes, and advanced groundwater models (Dupuit, MODFLOW), are not yet implemented.

## Classes

::: mobidic.core.simulation.Simulation

::: mobidic.core.simulation.SimulationState

::: mobidic.core.simulation.SimulationResults

## Simulation loop

The main simulation loop performs the following operations for each time step:

1. **Get forcing**: Precipitation from station data (interpolated to grid using IDW/nearest) or from raster data (direct sampling). When the energy balance is active, the temperature/humidity/wind/radiation grids are also fetched.
2. **Calculate PET**: 1-layer analytical [energy balance](energy_balance.md) pre-pass (with $\eta = 1$) when `simulation.energy_balance == "1L"`; read directly from the raster when the forcing already contains `pet`; otherwise a constant 1 mm/day fallback.
3. **Save interpolated meteo** (optional): Export interpolated grids (and PET when the energy balance is active) to `meteo_forcing.nc`
4. **Route previous flows**: Hillslope routing of surface runoff and lateral flow from previous timestep
5. **Soil water balance**: Four-reservoir hillslope water balance with routed inflows
6. **Groundwater dynamics** (if `parameters.groundwater.model == "Linear"`): Update groundwater head from net recharge (percolation − global loss) and add the resulting baseflow to the surface runoff rate; optionally average head within each class of the `Mf` raster (multi-aquifer mode)
7. **Reservoir routing** (if configured): Update reservoir volumes, calculate regulated discharge, zero basin fluxes
8. **Accumulate to reaches**: Accumulate surface runoff contributions to river reaches
9. **Channel routing**: Linear reservoir routing through river network
10. **Energy balance re-entry** (if `simulation.energy_balance == "1L"`): re-solve the surface energy budget using the actual $\eta = \text{ET} / \text{PET}$ from the soil module and refine $T_s$ and $T_d$ on water-limited cells
11. **Store results**: Save discharge and lateral inflow time series
12. **Output states**: Optionally save states (with automatic chunking if needed), including $T_s$ and $T_d$ when the corresponding `output_states` flags are enabled
13. **Update and advance**: Store flow fields for next timestep and advance simulation time

**Key feature**: The simulation uses a feedback loop where flows from timestep `t` are routed through the hillslope at timestep `t+1` before entering the soil water balance. This ensures proper spatial connectivity of overland flow.

## Performance

- **Meteorological interpolation caching**: Pre-computes time indices and spatial weights for all timesteps
- **Numba JIT compilation**: Hillslope and channel routing use compiled kernels
- **Memory efficiency**: State variables use NumPy arrays with F-contiguous memory layout
- **Contributing pixels optimization**: Processes only cells that can contribute flow to river network
- **Network topology caching**: Pre-extracts network structure to numpy arrays for fast routing
- **Progress logging**: Adaptive logging interval (max 20 logs or every 30s) with text-based progress bar
- **Automatic file chunking**: State files automatically split when reaching size limit

## Examples

### Basic simulation with station-based forcing

```python
from mobidic import load_config, load_gisdata, Simulation, MeteoData

# Load configuration and data
config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")
forcing = MeteoData.from_netcdf("meteo.nc")

# Create simulation
sim = Simulation(gisdata, forcing, config)

# Run simulation
results = sim.run("2020-01-01", "2020-12-31")

# Save discharge report
results.save_report("discharge.parquet")

# Save lateral inflow report
results.save_lateral_inflow_report("lateral_inflow.parquet")
```

### Simulation with raster-based forcing

```python
from mobidic import load_config, load_gisdata, Simulation, MeteoRaster

# Load configuration and data
config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")

# Load raster forcing (preload into memory for fast access)
forcing = MeteoRaster.from_netcdf("meteo_raster.nc")

# Create simulation (automatically detects raster mode)
sim = Simulation(gisdata, forcing, config)

# Run simulation (no interpolation needed, uses direct sampling)
results = sim.run("2020-01-01", "2020-12-31")

# Save results
results.save_report("discharge.parquet")
```

### Export and use interpolated meteorological data

```python
from mobidic import MeteoData, MeteoRaster, Simulation

# Run 1: Station-based with forcing output enabled
config.output_forcing_data.meteo_data = True
forcing_stations = MeteoData.from_netcdf("meteo_stations.nc")
sim1 = Simulation(gisdata, forcing_stations, config)
results1 = sim1.run("2020-01-01", "2020-12-31")
# Forcing data saved to: output/meteo_forcing.nc

# Run 2: Use exported raster forcing (faster, identical results)
config.output_forcing_data.meteo_data = False
forcing_raster = MeteoRaster.from_netcdf("output/meteo_forcing.nc")
sim2 = Simulation(gisdata, forcing_raster, config)
results2 = sim2.run("2020-01-01", "2020-12-31")
```

### Warm start (resume from saved simulation state)

The simulation supports warm start capability, allowing you to resume from previously saved states. This is useful for:

- **Multi-stage simulations**: Spin-up period followed by analysis period
- **Interrupted simulations**: Resume after crashes or timeouts
- **Ensemble runs**: Start multiple simulations from calibrated initial states
- **Seasonal forecasts**: Initialize from observed states

```python
from mobidic import Simulation, load_state

# Method 1: Load state from file and set before running
sim = Simulation(gisdata, forcing, config)
sim.set_initial_state(state_file="spinup_states.nc", time_index=-1)  # Use last timestep
results = sim.run("2020-06-01", "2020-12-31")

# Method 2: Load state object and use directly
from mobidic.io import load_state
state, time, metadata = load_state("spinup_states.nc", network_size=1235)
sim.set_initial_state(state=state)
results = sim.run("2020-06-01", "2020-12-31")

# Method 3: Multi-stage simulation
# Stage 1: Spin-up (1 year)
sim1 = Simulation(gisdata, forcing, config)
results1 = sim1.run("2019-01-01", "2019-12-31")  # Saves states.nc

# Stage 2: Analysis period (resume from spin-up)
sim2 = Simulation(gisdata, forcing, config)
sim2.set_initial_state(state_file="output/states.nc")  # Load last state from spin-up
results2 = sim2.run("2020-01-01", "2020-12-31")
```

### Working with large simulations and chunked states

For long simulations that generate a large number of states, the simulation automatically creates chunked files:

```python
from mobidic import Simulation

# Configure for large simulation
# Edit config.yaml:
# output_states_settings:
#   output_states: "all"
#   flushing: 100          # Flush every 100 timesteps (required for chunking)
#   max_file_size: 500.0   # Create new chunk at 500 MB
#   output_interval: 3600  # Save state every hour

config = load_config("config.yaml")
sim = Simulation(gisdata, forcing, config)

# Run long simulation (e.g., 10 years at 15-minute timesteps)
results = sim.run("2010-01-01", "2020-12-31")

# This creates multiple chunk files:
# - output/states_001.nc (500 MB)
# - output/states_002.nc (500 MB)
# - output/states_003.nc (350 MB)

# Resume from last chunk (automatically detected)
sim2 = Simulation(gisdata, forcing, config)
sim2.set_initial_state(state_file="output/states.nc")  # Auto-finds states_001.nc
results2 = sim2.run("2021-01-01", "2021-12-31")
```

### Custom report selection

Control which reaches to include in output reports:

```python
# Method 1: Save all reaches
results.save_report(
    "discharge_all.parquet",
    reach_selection="all"
)

# Method 2: Save reaches from file
results.save_report(
    "discharge_selected.parquet",
    reach_selection="file",
    reach_file="reach_ids.json"
)

# Method 3: Save specific reaches by mobidic_id
selected_reaches = [0, 10, 25, 100, 500]  # mobidic_id values
results.save_report(
    "discharge_selected.parquet",
    reach_selection="list",
    selected_reaches=selected_reaches
)

# Method 4: Export as CSV instead of Parquet
results.save_report(
    "discharge.csv",
    reach_selection="all",
    output_format="csv"
)
```

## Configuration

The simulation behavior is controlled by the configuration file. Key sections:

### Output states configuration

```yaml
output_states:
  Wc: true              # Save capillary water
  Wg: true              # Save gravitational water
  Wp: false             # Plant water (not yet implemented)
  Ws: true              # Save surface water
  discharge: true       # Save channel discharge
  lateral_inflow: true  # Save lateral inflow to reaches
  reservoir_states: true  # Save reservoir states (if reservoirs configured)

output_states_settings:
  output_format: "netCDF"
  output_states: "final"  # Options: "final", "all", "list", "None"
  flushing: 10            # Flush to disk every N timesteps (-1 = only at end)
  max_file_size: 500.0    # Maximum file size in MB (chunking threshold)
  output_interval: 3600   # Save interval in seconds (for "all" mode)
  output_list:            # List of specific datetimes (for "list" mode)
    - "2020-06-01 00:00:00"
    - "2020-12-31 23:45:00"
```

**Important notes about state output:**
- **Chunking requires `flushing > 0`**: Set `flushing` to a positive value (e.g., 10, 50, 100) for file chunking to work effectively
- **`flushing=-1` disables chunking**: All data is written at the end in one operation, preventing chunking
- **File size may slightly exceed limit**: Files can exceed `max_file_size` by up to one flush worth of data

### Output reports configuration

```yaml
output_report:
  discharge: true        # Export discharge time series
  lateral_inflow: true   # Export lateral inflow time series

output_report_settings:
  output_format: "Parquet"     # Options: "Parquet", "csv"
  reach_selection: "all"       # Options: "all", "file", "list"
  sel_list: [0, 10, 25, 100]  # List of reach IDs (for "list" mode)
  sel_file: "reaches.json"     # Path to file with reach IDs (for "file" mode)
```

### Reservoir configuration (optional)

```yaml
parameters:
  reservoirs:
    res_shape: reservoirs/reservoirs.shp
    stage_storage: reservoirs/stage_storage.csv
    regulation_curves: reservoirs/regulation_curves.csv
    regulation_schedule: reservoirs/regulation_schedule.csv

initial_conditions:
  reservoir_volumes: reservoirs/initial_volumes.csv  # Optional

paths:
  reservoirs: output/reservoirs.parquet  # Consolidated reservoir data
```

When reservoirs are configured:
- Reservoir polygons are rasterized to identify basin pixels
- Surface runoff and lateral flow are zeroed in reservoir basins
- Total inflow is computed from upstream discharge and basin contributions
- Reservoir volume is updated based on mass balance
- Stage is calculated from volume using cubic spline interpolation
- Regulated discharge is determined from time-varying stage-discharge curves
- Reservoir outflow is added to outlet reach lateral inflow
- Inlet reach discharge is zeroed to prevent double-counting

### Simulation configuration

```yaml
simulation:
  timestep: 900                      # Time step in seconds (15 minutes)
  precipitation_interp: "IDW"        # Options: "IDW", "Nearest" (for station-based forcing)
  energy_balance: "1L"               # Options: "None", "1L"

output_forcing_data:
  meteo_data: false                  # Save meteorological forcing grids
                                     # Output file: {output_dir}/meteo_forcing.nc
```

**Notes:**
- `precipitation_interp` only applies when using station-based forcing (`MeteoData`)
- When using raster-based forcing (`MeteoRaster`), interpolation is skipped entirely
- When `energy_balance == "1L"` the simulation additionally requires `temperature_min`, `temperature_max`, `humidity`, `wind_speed`, and `radiation` forcing variables, and the `basin.baricenter` coordinates (for solar hours). See the [Energy Balance reference](energy_balance.md) for the full configuration.
- When the input `MeteoRaster` already contains a `pet` variable, the energy balance is automatically bypassed and PET is read directly from the raster.

## Implemented modules

- [Preprocessing](preprocessing.md) - GIS data and reservoir preprocessing
- [Soil Water Balance](soil_water_balance.md) - Hillslope water balance
- [Routing](routing.md) - Hillslope, channel, and reservoir routing
- [Groundwater](groundwater.md) - Linear-reservoir groundwater dynamics
- [Energy Balance](energy_balance.md) - 1-layer analytical surface energy budget and PET
- [State I/O](state.md) - NetCDF state export/import
- [Report I/O](report.md) - Time series export/import

