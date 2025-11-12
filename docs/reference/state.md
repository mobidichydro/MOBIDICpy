# State I/O

The state I/O module provides functions to save and load MOBIDIC simulation state variables in NetCDF format, enabling simulation restart and state analysis.

## Overview

State files store:

- **Grid variables**: Capillary water (Wc), gravitational water (Wg), plant water (Wp), surface water (Ws)
- **Network variables**: Discharge, lateral inflow for each reach
- **Metadata**: Simulation time, grid coordinates, CRS, global attributes
- **CF-1.12 compliance**: NetCDF Climate and Forecast metadata conventions

State files enable:

- **Warm start**: Resume simulations from saved state
- **State analysis**: Examine spatial patterns of soil moisture, discharge
- **Model evaluation**: Compare simulated states against observations
- **Ensemble runs**: Initialize multiple simulations from different states

## Functions

::: mobidic.io.state.save_state

::: mobidic.io.state.load_state

## Examples

### Saving State During Simulation

```python
from mobidic import Simulation, load_config, load_gisdata, MeteoData
from datetime import datetime

# Initialize and run simulation
config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")
forcing = MeteoData.from_netcdf("meteo.nc")
sim = Simulation(gisdata, forcing, config)

results = sim.run("2020-01-01", "2020-12-31")

# Save final state
results.save_final_state("output/state_final.nc")

# Or save with custom metadata
results.save_final_state(
    "output/state_final.nc",
    add_metadata={
        "simulation_version": "v1.0",
        "calibration_run": "baseline",
        "notes": "Calibration run with default parameters"
    }
)
```

### Manual State Saving

```python
from mobidic.io import save_state
from datetime import datetime

# After running simulation
save_state(
    state=sim.state,
    output_path="output/state_2020-06-15.nc",
    time=datetime(2020, 6, 15),
    grid_metadata=sim.gisdata.metadata,
    network_size=len(sim.network),
    output_states=config.output_states,
    add_metadata={"run_type": "spinup"}
)
```

### Loading State for Warm Start

```python
from mobidic.io import load_state
from mobidic import Simulation

# Load previously saved state
state, time, metadata = load_state(
    input_path="output/state_2020-06-15.nc",
    network_size=1235  # Number of reaches in network
)

print(f"Loaded state at {time}")
print(f"Grid shape: {metadata['shape']}")
print(f"Mean capillary water: {state.wc.mean():.3f} m")
print(f"Mean discharge: {state.discharge.mean():.3f} m³/s")

# Use state to initialize simulation
sim = Simulation(gisdata, forcing, config)
sim.state = state  # Override initial state

# Resume simulation from this state
results = sim.run("2020-06-15", "2020-12-31")
```

### Inspecting State Files

```python
import xarray as xr
import matplotlib.pyplot as plt

# Open state file
ds = xr.open_dataset("output/state_final.nc")

# Examine contents
print(ds)
print(f"\nVariables: {list(ds.data_vars)}")
print(f"Coordinates: {list(ds.coords)}")
print(f"Time: {ds.time.values}")

# Plot capillary water content
if "Wc" in ds:
    fig, ax = plt.subplots(figsize=(10, 8))
    ds["Wc"].plot(ax=ax, cmap="Blues", cbar_kwargs={"label": "Wc [m]"})
    ax.set_title("Capillary Water Content")
    plt.savefig("output/Wc_map.png")

# Plot discharge along network
if "discharge" in ds:
    plt.figure(figsize=(12, 4))
    plt.plot(ds["reach"], ds["discharge"], 'b-', linewidth=0.5)
    plt.xlabel("Reach ID")
    plt.ylabel("Discharge [m³/s]")
    plt.title("River Discharge")
    plt.grid(True)
    plt.savefig("output/discharge_profile.png")

ds.close()
```

### Comparing States

```python
import xarray as xr
import numpy as np

# Load two states to compare
ds1 = xr.open_dataset("output/state_run1.nc")
ds2 = xr.open_dataset("output/state_run2.nc")

# Compare capillary water
if "Wc" in ds1 and "Wc" in ds2:
    diff = ds1["Wc"] - ds2["Wc"]

    print(f"Wc difference statistics:")
    print(f"  Mean: {float(diff.mean()):.6f} m")
    print(f"  Std: {float(diff.std()):.6f} m")
    print(f"  Max abs diff: {float(np.abs(diff).max()):.6f} m")

    # Plot difference
    fig, ax = plt.subplots(figsize=(10, 8))
    diff.plot(ax=ax, cmap="RdBu_r", center=0)
    ax.set_title("Capillary Water Difference (Run1 - Run2)")
    plt.savefig("output/Wc_diff.png")

ds1.close()
ds2.close()
```

## Configuration Control

State saving is controlled by the configuration file:

```yaml
output_states:
  discharge: true               # Save river discharge
  soil_capillary: true          # Save capillary water (Wc)
  soil_gravitational: true      # Save gravitational water (Wg)
  soil_plant: true              # Save plant/canopy water (Wp)
  soil_surface: true            # Save surface water (Ws)
  # Other state outputs (not yet implemented):
  reservoir_states: false
  surface_temperature: false
  ground_temperature: false
  aquifer_head: false
  et_prec: false

output_states_settings:
  output_format: "netCDF"       # Format (currently only netCDF)
  output_states: "final"        # "all", "final", or "list"
  output_interval: 3600         # Interval in seconds (for "all")
  output_list: [0, 100, 200]    # Timestep indices (for "list")
```

## File Structure

NetCDF state files contain:

### Dimensions
- `x`: Grid columns
- `y`: Grid rows
- `reach`: Number of reaches (if discharge enabled)

### Coordinates
- `x(x)`: X coordinates [m]
- `y(y)`: Y coordinates [m]
- `reach(reach)`: Reach indices [dimensionless]
- `time`: Simulation time [datetime64]

### Data Variables
- `Wc(y, x)`: Capillary water content [m]
- `Wg(y, x)`: Gravitational water content [m]
- `Wp(y, x)`: Plant/canopy water content [m] (optional)
- `Ws(y, x)`: Surface water content [m]
- `discharge(reach)`: River discharge [m³/s] (if enabled)
- `lateral_inflow(reach)`: Lateral inflow to reaches [m³/s] (if enabled)
- `crs`: Grid mapping (CRS metadata)

### Global Attributes
- `title`: "MOBIDIC Simulation State"
- `source`: "MOBIDICpy vX.X.X"
- `Conventions`: "CF-1.12"
- `creation_date`: ISO 8601 timestamp
- Custom metadata from `add_metadata` parameter

## Design Features

- **CF-1.12 compliant**: Follows Climate and Forecast metadata conventions
- **Compression**: zlib compression (level 4) for efficient storage
- **Selective saving**: Only configured state variables are saved
- **Flexible loading**: Missing variables initialized with sensible defaults (NaN for grids, zeros for networks)
- **CRS preservation**: Coordinate Reference System stored as WKT
- **Robust error handling**: Clear warnings for missing data or size mismatches

## Error Handling

The module provides clear error messages for common issues:

```python
from mobidic.io import load_state

try:
    state, time, metadata = load_state("missing_file.nc", 1235)
except FileNotFoundError as e:
    print(f"Error: {e}")

try:
    state, time, metadata = load_state("state.nc", 5000)  # Wrong network size
except ValueError as e:
    print(f"Warning: {e}")  # Warning about size mismatch, but loads anyway
```

## References

**File format**:

- NetCDF4 with CF-1.12 conventions
- Uses xarray for reading/writing
- Compatible with standard NetCDF tools (ncdump, ncview, Panoply)

**Related modules**:

- [Simulation](simulation.md) - Uses state I/O for warm start and final states
- [Report I/O](report.md) - Time series output (Parquet format)
