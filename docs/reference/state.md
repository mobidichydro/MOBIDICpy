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
- **Large simulations**: Automatic file chunking for simulations exceeding size limits

## Classes and functions

::: mobidic.io.state.StateWriter

::: mobidic.io.state.load_state

## Examples

### Saving states incrementally with StateWriter

```python
from mobidic.io import StateWriter
from mobidic import load_config, load_gisdata
from datetime import datetime, timedelta

# Load configuration and data
config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")

# Create StateWriter with flushing every 10 timesteps
with StateWriter(
    output_path="output/states.nc",
    grid_metadata=gisdata.metadata,
    network_size=len(gisdata.network),
    output_states=config.output_states,
    flushing=10,  # Flush every 10 timesteps (-1 = only at end)
    max_file_size=500.0,  # Create new chunk file when reaching 500 MB
    add_metadata={
        "simulation_version": "v1.0",
        "calibration_run": "baseline",
        "notes": "Calibration run with default parameters"
    }
) as writer:
    # Simulation loop
    current_time = datetime(2020, 1, 1)
    dt = timedelta(seconds=900)  # 15-minute timesteps

    for step in range(num_steps):
        # ... run simulation step ...
        # state = perform_simulation_step(...)

        # Append state to file (buffered)
        writer.append_state(state, current_time)
        current_time += dt

    # States are automatically flushed and file closed when exiting context
    # May create: states_001.nc, states_002.nc, states_003.nc, etc.

# Alternatively, manually manage the writer
writer = StateWriter(
    output_path="output/states.nc",
    grid_metadata=gisdata.metadata,
    network_size=len(gisdata.network),
    output_states=config.output_states,
    flushing=-1,  # Only flush at end
    max_file_size=500.0,
)

for step in range(num_steps):
    # ... simulation ...
    writer.append_state(state, current_time)
    current_time += dt

writer.close()
```

### Saving only final state

```python
from mobidic.io import StateWriter
from datetime import datetime

# Create writer with flushing=-1 (only flush at end)
writer = StateWriter(
    output_path="output/state_final.nc",
    grid_metadata=gisdata.metadata,
    network_size=len(gisdata.network),
    output_states=config.output_states,
    flushing=-1,
    add_metadata={"run_type": "final_state"}
)

# Only save the final state
writer.append_state(final_state, datetime(2020, 12, 31, 23, 45))
writer.close()
```

### Loading state for warm start

```python
from mobidic.io import load_state
from mobidic import Simulation, load_config, load_gisdata

# Load last timestep from multi-timestep file (default)
state, time, metadata = load_state(
    input_path="output/states.nc",
    network_size=1235  # Number of reaches in network
)

print(f"Loaded state at {time}")
print(f"Grid shape: {metadata['shape']}")
print(f"Mean capillary water: {state.wc.mean():.3f} m")
print(f"Mean discharge: {state.discharge.mean():.3f} m³/s")

# Load specific timestep (e.g., first timestep)
state_first, time_first, _ = load_state(
    input_path="output/states.nc",
    network_size=1235,
    time_index=0  # 0 = first, -1 = last (default)
)

# Load from single-timestep file
state_final, time_final, _ = load_state(
    input_path="output/state_final.nc",
    network_size=1235
)

# Load from chunked files (automatically detects chunk files)
# Even if you specify "states.nc", it will find and load "states_001.nc"
state_chunk, time_chunk, _ = load_state(
    input_path="output/states.nc",  # Auto-detects states_001.nc
    network_size=1235
)

# Load with config and gisdata for proper missing variable initialization
# If the state file is missing some variables, they will be initialized
# using the initial conditions from the config file. If config/gisdata are not
# provided, missing variables are set to zero.
config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")

state_with_init, time, metadata = load_state(
    input_path="output/states.nc",
    network_size=1235,
    config=config,      # Optional: for missing variable initialization
    gisdata=gisdata     # Optional: for missing variable initialization
)

# Use state to initialize simulation
sim = Simulation(gisdata, forcing, config)
sim.state = state  # Override initial state

# Resume simulation from this state
results = sim.run("2020-06-15", "2020-12-31")
```

### Handling missing variables

When loading a state file, some variables may be missing (e.g., if the file was saved with selective output). The `load_state()` function handles this automatically:

**Without config/gisdata (default behavior):**
```python
# Missing variables are initialized with zeros
state, time, metadata = load_state("output/states.nc", network_size=1235)

# Grid variables (Wc, Wg, Ws, Wp) → initialized with zeros
# Network variables (discharge, lateral_inflow) → initialized with zeros
```

**With config/gisdata (recommended for warm start):**
```python
from mobidic import load_config, load_gisdata

config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")

# Missing variables are initialized using config initial conditions
state, time, metadata = load_state(
    "output/states.nc",
    network_size=1235,
    config=config,
    gisdata=gisdata
)

# Grid variables (Wc, Wg, Ws, Wp) → initialized from config.initial_conditions
#   - Wc: Wc0 × Wcsat (with capacity factors and Wg_Wc_tr transition)
#   - Wg: Wg0 × Wgsat (with capacity factors and Wg_Wc_tr transition)
#   - Ws: config.initial_conditions.Ws
#   - Wp: zeros (with NaN outside domain)
#   - All grid variables: NaN outside domain (where flow_acc is NaN)
# Network variables (discharge, lateral_inflow) → initialized with zeros
```

**Example: Loading a partial state file**
```python
# Suppose a state file was saved with only Wc and Wg enabled
# (soil_surface=false in config), but you need Ws for the new simulation

config = load_config("config.yaml")
gisdata = load_gisdata("gisdata.nc", "network.parquet")

# Load state with config - Ws will be initialized from config.initial_conditions.Ws
state, time, metadata = load_state(
    "output/partial_state.nc",
    network_size=1235,
    config=config,
    gisdata=gisdata
)

# Now state.ws is properly initialized (not zeros) and ready for simulation
print(f"Ws initialized from config: {state.ws.mean():.6f} m")

# The Simulation class automatically uses config-based initialization
sim = Simulation(gisdata, forcing, config)
sim.set_initial_state(state_file="output/partial_state.nc")
# Missing variables are automatically initialized using config
```

This ensures that warm-start simulations always have properly initialized state variables, even when loading from files that don't contain all variables.

### Inspecting state files

```python
import xarray as xr
import matplotlib.pyplot as plt

# Open multi-timestep state file
ds = xr.open_dataset("output/states.nc")

# Examine contents
print(ds)
print(f"\nVariables: {list(ds.data_vars)}")
print(f"Coordinates: {list(ds.coords)}")
print(f"Number of timesteps: {len(ds.time)}")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# Plot capillary water content at last timestep
if "Wc" in ds:
    fig, ax = plt.subplots(figsize=(10, 8))
    ds["Wc"].isel(time=-1).plot(ax=ax, cmap="Blues", cbar_kwargs={"label": "Wc [m]"})
    ax.set_title(f"Capillary Water Content at {ds.time.values[-1]}")
    plt.savefig("output/Wc_map.png")

# Plot time series of mean soil moisture
if "Wc" in ds:
    mean_wc = ds["Wc"].mean(dim=["x", "y"])
    plt.figure(figsize=(12, 4))
    mean_wc.plot()
    plt.ylabel("Mean Wc [m]")
    plt.title("Mean Capillary Water Content Over Time")
    plt.grid(True)
    plt.savefig("output/Wc_timeseries.png")

# Plot discharge along network at last timestep
if "discharge" in ds:
    plt.figure(figsize=(12, 4))
    plt.plot(ds["reach"], ds["discharge"].isel(time=-1), 'b-', linewidth=0.5)
    plt.xlabel("Reach ID")
    plt.ylabel("Discharge [m³/s]")
    plt.title(f"River Discharge at {ds.time.values[-1]}")
    plt.grid(True)
    plt.savefig("output/discharge_profile.png")

# Plot discharge time series for a specific reach
if "discharge" in ds:
    reach_id = 500  # Example reach
    plt.figure(figsize=(12, 4))
    ds["discharge"].isel(reach=reach_id).plot()
    plt.ylabel("Discharge [m³/s]")
    plt.title(f"Discharge Time Series for Reach {reach_id}")
    plt.grid(True)
    plt.savefig("output/discharge_ts.png")

ds.close()
```

### Comparing states

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load two state files to compare
ds1 = xr.open_dataset("output/states_run1.nc")
ds2 = xr.open_dataset("output/states_run2.nc")

# Compare capillary water at last timestep
if "Wc" in ds1 and "Wc" in ds2:
    wc1_final = ds1["Wc"].isel(time=-1)
    wc2_final = ds2["Wc"].isel(time=-1)
    diff = wc1_final - wc2_final

    print(f"Wc difference statistics (final timestep):")
    print(f"  Mean: {float(diff.mean()):.6f} m")
    print(f"  Std: {float(diff.std()):.6f} m")
    print(f"  Max abs diff: {float(np.abs(diff).max()):.6f} m")

    # Plot difference
    fig, ax = plt.subplots(figsize=(10, 8))
    diff.plot(ax=ax, cmap="RdBu_r", center=0)
    ax.set_title("Capillary Water Difference (Run1 - Run2)")
    plt.savefig("output/Wc_diff.png")

# Compare time series
if "Wc" in ds1 and "Wc" in ds2:
    mean_wc1 = ds1["Wc"].mean(dim=["x", "y"])
    mean_wc2 = ds2["Wc"].mean(dim=["x", "y"])

    plt.figure(figsize=(12, 4))
    mean_wc1.plot(label="Run 1")
    mean_wc2.plot(label="Run 2")
    plt.ylabel("Mean Wc [m]")
    plt.title("Mean Capillary Water Content Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/Wc_comparison.png")

ds1.close()
ds2.close()
```

### Working with chunked state files

When saving very large simulations, `StateWriter` automatically splits the output into multiple chunk files when the size limit is reached:

```python
from mobidic.io import StateWriter, load_state
from datetime import datetime, timedelta

# Create writer with 500 MB chunk size and regular flushing
# NOTE: For chunking to work effectively, flushing must be > 0
writer = StateWriter(
    output_path="output/states.nc",
    grid_metadata=gisdata.metadata,
    network_size=len(gisdata.network),
    output_states=config.output_states,
    flushing=100,  # Flush every 100 steps (required for chunking)
    max_file_size=500.0,  # 500 MB per chunk
)

# Run long simulation that generates > 500 MB of data
for step in range(10000):
    writer.append_state(state, current_time)
    current_time += dt

writer.close()

# This creates multiple files:
# - states_001.nc (500 MB)
# - states_002.nc (500 MB)
# - states_003.nc (200 MB)

# Loading automatically detects and uses the first chunk
state, time, metadata = load_state(
    input_path="output/states.nc",  # Automatically finds states_001.nc
    network_size=1235
)

# Or load from a specific chunk directly
state_chunk2, time_chunk2, _ = load_state(
    input_path="output/states_002.nc",  # Load from second chunk
    network_size=1235,
    time_index=-1  # Last timestep in this chunk
)
```

**Important notes about chunking:**

- Files may slightly exceed `max_file_size` by up to one flush worth of data
- Size check occurs before writing each flush batch
- For effective chunking, set `flushing > 0` (e.g., 10, 50, 100)
- If `flushing=-1` (flush only at end), all data writes in one operation and chunking won't work as expected
- Chunk files are numbered sequentially: `_001.nc`, `_002.nc`, `_003.nc`, etc.
- Existing chunk files are automatically removed when creating a new StateWriter with the same base path

## Configuration control

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

## File structure

NetCDF state files contain:

### Dimensions
- `time`: Unlimited dimension for multiple timesteps
- `x`: Grid columns
- `y`: Grid rows
- `reach`: Number of reaches (if discharge enabled)

### Coordinates
- `time(time)`: Simulation time [datetime64]
- `x(x)`: X coordinates [m]
- `y(y)`: Y coordinates [m]
- `reach(reach)`: Reach indices [dimensionless]

### Data variables
- `Wc(time, y, x)`: Capillary water content [m] (if enabled)
- `Wg(time, y, x)`: Gravitational water content [m] (if enabled)
- `Wp(time, y, x)`: Plant/canopy water content [m] (optional, if enabled)
- `Ws(time, y, x)`: Surface water content [m] (if enabled)
- `discharge(time, reach)`: River discharge [m³/s] (if enabled)
- `lateral_inflow(time, reach)`: Lateral inflow to reaches [m³/s] (if enabled)
- `crs()`: Grid mapping (CRS metadata, scalar)

### Global attributes
- `title`: "MOBIDIC simulation states"
- `source`: "MOBIDICpy simulation"
- `Conventions`: "CF-1.12"
- `history`: Creation timestamp with MOBIDICpy version
- Custom metadata from `add_metadata` parameter

### Chunked files

When a simulation produces very large state files, `StateWriter` automatically creates multiple chunk files:

- **Naming convention**: `states_001.nc`, `states_002.nc`, `states_003.nc`, etc.
- **Size limit**: Each chunk is limited to `max_file_size` MB (default: 500 MB)
- **Independent files**: Each chunk is a complete, self-contained NetCDF file with its own timesteps
- **Sequential ordering**: Chunks are created in temporal order as the simulation progresses
- **Automatic detection**: `load_state()` automatically finds `states_001.nc` when you specify `states.nc`

## Design features

- **CF-1.12 compliant**: Follows Climate and Forecast metadata conventions
- **Incremental writing**: StateWriter appends states to a single file with unlimited time dimension
- **Automatic chunking**: Splits large outputs into multiple files when size limit is reached (configurable via `max_file_size`)
- **Memory efficient**: Configurable buffering with periodic flushing to disk
- **Fast append mode**: Uses netCDF4 library for efficient appending (avoids read-concatenate-write)
- **Compression**: zlib compression (level 4) for efficient storage
- **Selective saving**: Only configured state variables are saved
- **Context manager support**: Automatic resource cleanup with `with` statement
- **Multi-timestep support**: Can load any timestep from multi-timestep files
- **Chunk file detection**: `load_state()` automatically finds chunk files (e.g., `_001.nc`) when base path doesn't exist
- **CRS preservation**: Coordinate Reference System stored as WKT
- **Robust error handling**: Clear warnings for missing data or size mismatches

## Error handling

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
