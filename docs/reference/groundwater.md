# Groundwater

The groundwater module implements saturated-zone dynamics, producing the baseflow contribution that is added to surface runoff before lateral inflow accumulation in the river network.

## Overview

Currently, a single **linear reservoir** model is available. Each active cell is treated as an independent linear reservoir with a governing equation:

$$
q = k_f \cdot h, \qquad \frac{dh}{dt} = R - q
$$

where $h$ is the groundwater head [m], $k_f$ is the aquifer conductivity [1/s], $R$ is the net recharge [m/s], and $q$ is the baseflow [m/s].

The equation is solved analytically over each time step, and the average baseflow over the step is returned. Recharge is internally clamped to prevent the head from going negative.

**Recharge** is computed inside the simulation loop as:

$$
R = P_{\text{perc}} - \ell
$$

- $P_{\text{perc}}$: percolation flux from the soil water balance [m/s]
- $\ell$: per-cell global loss term, computed by distributing `parameters.groundwater.global_loss` [m³/s] uniformly across the active cells.

The resulting baseflow is added to the surface runoff grid before accumulation of lateral inflow into the river reaches.

## Multi-aquifer mode

The spatial structure of the aquifer is controlled by the optional `Mf` raster (freatic aquifer mask):

- **`Mf` not provided** or contains a **single positive class**: one linear reservoir is used per active cell, with no spatial grouping.
- **`Mf` contains multiple positive classes**: multi-aquifer mode is activated. After each integration step, the groundwater head $h$ is averaged within each positive class. Therefore, each class represents an independent aquifer with a single head value.

Cells with `Mf <= 0` are treated as outside the aquifer.

## Functions

::: mobidic.core.groundwater.groundwater_linear

## Configuration

Groundwater is configured under `parameters.groundwater` and `initial_conditions` in the YAML config:

```yaml
raster_files:
  # OPTIONAL: Grid of aquifer conductivity, in 1/s
  kf: example/raster/kf.tif

  # OPTIONAL: Grid mask defining the artesian aquifer(s) extension
  Ma: example/raster/Ma.tif

  # OPTIONAL: Grid defining the freatic aquifer(s) extension
  # A single positive class defines one aquifer; multiple positive classes
  # enable multi-aquifer mode (averaging of h within each class).
  Mf: example/raster/Mf.tif

parameters:
  soil:
    # Default value of aquifer conductivity, in 1/s (used if kf raster is not provided)
    kf: 1.000e-07

  groundwater:
    # REQUIRED: Groundwater model type
    # [one among: None, Linear, Dupuit, MODFLOW]
    model: Linear

    # OPTIONAL: Global water loss from aquifers, in m³/s (default: 0.0)
    global_loss: 0.0

initial_conditions:
  # OPTIONAL: Initial groundwater head [m], used when Linear model is active
  groundwater_head: 0.01

output_states:
  # OPTIONAL: Save groundwater head grid (h) to state file
  aquifer_head: true
```

**Notes:**

- `kf` can be supplied either as a raster (`raster_files.kf`) or as a single basin-wide default (`parameters.soil.kf`, default `1.0e-7` 1/s). The raster takes precedence when both are provided.
- The global loss is distributed uniformly across the active cells that participate in groundwater dynamics.
- When `output_states.aquifer_head = true`, the groundwater head field `h` is included in the NetCDF state file.

## Integration with the simulation loop

At every timestep, when `parameters.groundwater.model == "Linear"`, the simulation:

1. Computes the per-cell net recharge $R$ from the soil percolation and global-loss terms.
2. Calls [`groundwater_linear()`](groundwater.md#mobidic.core.groundwater.groundwater_linear) to update the head $h$ and obtain the average baseflow $q$.
3. (Optional) Averages $h$ within each class of the `Mf` raster in multi-aquifer mode.
4. Adds the baseflow to the surface runoff rate before lateral inflow accumulation into the river reaches.
5. Persists the updated head into [`SimulationState`](simulation.md#mobidic.core.simulation.SimulationState) (`state.h`), so it can be written to the state NetCDF when `output_states.aquifer_head` is enabled and reloaded on warm start.

## Model status

- **Linear** (single- and multi-aquifer via `Mf`)
- **Dupuit** — 2-D physics-based aquifer with explicit surface-subsurface coupling (not yet implemented)
- **MODFLOW** — coupling with USGS MODFLOW (not yet implemented)
