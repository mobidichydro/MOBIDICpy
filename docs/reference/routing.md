# Routing

The routing module implements hillslope and channel routing algorithms for water propagation through the landscape and river network.

## Overview

The module provides three main routing components:

- **Hillslope routing**: Accumulates lateral flow contributions from upslope cells following D8 flow directions
- **Channel routing**: Routes water through the river network using linear reservoir method
- **Reservoir routing**: Simulates reservoir storage dynamics with time-varying regulation and stage-discharge relationships

Hillslope and channel routing functions use **Numba JIT compilation** for high-performance execution.

## Functions

::: mobidic.core.routing.hillslope_routing

::: mobidic.core.routing.linear_channel_routing

::: mobidic.core.reservoir.reservoir_routing

::: mobidic.core.reservoir.ReservoirState


## Design Features

### Hillslope Routing

- **One-step routing**: Each call routes water ONE STEP to immediate downstream neighbors
- **Gradual propagation**: Water moves cell-by-cell over multiple timesteps
- **D8 flow direction**: Supports MOBIDIC notation (1-8)
- **NaN handling**: Robust handling of no-data cells
- **Outlet detection**: Handles outlet cells (flow_dir = 0 or -1)
- **Numba acceleration**: JIT-compiled kernel for maximum performance

### Linear Channel Routing

- **Linear reservoir model**: Simple exponential decay for each reach
- **Topological routing**: Processes reaches in correct upstream→downstream order
- **Binary tree topology**: Supports 0, 1, or 2 upstream tributaries per reach
- **Mean integral method**: Computes mean upstream discharge over time step
- **Mass conservative**: Total water balance preserved
- **Configurable storage**: Uses storage coefficient (K) from network attributes

### Reservoir Routing

- **Volume-stage-discharge model**: Simulates reservoir storage dynamics with stage-storage and stage-discharge relationships
- **Cubic spline interpolation**: Smooth stage-volume curves using cubic splines
- **Time-varying regulation**: Supports multiple regulation periods (e.g., seasonal winter/summer operations)
- **Adaptive sub-stepping**: Automatically divides timestep for numerical stability based on discharge variability
- **Negative volume handling**: Gracefully handles edge cases when reservoir would empty
- **Basin zeroing**: Automatically zeros surface runoff and lateral flow in reservoir basin pixels
- **Network integration**: Identifies inlet/outlet reaches, adds outflow to outlet lateral inflow, zeros inlet discharge

## Routing Equations

The linear reservoir routing equation for each reach is:

$$
\frac{dQ}{dt} = A(q_L + U \cdot Q - Q)
$$

where $Q$ is discharge, $q_L$ is lateral inflow (surface + groundwater), $A$ is a diagonal matrix of inverse characteristic times ($1/K$), and $U$ is a binary topology matrix indicating tributary connections.

### Linear Reservoir Model

For each reach:

$$
C_3 = \exp\left(-\frac{\Delta t}{K}\right) \quad \text{[recession coefficient]}
$$

$$
C_4 = 1 - C_3 \quad \text{[lateral inflow coefficient]}
$$

$$
Q_{\text{out}}(t + \Delta t) = C_3 \cdot Q_{\text{out}}(t) + C_4 \cdot q_L
$$

Where:

- $K$ = storage coefficient (lag time) [s]
- $q_L$ = lateral inflow + integrated upstream contributions [m³/s]
- $\Delta t$ = time step [s]

The mean integral of upstream discharge is computed as:

$$
\overline{Q} = \frac{q_L}{C_4} + \frac{q_L - Q_{\text{initial}} \cdot C_4}{\ln(C_3)}
$$

Special cases:

- $C_3 = 1$ ($K \to \infty$): No attenuation, $\overline{Q} = q_L / C_4$
- $C_3 \approx 0$ ($K \to 0$): Instant routing, $\overline{Q} = q_L / C_4$

### Reservoir Routing Model

For each reservoir at each timestep:

**1. Volume update:**
$$
V(t + \Delta t) = V(t) + (Q_{\text{in}} - Q_{\text{out}} - W) \cdot \Delta t
$$

where $V$ is volume [m³], $Q_{\text{in}}$ is total inflow (upstream discharge + basin contributions) [m³/s], $Q_{\text{out}}$ is regulated discharge [m³/s], and $W$ is withdrawal [m³/s].

**2. Stage calculation:**
$$
h = f_{\text{stage}}(V)
$$

Using cubic spline interpolation of the stage-storage curve.

**3. Regulation period determination:**

Based on the current date and regulation schedule, select the active regulation curve (e.g., "winter" or "summer").

**4. Discharge calculation:**
$$
Q_{\text{out}} = f_{\text{discharge}}(h, \text{period})
$$

Using linear interpolation of the stage-discharge curve for the active period.

**5. Sub-stepping:**

If discharge variability is high, divide $\Delta t$ into $N$ sub-steps where $N$ is determined by:
$$
N = 2^k \quad \text{where } k = \max(0, \lceil \log_2(\text{max discharge change} / \text{threshold}) \rceil)
$$

This ensures numerical stability by limiting the rate of volume change per sub-step.

**6. Negative volume handling:**

If $V(t + \Delta t) < 0$, reduce $Q_{\text{out}}$ such that $V(t + \Delta t) = 0$:
$$
Q_{\text{out, adjusted}} = \frac{V(t) + Q_{\text{in}} \cdot \Delta t - W \cdot \Delta t}{\Delta t}
$$

## Performance

Hillslope and channel routing functions use **Numba JIT compilation** for significant performance improvements:

- Hillslope routing: ~10-50× faster than pure Python
- Channel routing: ~5-20× faster than pure Python

The first call compiles the functions (slight delay), subsequent calls use cached machine code.