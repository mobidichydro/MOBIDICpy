# Energy balance

The energy balance module computes the surface energy budget at the land surface and provides the **potential evapotranspiration (PET)** that drives the soil-water balance. Surface ($T_s$) and deep-soil ($T_d$) temperatures are calculated as additional state variables.

## Overview

The currently available scheme is the **1-layer (1L)** analytical formulation. It solves a linearised Fourier decomposition of the surface energy budget, with the diurnal cycle of incoming shortwave radiation represented as a sinusoid plus a constant term. Each model timestep is split at the sunrise and sunset boundaries so that day and night sub-periods are integrated separately (radiation forcing is only present in the day sub-period).

The energy budget includes:

- Net shortwave radiation (corrected by surface albedo $\alpha$)
- Net longwave radiation (Stefan-Boltzmann linearised around mean air temperature)
- Sensible heat flux (turbulent exchange coefficient $C_H$ times wind speed)
- Latent heat flux (Magnus formula for saturation specific humidity)
- Ground heat flux (1-D conduction with deep-temperature boundary $T_{const}$)

## Two-step strategy

For each timestep the energy balance is solved in two steps:

1. **Initial** (saturated assumption, $\eta = 1$): produces the potential evapotranspiration $\text{PET}$ that the soil module will use. The ratio $\eta = \text{ET}_r / \text{ET}_p$ is set to 1 (no water limitation).
2. **Second step** (after the soil water balance): re-computes $T_s$ and $T_d$ using the actual ratio $\eta$ obtained from the soil module. The result is stored into the state only where $\eta < 1$ (water-limited cells); cells that remain saturated keep the initial result.

The deep-temperature value evaluated at sunrise is preserved across timesteps and is used as the starting point for the day sub-period in the second step calculation.

## Diurnal radiation cycle

Daily incoming shortwave radiation $R_s$ is decomposed into:

$$
R_s(t) = A \cdot \sin(\omega t + \varphi) + C
$$

with $\omega = 2\pi / 86400$ rad/s. Two modes are supported:

- **average**: $R_s$ is the day-average value; the amplitude $A$ and constant $C$ are fitted so that the integral between sunrise and sunset returns the daily mean.
- **instant**: $R_s$ is interpreted as an instantaneous value at sunrise time.

Sunrise and sunset hours are obtained by bisection on the solar elevation, computed from the basin baricenter latitude/longitude and the Julian day.

## Functions

::: mobidic.core.energy_balance.compute_energy_balance_1l

::: mobidic.core.energy_balance.energy_balance_1l

::: mobidic.core.energy_balance.diurnal_radiation_cycle

::: mobidic.core.energy_balance.solar_hours

::: mobidic.core.energy_balance.solar_position

::: mobidic.core.energy_balance.saturation_specific_humidity

## Configuration

The energy balance is enabled under `simulation.energy_balance` and is configured through `parameters.energy`, the optional `CH` / `Alb` rasters, and the `basin.baricenter` coordinates required to compute solar hours.

```yaml
basin:
  baricenter:
    lon: 11.10   # Longitude of basin baricenter [deg. East]
    lat: 43.77   # Latitude of basin baricenter [deg. North]

simulation:
  # Energy balance scheme [None | 1L]
  energy_balance: 1L

raster_files:
  # OPTIONAL: Grid of turbulent exchange coefficient for heat, non dimensional
  CH: example/raster/CH.tif

  # OPTIONAL: Grid of surface albedo, non dimensional
  Alb: example/raster/Alb.tif

parameters:
  energy:
    Tconst: 290.0     # Deep ground temperature [K] (default: 290.0)
    kaps:   2.5       # Soil thermal conductivity [W/m/K] (default: 2.5)
    nis:    0.8e-6    # Soil thermal diffusivity [m^2/s] (default: 0.8e-6)
    CH:     1.0e-3    # Default turbulent exchange coefficient (used if CH raster missing)
    Alb:    0.2       # Default surface albedo (used if Alb raster missing)

  multipliers:
    CH_factor: 1.0    # Calibration multiplier applied to CH raster/scalar

output_states:
  surface_temperature: true   # Save Ts grid to state file
  ground_temperature:  true   # Save Td grid to state file

output_forcing_data:
  meteo_data: true            # Saves precipitation, temperature_min/max, humidity,
                              # wind_speed, radiation, and pet to meteo_forcing.nc
```

**Required meteorological forcing.** When `simulation.energy_balance == "1L"`, the simulation needs the following variables in addition to precipitation:

- `temperature_min`, `temperature_max` — daily air temperature extremes [°C]
- `humidity` — relative humidity [%]
- `wind_speed` — wind speed [m/s]
- `radiation` — incoming shortwave radiation [W/m²]

Both station-based ([`MeteoData`](meteo.md#mobidic.preprocessing.meteo_preprocessing.MeteoData)) and raster-based ([`MeteoRaster`](meteo.md#mobidic.preprocessing.meteo_raster.MeteoRaster)) inputs are supported.

## Integration with the simulation loop

When `simulation.energy_balance == "1L"`:

1. The simulation initialises $T_s = T_d = T_{air,lin}$ on the first step (when no warm start is used) or restores them from the loaded state file. `td_rise` is initialised from $T_d$.
2. Each timestep, the **initial step** is run with $\eta = 1$ to obtain PET and a tentative $(T_s, T_d)$.
3. PET feeds the soil water balance, which produces the actual evapotranspiration ET.
4. The **second step** is run with $\eta = \text{ET} / \text{PET}$, restarting the day sub-period from the initial `td_rise`. The result overwrites $(T_s, T_d)$ only on water-limited cells.
5. $T_s$ and $T_d$ are written to the state file when the corresponding `output_states` flags are enabled.

## Use of PET in raster forcing: speed up

When the input [`MeteoRaster`](meteo.md#mobidic.preprocessing.meteo_raster.MeteoRaster) already contains a `pet` variable (for example, a `meteo_forcing.nc` produced by an earlier run), the simulation **skips the energy balance entirely** and reads PET directly from the raster. In this case, the simulation **speeds up** significantly.
In this mode, $T_s$ and $T_d$ states are not updated (they keep their initial values), and the temperature/humidity/wind/radiation variables are **not required** in the input file.

## Model status

- **None** — energy balance disabled, constant 1 mm/day PET is used
- **1L** — single-layer analytical Fourier scheme (this module)
- **5L** — multi-layer soil temperature profile (not yet implemented)
- **Snow** — snow accumulation and melt (not yet implemented)
