# Soil water balance

The soil water balance module implements the water balance, including above-ground processes (canopy interception, surface runoff) and subsurface soil processes (infiltration, percolation, lateral flow).

## Overview

The module simulates water balance across four reservoirs:

- **Capillary (Wc)**: Water held by capillary forces in soil small pores
- **Gravitational (Wg)**: Drainable water in soil large pores
- **Plants (Wp)**: Canopy interception storage (optional)
- **Surface (Ws)**: Surface depression storage

Key processes simulated:

- **Surface processes**: Canopy interception, throughfall, surface ET
- **Subsurface processes**: Horton infiltration (stochastic), Dunne runoff
- **Soil ET**: Single-parameter S-curve for soil moisture stress
- **Capillary absorption**: Wg → Wc transfer
- **Percolation**: Deep drainage with optional capillary rise feedback
- **Lateral flow**: Subsurface lateral drainage
- **Capillary rise**: Optional upward flux from groundwater (Salvucci 1993)
- **Surface routing**: Depression storage with linear routing

## Functions

::: mobidic.core.soil_water_balance.soil_mass_balance

::: mobidic.core.soil_water_balance.capillary_rise
