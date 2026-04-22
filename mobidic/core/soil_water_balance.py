"""
Soil water balance module for MOBIDIC.

This module implements the hillslope water balance, including above-ground
processes (canopy interception, surface runoff) and subsurface soil processes
(infiltration, percolation, lateral flow).

Translated from MATLAB: massbalance_step_soil.m and caprise.m
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def capillary_rise(
    s: NDArray[np.float64],
    z: NDArray[np.float64],
    ksat: NDArray[np.float64],
    psi1: NDArray[np.float64],
    a: NDArray[np.float64],
    n: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate capillary rise from groundwater table.
    (Currently not used in main water balance function but available for future use)

    Uses the Salvucci (1993) approximate solution for steady vertical flux of
    moisture through unsaturated homogeneous soil.

    Args:
        s: Volumetric soil moisture content / porosity [-],
            = (Wr+Wc+Wg)/(Wr+Wc0+Wg0), range [0-1]
        z: Distance from soil center to water table [m]
        ksat: Saturated hydraulic conductivity [m/dt]
        psi1: Bubbling pressure head [m]
        a: Brooks-Corey parameter a = -1/m [-]
        n: Brooks-Corey parameter n = m*c [-]

    Returns:
        Capillary rise flux [m/dt]

    References:
        - Salvucci, G. D. (1993), An approximate solution for steady vertical
          flux of moisture through an unsaturated homogenous soil, WRR 29(11),
          pp. 3749-3753
        - Bras, R. L. (1990), Hydrology: an introduction to hydrologic science,
          Addison-Wesley Publishing Company, pp. 350-352

    Note:
        - Returns 0 for cells where z <= 0
        - Handles NaN values by setting them to 0
        - Uses real part only (discards imaginary components if any)
    """
    # Matric potential [m]
    psi = psi1 * (s**a)

    # Simplifications for better speed
    # Use errstate to suppress warnings for division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (psi1 / z) ** n
        term2 = (psi1 / psi) ** n

        # Capillary rise [m/dt]
        qcap = ksat * (term1 - term2) / (1 + term2 + (n - 1) * term1)

    # Checks & corrections
    qcap = np.real(qcap)
    qcap[np.isnan(qcap)] = 0
    qcap[z <= 0] = 0

    return qcap


def soil_mass_balance(
    # State variables (current and maximum capacities)
    wc: NDArray[np.float64],
    wc0: NDArray[np.float64],
    wg: NDArray[np.float64],
    wg0: NDArray[np.float64],
    wp: NDArray[np.float64] | None,
    wp0: NDArray[np.float64] | None,
    ws: NDArray[np.float64],
    ws0: NDArray[np.float64] | None,
    wtot0: NDArray[np.float64],
    # Inputs
    precipitation: NDArray[np.float64],
    surface_runoff_in: NDArray[np.float64],
    lateral_flow_in: NDArray[np.float64],
    potential_et: NDArray[np.float64],
    # Soil parameters
    hydraulic_conductivity: NDArray[np.float64] | None,
    hydraulic_conductivity_min: NDArray[np.float64] | None,
    hydraulic_conductivity_max: NDArray[np.float64] | None,
    channelized_fraction: NDArray[np.float64],
    surface_flow_exp: NDArray[np.float64],
    lateral_flow_coeff: NDArray[np.float64],
    percolation_coeff: NDArray[np.float64],
    absorption_coeff: NDArray[np.float64],
    rainfall_fraction: NDArray[np.float64],
    et_shape: float,
    # Capillary rise parameters (optional)
    capillary_rise_enabled: bool,
    soil_depth: NDArray[np.float64] | None = None,
    depth_to_water_table: NDArray[np.float64] | None = None,
    capillary_conductivity: NDArray[np.float64] | None = None,
    bubbling_pressure: NDArray[np.float64] | None = None,
    capillary_param_a: NDArray[np.float64] | None = None,
    capillary_param_n: NDArray[np.float64] | None = None,
    capillary_multiplier: NDArray[np.float64] | None = None,
    # Other parameters
    test_mode: bool = False,
    alpsur: NDArray[np.float64] | None = None,
    no_aquifer_indices: NDArray[np.int_] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Perform hillslope water balance for one time step.

    This function simulates water balance across four reservoirs:
    - Capillary (Wc): Water held by capillary forces in soil
    - Gravitational (Wg): Drainable water in soil
    - Plants (Wp): Canopy interception storage
    - Surface (Ws): Surface depression storage

    Args:
        wc: Initial water in capillary reservoir [m]
        wc0: Maximum capacity of capillary reservoir [m]
        wg: Initial water in gravitational reservoir [m]
        wg0: Maximum capacity of gravitational reservoir [m]
        wp: Initial water in plant canopy reservoir [m] (None to disable)
        wp0: Maximum capacity of plant canopy reservoir [m] (None to disable)
        ws: Initial water in surface depression reservoir [m]
        ws0: Maximum capacity of surface depression reservoir [m] (unused in current version)
        wtot0: Total soil capacity (wc0 + wg0) [m]
        precipitation: Precipitation [m]
        surface_runoff_in: Total surface runoff from upstream cells [m]
        lateral_flow_in: Lateral (subsurface) flow from upstream cells [m]
        potential_et: Potential evapotranspiration [m]
        hydraulic_conductivity: Saturated hydraulic conductivity [m] (None to use min/max)
        hydraulic_conductivity_min: Minimum hydraulic conductivity [m]
        hydraulic_conductivity_max: Maximum hydraulic conductivity [m]
        channelized_fraction: Fraction of channelized runoff [-]
        surface_flow_exp: Exponential parameter for surface flow (exp(-alp*alpsur*dt)) [-]
        lateral_flow_coeff: Lateral flow parameter (bet*dt) [-]
        percolation_coeff: Percolation parameter (gam*dt) [-]
        absorption_coeff: Capillary absorption parameter (kap*dt) [-]
        rainfall_fraction: Fraction of rainfall area (f0) [-]
        et_shape: Shape parameter for ET (0-5, default=0, recommended=3) [-]
        capillary_rise_enabled: Enable capillary rise simulation
        soil_depth: Depth of modeled topsoil [m]
        depth_to_water_table: Depth from surface to groundwater table [m]
        capillary_conductivity: Capillary hydraulic conductivity [m]
        bubbling_pressure: Bubbling pressure head [m]
        capillary_param_a: Capillary rise parameter a [-]
        capillary_param_n: Capillary rise parameter n [-]
        capillary_multiplier: Binary multiplier for capillary rise [-]
        test_mode: Enable mass balance checking (default=False)
        alpsur: Surface flow parameter (alpsur*dt) [-]
        no_aquifer_indices: Indices of cells outside aquifer

    Returns:
        Tuple containing:
        - wc: Updated water in capillary reservoir [m]
        - wg: Updated water in gravitational reservoir [m]
        - wp: Updated water in plant canopy reservoir [m] (or None)
        - ws: Updated water in surface depression reservoir [m]
        - surface_runoff_out: Total surface runoff [m]
        - lateral_flow_out: Lateral subsurface flow [m]
        - et: Evapotranspiration [m]
        - percolation: Percolation to deep groundwater [m]
        - capillary_flux: Capillary rise [m]
        - wg_before_absorption: Wg before absorption (for assimilation) [m]

    Note:
        All fluxes and parameters are assumed to be pre-multiplied by dt (time step).
        Time basis is 1 time step or dt seconds.
        Fluxes have units of meters (not m/s).
    """
    tolerance = 1e-8

    # Initialize output arrays
    n_cells = len(wc)
    zeros = np.zeros(n_cells, dtype=np.float64)

    # Handle optional plant reservoir
    if wp is None or wp0 is None:
        wp = zeros.copy()
        wp0 = zeros.copy()
        wp_enabled = False
    else:
        wp_enabled = True

    # Test mode: check inputs
    if test_mode:
        if np.any(lateral_flow_coeff > 1 + tolerance):
            logger.warning("lateral_flow_coeff > 1 detected")
        if np.any(percolation_coeff > 1 + tolerance):
            logger.warning("percolation_coeff > 1 detected")
        if np.any(absorption_coeff > 1 + tolerance):
            logger.warning("absorption_coeff > 1 detected")
        if np.any(channelized_fraction > 1 + tolerance):
            logger.warning("channelized_fraction > 1 detected")
        if np.any(potential_et < -tolerance):
            logger.warning("potential_et < 0 detected")
        water_in = wc + wg + wp + ws + precipitation + surface_runoff_in + lateral_flow_in

    # ========================================================================
    # SURFACE PROCESSES
    # ========================================================================

    # Plants / Canopy Interception
    balance = wp + precipitation  # Water available to plants + new precipitation
    et1 = np.minimum(balance, potential_et)  # ET from canopy & new precipitation
    remaining_et = potential_et - et1  # Remaining ET potential
    balance = balance - et1
    wp = np.minimum(balance, wp0)  # Update plant reservoir

    # Flow from upstream cells - separate channelized and unchannelized
    surface_runoff_channelized = channelized_fraction * surface_runoff_in
    lateral_flow_channelized = zeros.copy()  # Currently set to 0 in MATLAB
    surface_runoff_unchannelized = surface_runoff_in - surface_runoff_channelized
    lateral_flow_unchannelized = lateral_flow_in - lateral_flow_channelized

    # Throughfall + unchannelized surface runoff from upstream
    balance = (balance - wp) + surface_runoff_unchannelized

    # ET from ground surface
    et2 = np.minimum(balance, remaining_et)
    remaining_et = remaining_et - et2
    balance = balance - et2

    # ========================================================================
    # SUB-SURFACE PROCESSES
    # ========================================================================

    # Horton runoff and infiltration (using random variable approach)
    # Augmentation factor (increases as soil becomes drier)
    with np.errstate(divide="ignore", invalid="ignore"):
        kaug = 1 + 500 * np.exp(-50 * wc / wc0) / ((wc / wc0) ** 2)
    kaug[np.isinf(kaug)] = 1 + 500 * np.exp(-50 * 1e-5) / (1e-5**2)

    # Handle zero balance to avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        if hydraulic_conductivity is not None:
            # Using single ks value
            horton_runoff = balance * np.exp(-(1 - rainfall_fraction) * hydraulic_conductivity * kaug / balance)
        else:
            # Using ks_min and ks_max
            horton_runoff = (
                balance
                * balance
                / (hydraulic_conductivity_max - hydraulic_conductivity_min)
                * (
                    np.exp(-(1 - rainfall_fraction) * hydraulic_conductivity_min * kaug / balance)
                    - np.exp(-(1 - rainfall_fraction) * hydraulic_conductivity_max * kaug / balance)
                )
            )

    # Set horton runoff to zero where balance is zero
    horton_runoff[balance == 0] = 0
    horton_runoff[~np.isfinite(horton_runoff)] = 0

    # Infiltration
    infiltration = balance - horton_runoff
    wg = wg + infiltration + lateral_flow_unchannelized

    # Soil ET using single-parameter S-curve
    if et_shape:
        # Effective soil saturation [-]
        effective_saturation = (wc + wg) / wtot0
        effective_saturation[effective_saturation > 1] = 1
        remaining_et = remaining_et / (1 + np.exp(et_shape - 10 * effective_saturation))

    # Standard absorption model
    capillary_absorption = absorption_coeff * (wc0 - wc)
    capillary_absorption = np.minimum.reduce([wg, capillary_absorption, wc0 - wc + remaining_et])

    # Store Wg before absorption (for data assimilation)
    wg_before_absorption = wg.copy()

    # Update capillary reservoir
    wc = wc + capillary_absorption

    # ET from capillary reservoir
    et3 = np.minimum(remaining_et, wc)
    et_total = et1 + et2 + et3

    # Update reservoirs after ET
    wc = wc - et3
    wg = wg - capillary_absorption

    # Test mode: check intermediate state
    if test_mode:
        if np.any(et_total - potential_et > tolerance):
            logger.warning("et_total > potential_et detected")
        if np.any(et3 < -tolerance):
            logger.warning("et3 < 0 detected")
        if np.any(wc < -tolerance):
            logger.warning("wc < 0 detected")
        if np.any(wg < -tolerance):
            logger.warning("wg < 0 detected")
        if np.any(capillary_absorption < -tolerance):
            logger.warning("capillary_absorption < 0 detected")

    # Percolation
    if capillary_rise_enabled and depth_to_water_table is not None and soil_depth is not None:
        wg_temp = np.minimum(wg, wg0)
        wg_deficit = wg0 - wg_temp  # Wg deficit
        percolation_normal = percolation_coeff * wg_temp
        percolation_1 = (wg_temp + (depth_to_water_table / soil_depth - 1) * wg0) / 2
        percolation = np.minimum(percolation_normal, percolation_1) * (depth_to_water_table >= 0) + np.maximum(
            (depth_to_water_table - wg_deficit) / 2, -wg_deficit
        ) * (depth_to_water_table < 0)
        seep_max = -percolation_coeff * wg0
        percolation = percolation * (percolation >= seep_max) + seep_max * (percolation < seep_max)

        # Cells outside aquifer: use standard percolation
        if no_aquifer_indices is not None:
            percolation[no_aquifer_indices] = percolation_coeff[no_aquifer_indices] * np.minimum(
                wg[no_aquifer_indices], wg0[no_aquifer_indices]
            )

        percolation = np.minimum(percolation, wg_temp)
    else:
        # Standard percolation
        percolation = percolation_coeff * np.minimum(wg, wg0)
        percolation = np.minimum(percolation, wg)

    wg = wg - percolation

    # Lateral Flow
    if capillary_rise_enabled and depth_to_water_table is not None:
        lateral_flow_out = lateral_flow_coeff * np.minimum(wg, wg0) * (depth_to_water_table > 0)

        # Cells outside aquifer: use standard lateral flow
        if no_aquifer_indices is not None:
            lateral_flow_out[no_aquifer_indices] = lateral_flow_coeff[no_aquifer_indices] * np.minimum(
                wg[no_aquifer_indices], wg0[no_aquifer_indices]
            )
            lateral_flow_out[no_aquifer_indices] = np.minimum(
                lateral_flow_out[no_aquifer_indices], wg[no_aquifer_indices]
            )
    else:
        # Standard lateral flow
        lateral_flow_out = lateral_flow_coeff * np.minimum(wg, wg0)
        lateral_flow_out = np.minimum(lateral_flow_out, wg)

    wg = wg - lateral_flow_out

    # Dunne runoff + return flow (saturation excess)
    dunne_runoff = (wg - wg0) * (wg > wg0)
    wg = wg - dunne_runoff

    # Test mode: check state
    if test_mode:
        if np.any(wc < -tolerance):
            logger.warning("wc < 0 after subsurface processes")
        if np.any(wg < -tolerance):
            logger.warning("wg < 0 after subsurface processes")

    # Capillary rise
    if capillary_rise_enabled and depth_to_water_table is not None:
        # Adjust depth to be from center of unsaturated soil column to water table
        zw_adjusted = depth_to_water_table.copy()
        zw_adjusted = zw_adjusted - (depth_to_water_table > soil_depth) * soil_depth / 2
        zw_adjusted = (
            zw_adjusted - ((depth_to_water_table > 0) & (depth_to_water_table <= soil_depth)) * depth_to_water_table / 2
        )

        # Soil saturation (only where Wg=0)
        saturation = wc / wtot0

        # Index of cells where capillary rise is active
        active_cells = (zw_adjusted > 0) & (wg == 0) & (capillary_multiplier == 1)

        # Calculate capillary rise
        capillary_flux = zeros.copy()
        if np.any(active_cells):
            capillary_flux[active_cells] = capillary_rise(
                saturation[active_cells],
                zw_adjusted[active_cells],
                capillary_conductivity[active_cells],
                bubbling_pressure[active_cells],
                capillary_param_a[active_cells],
                capillary_param_n[active_cells],
            )
        capillary_flux = np.minimum(capillary_flux, wc0 - wc)
        capillary_flux[capillary_flux < 0] = 0

        # Cells outside aquifer: no capillary rise
        if no_aquifer_indices is not None:
            capillary_flux[no_aquifer_indices] = 0

        wc = wc + capillary_flux
    else:
        capillary_flux = zeros.copy()

    # ========================================================================
    # SURFACE WATER ADJUSTMENT
    # ========================================================================

    # Surface runoff routing
    total_surface_runoff = horton_runoff + dunne_runoff + surface_runoff_channelized
    ws = ws + total_surface_runoff

    if alpsur is not None:
        # Use surface flow parameter directly
        surface_runoff_out = np.minimum(ws * alpsur, ws)
    else:
        # Use exponential decay (original MATLAB version - currently not used)
        logger.warning("surface_flow_exp parameter used but alpsur is None")
        surface_runoff_out = total_surface_runoff

    ws = ws - surface_runoff_out

    # Test mode: check mass balance
    if test_mode:
        water_out = wc + wg + wp + ws + et_total + percolation + surface_runoff_out + lateral_flow_out - capillary_flux
        water_balance = water_in - water_out
        water_balance_mean = np.mean(water_balance)
        if abs(water_balance_mean) > tolerance:
            logger.warning(f"Mass balance error: {water_balance_mean}")

    # Check for saturation excess in gravitational reservoir
    if np.any(wg > wg0 + tolerance):
        logger.warning("wg > wg0 detected at end of time step")

    # Return None for wp if it was not enabled
    if not wp_enabled:
        wp = None

    return (
        wc,
        wg,
        wp,
        ws,
        surface_runoff_out,
        lateral_flow_out,
        et_total,
        percolation,
        capillary_flux,
        wg_before_absorption,
    )
