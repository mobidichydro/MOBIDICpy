"""Reservoir routing for MOBIDIC.

This module implements reservoir water balance and routing, including:
- Volume-stage-discharge relationships
- Time-varying regulation curves
- Sub-stepping for numerical stability
- Interaction with soil water balance in reservoir basin

Translated from MATLAB: reservoir_routing_include.m
"""

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ReservoirState:
    """State variables for a single reservoir.

    Attributes:
        volume: Current reservoir volume [m³]
        stage: Current water stage/level [m]
        inflow: Current inflow from upstream reaches [m³/s]
        outflow: Current outflow/release [m³/s]
        withdrawal: Current withdrawal for water use [m³/s] (not implemented yet)
    """

    volume: float
    stage: float
    inflow: float = 0.0
    outflow: float = 0.0
    withdrawal: float = 0.0


def _interpolate_stage_from_volume(
    stage_storage_curve: pd.DataFrame,
    volume: float,
) -> float:
    """Interpolate stage from volume using stage-storage curve.

    Uses cubic spline interpolation with extrapolation.

    Args:
        stage_storage_curve: DataFrame with 'stage_m' and 'volume_m3' columns
        volume: Target volume [m³]

    Returns:
        Interpolated stage [m]
    """
    stages = stage_storage_curve["stage_m"].values
    volumes = stage_storage_curve["volume_m3"].values

    # Sort by volume if not already sorted
    if not np.all(volumes[:-1] <= volumes[1:]):
        sort_idx = np.argsort(volumes)
        volumes = volumes[sort_idx]
        stages = stages[sort_idx]

    # Interpolate with extrapolation
    # Use linear interpolation (matching MATLAB's 'linear' mode)
    stage = np.interp(volume, volumes, stages)

    return float(stage)


def _interpolate_discharge_from_stage(
    stage_values: np.ndarray,
    discharge_values: np.ndarray,
    stage: float,
) -> float:
    """Interpolate discharge from stage using regulation curve.

    Uses linear interpolation with extrapolation.

    Args:
        stage_values: Array of stage values [m] (can contain NaN)
        discharge_values: Array of discharge values [m³/s] (can contain NaN)
        stage: Current stage [m]

    Returns:
        Interpolated discharge [m³/s]
    """
    # Filter out NaN values (matching MATLAB's isfinite filter)
    valid = np.isfinite(stage_values) & np.isfinite(discharge_values)
    stage_clean = stage_values[valid]
    discharge_clean = discharge_values[valid]

    if len(stage_clean) == 0:
        logger.warning("No valid stage-discharge points available, returning 0")
        return 0.0

    # Sort by stage if not already sorted
    if not np.all(stage_clean[:-1] <= stage_clean[1:]):
        sort_idx = np.argsort(stage_clean)
        stage_clean = stage_clean[sort_idx]
        discharge_clean = discharge_clean[sort_idx]

    # Interpolate with extrapolation (matching MATLAB's 'linear','extrap')
    # numpy.interp doesn't extrapolate by default, so we need to handle it
    if stage < stage_clean[0]:
        # Extrapolate linearly below
        if len(stage_clean) >= 2:
            dstage = stage_clean[1] - stage_clean[0]
            if abs(dstage) > 1e-10:  # Avoid division by zero
                slope = (discharge_clean[1] - discharge_clean[0]) / dstage
                discharge = discharge_clean[0] + slope * (stage - stage_clean[0])
            else:
                # Stages are identical, use first discharge
                discharge = discharge_clean[0]
        else:
            discharge = discharge_clean[0]
    elif stage > stage_clean[-1]:
        # Extrapolate linearly above
        if len(stage_clean) >= 2:
            dstage = stage_clean[-1] - stage_clean[-2]
            if abs(dstage) > 1e-10:  # Avoid division by zero
                slope = (discharge_clean[-1] - discharge_clean[-2]) / dstage
                discharge = discharge_clean[-1] + slope * (stage - stage_clean[-1])
            else:
                # Stages are identical, use last discharge
                discharge = discharge_clean[-1]
        else:
            discharge = discharge_clean[-1]
    else:
        # Interpolate within range
        discharge = np.interp(stage, stage_clean, discharge_clean)

    # Ensure discharge is non-negative
    discharge = max(0.0, discharge)

    return float(discharge)


def _calculate_substeps(
    stage_discharge_h: np.ndarray,
    stage_discharge_q: np.ndarray,
    stage_storage_curve: pd.DataFrame,
    dt: float,
    base_substeps: int = 1,
) -> int:
    """Calculate number of sub-steps needed for numerical stability.

    The sub-stepping is determined by the minimum time constant of the
    regulation curve (dV/dQ), where dV is the volume change between consecutive
    stage points and dQ is the discharge change.

    Matching MATLAB lines 23-48.

    Args:
        stage_discharge_h: Stage values for current period [m]
        stage_discharge_q: Discharge values for current period [m³/s]
        stage_storage_curve: Stage-storage DataFrame (not used in current impl)
        dt: Time step [s]
        base_substeps: Base number of sub-steps (default: 1)

    Returns:
        Number of sub-steps to use
    """
    # Get consecutive pairs of stage and discharge
    H1 = stage_discharge_h[:-1]
    H2 = stage_discharge_h[1:]
    Q2 = stage_discharge_q[1:]

    # Filter valid pairs (matching MATLAB's kfin = isfinite(H1+H2+Q2))
    valid = np.isfinite(H1) & np.isfinite(H2) & np.isfinite(Q2) & (Q2 > 0)

    if not valid.any():
        return base_substeps

    H1_valid = H1[valid]
    H2_valid = H2[valid]
    Q2_valid = Q2[valid]

    # Calculate volume differences using stage-storage curve
    # MATLAB: dV = interp1(reserv(i).h, reserv(i).Vol, H2, 'cubic') - interp1(reserv(i).h, reserv(i).Vol, H1, 'cubic')
    volumes = stage_storage_curve["volume_m3"].values
    stages = stage_storage_curve["stage_m"].values

    # Interpolate volumes at H1 and H2
    V1 = np.interp(H1_valid, stages, volumes)
    V2 = np.interp(H2_valid, stages, volumes)
    dV = V2 - V1

    # Calculate minimum time constant (matching MATLAB lines 44-45)
    dtmin = dV / Q2_valid
    dtmin = np.nanmin(dtmin)

    # Determine number of substeps (matching MATLAB lines 46-48)
    nsbgo = base_substeps
    while dt / nsbgo > dtmin:
        nsbgo = nsbgo * 2

    return nsbgo


def reservoir_routing(
    reservoirs_data: list,
    reservoir_states: list[ReservoirState],
    reach_discharge: np.ndarray,
    surface_runoff: np.ndarray,
    lateral_flow: np.ndarray,
    soil_wg: np.ndarray,
    soil_wg0: np.ndarray,
    current_time: datetime,
    dt: float,
    cell_area: float,
    base_substeps: int = 1,
) -> tuple[list[ReservoirState], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route water through reservoirs.

    This function performs reservoir water balance calculations including:
    - Volume update based on inflows and outflows
    - Stage calculation from volume
    - Discharge calculation from stage using regulation curves
    - Sub-stepping for numerical stability
    - Interaction with soil water balance in reservoir basins

    Matching MATLAB: reservoir_routing_include.m (lines 1-225)

    Args:
        reservoirs_data: List of Reservoir objects from preprocessing
        reservoir_states: List of current ReservoirState objects
        reach_discharge: Array of reach discharges [m³/s]
        surface_runoff: 2D grid of surface runoff rates [m/s]
        lateral_flow: 2D grid of lateral flow rates [m/s]
        soil_wg: 2D grid of gravitational water content [m]
        soil_wg0: 2D grid of gravitational water capacity [m]
        current_time: Current simulation time
        dt: Time step [s]
        cell_area: Grid cell area [m²]
        base_substeps: Base number of sub-steps (default: 1)

    Returns:
        Tuple of:
            - Updated reservoir states
            - Modified reach discharge array
            - Modified surface runoff grid
            - Modified lateral flow grid
            - Modified soil_wg grid
    """
    # Create copies to avoid modifying inputs
    reach_discharge = reach_discharge.copy()
    surface_runoff = surface_runoff.copy()
    lateral_flow = lateral_flow.copy()
    soil_wg = soil_wg.copy()

    # Convert current_time to pandas Timestamp for comparison
    current_timestamp = pd.Timestamp(current_time)

    # Loop through all reservoirs (matching MATLAB line 16)
    for i, reservoir in enumerate(reservoirs_data):
        # Check if reservoir is active at current time
        if reservoir.date_start is not None and current_timestamp < reservoir.date_start:
            # Reservoir not yet active
            reservoir_states[i].outflow = 0.0
            reservoir_states[i].inflow = 0.0
            reservoir_states[i].volume = 0.0
            reservoir_states[i].stage = 0.0
            continue

        # Find current regulation period (matching MATLAB lines 19-21)
        # period_times is a dict: {"000": "2020-01-01T00:00:00", "001": "2020-06-01T00:00:00", ...}
        period_times = reservoir.period_times
        if period_times is None or len(period_times) == 0:
            logger.warning(f"Reservoir {reservoir.id} has no regulation periods, skipping")
            reservoir_states[i].outflow = 0.0
            reservoir_states[i].inflow = 0.0
            continue

        # Convert period_times to list of (index, timestamp) tuples
        period_list = []
        for key, time_str in period_times.items():
            period_idx = int(key)
            period_time = pd.Timestamp(time_str)
            if period_time <= current_timestamp:
                period_list.append((period_idx, period_time))

        if len(period_list) == 0:
            # No active period yet
            reservoir_states[i].outflow = 0.0
            reservoir_states[i].inflow = 0.0
            continue

        # Get the most recent period (matching MATLAB: iT = iT(end))
        period_list.sort(key=lambda x: x[1])  # Sort by time
        iT = period_list[-1][0]  # Get the index of the most recent period

        # Get regulation curves for current period (matching MATLAB lines 27-38)
        period_key = f"{iT:03d}"
        lawH = np.array(reservoir.stage_discharge_h[period_key])
        lawQ = np.array(reservoir.stage_discharge_q[period_key])

        # Calculate number of sub-steps (matching MATLAB lines 23-48)
        nsbgo = _calculate_substeps(
            lawH,
            lawQ,
            reservoir.stage_storage_curve,
            dt,
            base_substeps,
        )

        if nsbgo > 1:
            logger.debug(f"Reservoir {reservoir.id}: using {nsbgo} sub-steps")

        # Get reservoir basin pixels (linear indices)
        ibac = reservoir.basin_pixels
        if ibac is None or len(ibac) == 0:
            logger.warning(f"Reservoir {reservoir.id} has no basin pixels, skipping")
            continue

        # Convert linear indices to 2D indices (Fortran order)
        nrows, ncols = surface_runoff.shape
        ibac_row = ibac // ncols
        ibac_col = ibac % ncols

        # Initialize accumulated outflow (will be averaged over sub-steps)
        total_outflow = 0.0

        # Loop over sub-steps (matching MATLAB line 78)
        for ir in range(nsbgo):
            # Calculate stage from volume (matching MATLAB lines 80-105)
            # Using stage-storage curve interpolation
            stage = _interpolate_stage_from_volume(
                reservoir.stage_storage_curve,
                reservoir_states[i].volume,
            )
            reservoir_states[i].stage = stage

            # Calculate outflow from stage using regulation curve (matching MATLAB lines 107-120)
            discharge = _interpolate_discharge_from_stage(lawH, lawQ, stage)

            # Calculate inflow from upstream reaches (matching MATLAB line 132)
            inlet_reaches = reservoir.inlet_reaches
            if inlet_reaches is not None and len(inlet_reaches) > 0:
                inflow = np.sum(reach_discharge[inlet_reaches])
            else:
                inflow = 0.0

            reservoir_states[i].inflow = inflow

            # Calculate lateral inflows from basin cells (matching MATLAB line 134)
            # Sum of (pir + pid) over basin cells, converted to discharge [m³/s]
            pir_basin = surface_runoff[ibac_row, ibac_col]
            pid_basin = lateral_flow[ibac_row, ibac_col]
            lateral_inflow = np.sum(pir_basin + pid_basin) * cell_area

            # Calculate soil water change in basin (matching MATLAB line 135)
            # DV1 = -sum(Wg0(ibac) - Wg(ibac)) * cellsize^2
            wg_basin = soil_wg[ibac_row, ibac_col]
            wg0_basin = soil_wg0[ibac_row, ibac_col]
            soil_water_change = -np.sum(wg0_basin - wg_basin) * cell_area

            # Update volume (matching MATLAB lines 134-136)
            # DV = (Qin - Qout + lateral_inflow) * dt/nsbgo
            DV = (inflow - discharge + lateral_inflow) * dt / nsbgo
            new_volume = reservoir_states[i].volume + DV + soil_water_change

            # Handle negative volumes (matching MATLAB lines 137-148)
            if new_volume < 0:
                # Try to reduce outflow to keep volume non-negative
                if new_volume + discharge * dt / nsbgo > 0:
                    # Reduce discharge proportionally (matching MATLAB line 139)
                    discharge = discharge + new_volume / (dt / nsbgo)
                    new_volume = 0.0
                    # Refill soil to capacity (matching MATLAB line 141)
                    soil_wg[ibac_row, ibac_col] = wg0_basin
                else:
                    # Not enough water even with zero discharge
                    # Adjust volume without changing discharge (matching MATLAB line 143)
                    new_volume = new_volume + discharge * dt / nsbgo - soil_water_change
                    discharge = 0.0
            else:
                # Volume is positive, refill soil to capacity (matching MATLAB line 147)
                soil_wg[ibac_row, ibac_col] = wg0_basin

            reservoir_states[i].volume = new_volume

            # Accumulate outflow for averaging (matching MATLAB lines 162-165)
            total_outflow += discharge

        # Average outflow over sub-steps
        reservoir_states[i].outflow = total_outflow / nsbgo

        # Check for invalid outflow (matching MATLAB lines 175-177)
        if not np.isfinite(reservoir_states[i].outflow) or reservoir_states[i].outflow < 0:
            logger.error(
                f"Invalid outflow for reservoir {reservoir.id}: {reservoir_states[i].outflow}, setting to 0"
            )
            reservoir_states[i].outflow = 0.0

        # Zero out surface runoff and lateral flow in reservoir basin (matching MATLAB lines 178-179)
        surface_runoff[ibac_row, ibac_col] = 0.0
        lateral_flow[ibac_row, ibac_col] = 0.0

        # Zero out discharge of upstream reaches (matching MATLAB lines 180-184)
        if inlet_reaches is not None and len(inlet_reaches) > 0:
            reach_discharge[inlet_reaches] = 0.0

    return reservoir_states, reach_discharge, surface_runoff, lateral_flow, soil_wg
