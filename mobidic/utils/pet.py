"""Potential evapotranspiration (PET) calculation for MOBIDIC.

This module provides PET calculation matching the MATLAB implementation.
For Phase 1 (no energy balance), MOBIDIC uses constant 1 mm/day PET.

Translated from MATLAB: mobidic_sid.m (line 332)
"""

import numpy as np
from numpy.typing import NDArray
from loguru import logger


def calculate_pet(
    grid_shape: tuple[int, int],
    dt: float,
    pet_rate_mm_day: float = 1.0,
) -> NDArray[np.float64]:
    """
    Calculate potential evapotranspiration using constant rate.

    This function replicates the MATLAB approach for PET when energy balance
    is not active. MATLAB uses: etp = Mones/(1000*3600*24) which equals
    constant 1 mm/day.

    Args:
        grid_shape: Shape of grid (nrows, ncols)
        dt: Time step duration [s]
        pet_rate_mm_day: PET rate [mm/day] (default: 1.0, matching MATLAB)

    Returns:
        Potential evapotranspiration [m] over time step dt

    Notes:
        - MATLAB default: etp = 1/(1000*3600*24) m/s = 1 mm/day
        - When energy balance is active, PET comes from forenergybal.m
        - For Phase 1 (no energy balance), uses constant rate

    References:
        MATLAB: mobidic_sid.m line 332:
        etp = Mones/(1000*3600*24);  % = 1 mm/day constant

    Examples:
        >>> # Calculate PET for 100x100 grid, 900s time step, 1 mm/day rate
        >>> pet = calculate_pet((100, 100), 900)
        >>> # Result: 1 mm/day * (900s / 86400s) = 0.0104 mm = 1.04e-5 m
    """
    # Convert from mm/day to m/timestep
    # pet_rate_mm_day [mm/day] -> m/s -> m/timestep
    pet_rate_m_s = pet_rate_mm_day / (1000.0 * 86400.0)  # mm/day to m/s
    pet_m = pet_rate_m_s * dt  # m/s to m over dt

    # Create constant PET grid
    pet = np.full(grid_shape, pet_m, dtype=np.float64)

    logger.debug(f"Constant PET: {pet_rate_mm_day:.2f} mm/day, {pet[0, 0] * 1000:.4f} mm over {dt / 3600:.2f} hours")

    return pet
