"""Spatial interpolation utilities for meteorological data.

This module provides functions to interpolate meteorological station data
onto the model grid using various spatial interpolation methods.

Translated from MATLAB:
    - pluviomap.m -> precipitation_interpolation()
    - tempermap.m -> temperature_interpolation()
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata
from loguru import logger


def precipitation_interpolation(
    station_x: NDArray[np.float64],
    station_y: NDArray[np.float64],
    station_values: NDArray[np.float64],
    dtm: NDArray[np.float64],
    xllcorner: float,
    yllcorner: float,
    resolution: float,
) -> NDArray[np.float64]:
    """
    Interpolate precipitation data to grid using nearest neighbor method.

    This function replicates the MATLAB pluviomap.m logic, which uses
    nearest neighbor interpolation for precipitation data.

    Args:
        station_x: X coordinates of stations (e.g., UTM east) [m]
        station_y: Y coordinates of stations (e.g., UTM north) [m]
        station_values: Precipitation values at stations (can contain NaN) [mm or m]
        dtm: Digital Terrain Model (elevation grid) [m]
        xllcorner: X coordinate of lower-left corner [m]
        yllcorner: Y coordinate of lower-left corner [m]
        resolution: Grid cell size [m]

    Returns:
        2D array of interpolated precipitation with same shape as dtm.
        Grid cells outside basin (NaN in dtm) will be NaN.

    Notes:
        - Follows MATLAB pluviomap.m logic exactly
        - Uses nearest neighbor when 3+ stations available
        - Uses distance comparison for 2 stations
        - Returns constant value for 1 station
        - Returns NaN grid for 0 valid stations

    Examples:
        >>> # Interpolate precipitation from 3 stations
        >>> dtm = np.random.rand(100, 100) * 1000  # 100x100 DEM
        >>> station_x = np.array([1000, 5000, 9000])
        >>> station_y = np.array([1000, 5000, 9000])
        >>> station_values = np.array([10.0, 15.0, 12.0])
        >>> precip = precipitation_interpolation(
        ...     station_x, station_y, station_values, dtm, 0, 0, 100
        ... )
    """
    nrows, ncols = dtm.shape

    # Filter out stations with NaN values
    valid_mask = np.isfinite(station_values)
    n_valid = np.sum(valid_mask)

    logger.debug(f"Precipitation interpolation: {n_valid} valid stations out of {len(station_values)}")

    # Case 0: No valid stations
    if n_valid == 0:
        logger.warning("No valid stations for precipitation interpolation, returning NaN grid")
        return np.full_like(dtm, np.nan)

    # Filter to valid stations only
    st_x = station_x[valid_mask]
    st_y = station_y[valid_mask]
    st_val = station_values[valid_mask]

    # Case 1: Single station - use constant value
    if n_valid == 1:
        logger.debug(f"Single station: using constant value {st_val[0]:.2f}")
        result = np.full_like(dtm, st_val[0])
        result[np.isnan(dtm)] = np.nan
        return result

    # Convert station coordinates to grid indices
    st_j = np.round((st_x - xllcorner) / resolution).astype(int)  # column (x direction)
    st_i = np.round((st_y - yllcorner) / resolution).astype(int)  # row (y direction)

    # Create grid indices for interpolation
    jj, ii = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # Case 2: Two stations - use distance comparison
    if n_valid == 2:
        logger.debug("Two stations: using distance comparison")
        result = np.full_like(dtm, st_val[0])
        d1 = (jj - st_j[0]) ** 2 + (ii - st_i[0]) ** 2
        d2 = (jj - st_j[1]) ** 2 + (ii - st_i[1]) ** 2
        result[d2 < d1] = st_val[1]
        result[np.isnan(dtm)] = np.nan
        return result

    # Case 3: Three or more stations - use nearest neighbor interpolation
    logger.debug("Multiple stations: using nearest neighbor interpolation")

    # Use scipy's griddata with nearest neighbor method
    # Note: griddata expects (x, y) as (row, col) for the points
    result = griddata((st_i, st_j), st_val, (ii, jj), method="nearest")

    # Mask out cells outside basin
    result[np.isnan(dtm)] = np.nan

    logger.debug(
        f"Precipitation interpolation completed. Value range: [{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]"
    )

    return result


def temperature_interpolation(
    station_x: NDArray[np.float64],
    station_y: NDArray[np.float64],
    station_elevation: NDArray[np.float64],
    station_values: NDArray[np.float64],
    dtm: NDArray[np.float64],
    xllcorner: float,
    yllcorner: float,
    resolution: float,
    weights_matrix: NDArray[np.float64] | None = None,
    apply_elevation_correction: bool = True,
    power: float = 2.0,
) -> NDArray[np.float64]:
    """
    Interpolate temperature (or other variables) using IDW with elevation correction.

    This function replicates the MATLAB tempermap.m logic, which uses
    Inverse Distance Weighting (IDW) with optional elevation-based correction.

    Args:
        station_x: X coordinates of stations (e.g., UTM east) [m]
        station_y: Y coordinates of stations (e.g., UTM north) [m]
        station_elevation: Elevation of stations [m a.s.l.]
        station_values: Values at stations (temperature, humidity, etc.)
        dtm: Digital Terrain Model (elevation grid) [m a.s.l.]
        xllcorner: X coordinate of lower-left corner [m]
        yllcorner: Y coordinate of lower-left corner [m]
        resolution: Grid cell size [m]
        weights_matrix: Pre-computed weights matrix (3D: nrows x ncols x n_stations).
            If None, weights will be computed. [optional]
        apply_elevation_correction: Apply linear regression for elevation correction
        power: Power parameter for IDW (default: 2.0)

    Returns:
        2D array of interpolated values with same shape as dtm.

    Notes:
        - Follows MATLAB tempermap.m logic exactly
        - Applies elevation correction using linear regression if apply_elevation_correction=True
        - Uses IDW with specified power parameter
        - For stations inside grid, uses DEM elevation; otherwise uses station elevation

    Examples:
        >>> # Interpolate temperature from stations with elevation correction
        >>> dtm = np.random.rand(100, 100) * 1000  # 100x100 DEM
        >>> station_x = np.array([1000, 5000, 9000])
        >>> station_y = np.array([1000, 5000, 9000])
        >>> station_elev = np.array([100, 500, 300])
        >>> station_temp = np.array([20.0, 15.0, 18.0])  # °C
        >>> temp = temperature_interpolation(
        ...     station_x, station_y, station_elev, station_temp,
        ...     dtm, 0, 0, 100
        ... )
    """
    nrows, ncols = dtm.shape

    # Convert station coordinates to grid indices
    st_j = np.round((station_x - xllcorner) / resolution).astype(int)  # column
    st_i = np.round((station_y - yllcorner) / resolution).astype(int)  # row

    # Find stations inside the grid
    inside_mask = (st_i >= 0) & (st_i < nrows) & (st_j >= 0) & (st_j < ncols)
    st_indices_inside = np.where(inside_mask)[0]

    # Get station elevations from DEM where possible
    st_zz = np.full(len(station_elevation), np.nan)
    for idx in st_indices_inside:
        st_zz[idx] = dtm[st_i[idx], st_j[idx]]

    # Use station's own elevation if DEM value not available
    st_zz[np.isnan(st_zz)] = station_elevation[np.isnan(st_zz)]

    # Filter to stations with valid values and elevations
    valid_mask = np.isfinite(station_values) & np.isfinite(st_zz)
    n_valid = np.sum(valid_mask)

    logger.debug(f"Temperature interpolation: {n_valid} valid stations with elevation")

    if n_valid == 0:
        logger.warning("No valid stations for temperature interpolation, returning NaN grid")
        return np.full_like(dtm, np.nan)

    # Extract valid stations
    k_ok = np.where(valid_mask)[0]
    st_val_ok = station_values[k_ok]
    st_zz_ok = st_zz[k_ok]

    # Elevation correction: fit linear regression if sufficient stations
    if apply_elevation_correction and n_valid > 3:
        # Linear regression: value = p[0] * elevation + p[1]
        p = np.polyfit(st_zz_ok, st_val_ok, 1)
        logger.debug(f"Elevation correction: slope={p[0]:.6f}, intercept={p[1]:.2f}")
    else:
        # No elevation correction
        p = np.array([0.0, 0.0])

    # Remove elevation trend from station values
    st_val_corr = station_values - p[0] * st_zz - p[1]

    # Initialize result grids
    result = np.zeros((nrows, ncols))
    weights_sum = np.zeros((nrows, ncols))

    # Compute or use pre-computed weights
    if weights_matrix is not None:
        # Use pre-computed weights
        for i in range(n_valid):
            station_idx = k_ok[i]
            result += st_val_corr[station_idx] * weights_matrix[:, :, station_idx]
            weights_sum += weights_matrix[:, :, station_idx]
    else:
        # Compute weights on-the-fly using IDW
        jj, ii = np.meshgrid(np.arange(ncols), np.arange(nrows))

        for i in range(n_valid):
            station_idx = k_ok[i]

            # Distance from each grid cell to station
            dx = jj - st_j[station_idx]
            dy = ii - st_i[station_idx]
            dist_squared = dx**2 + dy**2 + 0.01  # Add small value to avoid division by zero

            # Inverse distance weights
            if power == 2.0:
                w = 1.0 / dist_squared
            else:
                w = 1.0 / (dist_squared ** (power / 2.0))

            # Accumulate weighted values
            result += st_val_corr[station_idx] * w
            weights_sum += w

    # Normalize by sum of weights
    result = result / weights_sum

    # Add back elevation trend
    result = result + p[0] * dtm + p[1]

    # Mask out cells outside basin
    result[np.isnan(dtm)] = np.nan

    logger.debug(
        f"Temperature interpolation completed. Value range: [{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]"
    )

    return result


def create_grid_coordinates(
    xllcorner: float,
    yllcorner: float,
    resolution: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Create 2D coordinate arrays for grid cells.

    Args:
        xllcorner: X coordinate of lower-left corner [m]
        yllcorner: Y coordinate of lower-left corner [m]
        resolution: (x_resolution, y_resolution) [m]
        shape: (nrows, ncols)

    Returns:
        Tuple of (grid_x, grid_y) where each is a 2D array of shape (nrows, ncols)

    Examples:
        >>> # Create coordinate arrays for a 10x10 grid with 100m resolution
        >>> grid_x, grid_y = create_grid_coordinates(0, 0, (100, 100), (10, 10))
    """
    nrows, ncols = shape
    res_x, res_y = resolution

    # Create 1D coordinate arrays (cell centers)
    x_coords = xllcorner + np.arange(ncols) * res_x
    y_coords = yllcorner + np.arange(nrows) * res_y

    # Create 2D meshgrid
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    return grid_x, grid_y
