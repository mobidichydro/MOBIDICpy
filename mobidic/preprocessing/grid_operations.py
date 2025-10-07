"""Grid operations for spatial data processing.

This module provides functions for coarsening resolution of raster grids,
and flow direction conversion between different notation systems.
"""

from typing import Literal

import numpy as np
from loguru import logger


def degrade_raster(
    data: np.ndarray,
    factor: int,
    min_valid_fraction: float = 0.125,
) -> np.ndarray:
    """Coarsen raster resolution by aggregating cells.

    Aggregates a fine-resolution grid into a coarser grid by averaging values
    within each block of size factor x factor. Blocks with too few valid cells
    (less than min_valid_fraction) are marked as NaN.

    Args:
        data: 2D numpy array with raster values (NaN for nodata).
        factor: Degradation factor (e.g., 2 means 2x2 blocks -> 1 cell).
        min_valid_fraction: Minimum fraction of valid cells required in each block
            to compute the mean. Default is 0.125 (1/8 of cells).

    Returns:
        Degraded 2D numpy array with shape (floor(nr/factor), floor(nc/factor)).

    Examples:
        >>> import numpy as np
        >>> data = np.random.rand(100, 100)
        >>> degraded = degrade_raster(data, factor=2)
        >>> degraded.shape
        (50, 50)
    """
    if factor < 1:
        logger.error(f"Degradation factor must be >= 1, got {factor}")
        raise ValueError(f"Degradation factor must be >= 1, got {factor}")

    if factor == 1:
        logger.debug("Degradation factor is 1, returning original data")
        return data.copy()

    nr, nc = data.shape
    nrv = nr // factor
    ncv = nc // factor

    logger.info(f"Degrading raster from {data.shape} to ({nrv}, {ncv}) with factor {factor}")

    # Initialize output array
    degraded = np.full((nrv, ncv), np.nan, dtype=float)

    # Minimum number of valid cells required
    min_valid_cells = int(factor * factor * min_valid_fraction)

    # Process each block
    for i in range(nrv):
        iv_start = i * factor
        iv_end = (i + 1) * factor

        for j in range(ncv):
            jv_start = j * factor
            jv_end = (j + 1) * factor

            # Extract block
            block = data[iv_start:iv_end, jv_start:jv_end]

            # Count valid (non-NaN) cells
            n_valid = np.sum(np.isfinite(block))

            # Compute mean if enough valid cells
            if n_valid >= min_valid_cells:
                degraded[i, j] = np.nanmean(block)

    logger.success(f"Raster degradation complete: {np.sum(np.isfinite(degraded))} valid cells")

    return degraded


def degrade_flow_direction(
    flow_dir: np.ndarray,
    flow_acc: np.ndarray,
    factor: int,
    min_valid_fraction: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Coarsen flow direction and flow accumulation grids.

    Aggregates fine-resolution flow direction and flow accumulation grids into
    coarser grids. For each coarse cell, identifies the fine cell with maximum
    flow accumulation and determines the coarse flow direction based on where
    that cell drains to.

    Args:
        flow_dir: 2D numpy array with flow directions (1-8 notation, NaN for nodata).
        flow_acc: 2D numpy array with flow accumulation values (NaN for nodata).
        factor: Degradation factor (e.g., 2 means 2x2 blocks -> 1 cell).
        min_valid_fraction: Minimum fraction of valid cells required in each block.
            Default is 0.125 (1/8 of cells).

    Returns:
        Tuple of (degraded_flow_dir, degraded_flow_acc) as 2D numpy arrays.
        Flow accumulation is normalized by factor **2 to account for cell size change.

    Notes:
        This function assumes flow_dir uses Grass 1-8 notation:
            1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE, 8=E

    Examples:
        >>> flow_dir = np.array([[2, 2], [2, 2]])  # All cells flow north
        >>> flow_acc = np.array([[1, 1], [3, 4]])  # Different accumulation
        >>> deg_dir, deg_acc = degrade_flow_direction(flow_dir, flow_acc, factor=2)
    """
    if factor < 1:
        logger.error(f"Degradation factor must be >= 1, got {factor}")
        raise ValueError(f"Degradation factor must be >= 1, got {factor}")

    if factor == 1:
        logger.debug("Degradation factor is 1, returning original data")
        return flow_dir.copy(), flow_acc.copy()

    if flow_dir.shape != flow_acc.shape:
        logger.error("Flow direction and accumulation grids must have the same shape")
        raise ValueError("Flow direction and accumulation grids must have the same shape")

    nr, nc = flow_dir.shape
    nrv = nr // factor
    ncv = nc // factor

    logger.info(f"Degrading flow direction/accumulation from {flow_dir.shape} to ({nrv}, {ncv})")

    # Initialize output arrays
    flow_dir_degraded = np.full((nrv, ncv), np.nan, dtype=float)
    flow_acc_degraded = np.full((nrv, ncv), np.nan, dtype=float)

    # Direction offsets for Grass 1-8 notation to MOBIDIC notation
    # 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE, 8=E
    di = np.array([-1, -1, -1, 0, 1, 1, 1, 0])  # row offset
    dj = np.array([-1, 0, 1, 1, 1, 0, -1, -1])  # column offset

    # Minimum number of valid cells required
    min_valid_cells = int(factor * factor * min_valid_fraction)

    # Process each coarse block
    for i in range(nrv):
        iv_start = i * factor
        iv_end = (i + 1) * factor

        for j in range(ncv):
            jv_start = j * factor
            jv_end = (j + 1) * factor

            # Extract blocks
            block_dir = flow_dir[iv_start:iv_end, jv_start:jv_end]
            block_acc = flow_acc[iv_start:iv_end, jv_start:jv_end]

            # Count valid cells
            n_valid = np.sum(np.isfinite(block_dir))

            # Skip if not enough valid cells
            if n_valid < min_valid_cells:
                continue

            # Set NaN in accumulation where direction is NaN
            block_acc_masked = block_acc.copy()
            block_acc_masked[np.isnan(block_dir)] = np.nan

            # Find cell with maximum flow accumulation
            if not np.any(np.isfinite(block_acc_masked)):
                continue

            max_idx = np.nanargmax(block_acc_masked)
            max_i, max_j = np.unravel_index(max_idx, block_acc_masked.shape)

            # Store normalized flow accumulation
            flow_acc_degraded[i, j] = np.nanmax(block_acc_masked) / (factor * factor)

            # Get flow direction of the cell with max accumulation
            direction = int(block_dir[max_i, max_j])

            if not (1 <= direction <= 8):
                # Invalid direction, try to find a valid neighbor
                flow_dir_degraded[i, j] = -999
                continue

            # Calculate which coarse cell this fine cell drains to
            target_i = max_i + di[direction - 1]
            target_j = max_j + dj[direction - 1]

            # Determine coarse cell index offset
            coarse_di = target_i // factor
            coarse_dj = target_j // factor

            # Calculate target coarse cell coordinates
            target_coarse_i = i + coarse_di
            target_coarse_j = j + coarse_dj

            # Check if target is within bounds
            if not (0 <= target_coarse_i < nrv and 0 <= target_coarse_j < ncv):
                flow_dir_degraded[i, j] = -999
                continue

            # Map coarse cell offset to direction (1-8)
            # coarse_di, coarse_dj -> direction
            offset_to_dir = {
                (-1, -1): 1,  # NE (IP=-1, JP=-1 -> -11 -> case 1)
                (-1, 0): 2,  # N  (IP=-1, JP=0  -> -10 -> case 2)
                (-1, 1): 3,  # NW (IP=-1, JP=1  -> -9  -> case 3)
                (0, 1): 4,  # W  (IP=0,  JP=1  -> 1   -> case 4)
                (1, 1): 5,  # SW (IP=1,  JP=1  -> 11  -> case 5)
                (1, 0): 6,  # S  (IP=1,  JP=0  -> 10  -> case 6)
                (1, -1): 7,  # SE (IP=1,  JP=-1 -> 9   -> case 7)
                (0, -1): 8,  # E  (IP=0,  JP=-1 -> -1  -> case 8)
            }

            flow_dir_degraded[i, j] = offset_to_dir.get((coarse_di, coarse_dj), -999)

    # Fix invalid directions by finding valid neighbors
    invalid_mask = flow_dir_degraded == -999
    if np.any(invalid_mask):
        logger.debug(f"Fixing {np.sum(invalid_mask)} invalid flow directions")

        invalid_indices = np.argwhere(invalid_mask)
        for idx in invalid_indices:
            i, j = idx

            # Try each direction to find a valid neighbor
            for d in range(8):
                ii = i + di[d]
                jj = j + dj[d]

                if 0 <= ii < nrv and 0 <= jj < ncv:
                    if flow_dir_degraded[ii, jj] > 0:
                        flow_dir_degraded[i, j] = d + 1
                        break

    logger.success(
        f"Flow direction degradation complete: "
        f"{np.sum(np.isfinite(flow_dir_degraded))} valid cells, "
        f"{np.sum(flow_dir_degraded == -999)} invalid directions"
    )

    return flow_dir_degraded, flow_acc_degraded


def convert_flow_direction(
    flow_dir: np.ndarray,
    from_notation: Literal["Grass", "Arc"],
    to_notation: Literal["Grass", "Arc"],
) -> np.ndarray:
    """Convert flow direction between Grass (1-8) and Arc (power-of-2) notations.

    Args:
        flow_dir: 2D numpy array with flow directions (NaN for nodata).
        from_notation: Source notation ('Grass' for 1-8 or 'Arc' for power-of-2).
        to_notation: Target notation ('Grass' for 1-8 or 'Arc' for power-of-2).

    Returns:
        Converted flow direction array with the same shape as input.

    Notes:
        Notation systems:
            - Grass (1-8): 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE, 8=E
            - Arc (power-of-2): 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE

    Examples:
        >>> flow_dir_grass = np.array([[1, 2], [3, 4]])  # Grass notation
        >>> flow_dir_arc = convert_flow_direction(flow_dir_grass, 'Grass', 'Arc')
        >>> flow_dir_arc
        array([[128,  64],
               [ 32,  16]])
    """
    if from_notation == to_notation:
        logger.debug(f"Same notation ({from_notation}), returning original data")
        return flow_dir.copy()

    logger.info(f"Converting flow direction from {from_notation} to {to_notation} notation")

    # Mapping between Grass (1-8) and Arc (power-of-2) notations
    # Direction: NE, N, NW, W, SW, S, SE, E
    grass_to_arc = {
        1: 128,  # NE
        2: 64,  # N
        3: 32,  # NW
        4: 16,  # W
        5: 8,  # SW
        6: 4,  # S
        7: 2,  # SE
        8: 1,  # E
    }

    arc_to_grass = {v: k for k, v in grass_to_arc.items()}

    # Create output array
    converted = flow_dir.copy()

    # Get conversion mapping
    if from_notation == "Grass" and to_notation == "Arc":
        mapping = grass_to_arc
    elif from_notation == "Arc" and to_notation == "Grass":
        mapping = arc_to_grass
    else:
        raise ValueError(f"Invalid notation combination: {from_notation} -> {to_notation}")

    # Convert valid (non-NaN) cells
    valid_mask = np.isfinite(flow_dir)

    # Apply mapping
    for old_val, new_val in mapping.items():
        converted[valid_mask & (flow_dir == old_val)] = new_val

    logger.success(f"Flow direction conversion complete: {np.sum(valid_mask)} cells converted")

    return converted
