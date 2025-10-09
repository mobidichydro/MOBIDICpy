"""Grid operations for spatial data processing.

This module provides functions for coarsening resolution of raster grids,
flow direction conversion between different notation systems, and reading
grid files in various formats.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
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


def convert_to_mobidic_notation(
    flow_dir: np.ndarray,
    from_notation: Literal["Grass", "Arc"] = "Grass",
) -> np.ndarray:
    """Convert flow direction from Grass or Arc notation to MOBIDIC notation.

    MOBIDIC uses a transformed version of the Grass notation with a 180-degree rotation.
    This transformation is applied in MATLAB's buildgis_mysql_include.m:
        AI=[1 2 3 4 5 6 7 8]; MD=[5 6 7 8 1 2 3 4];

    Args:
        flow_dir: 2D numpy array with flow directions (NaN for nodata).
        from_notation: Source notation ('Grass' for 1-8 or 'Arc' for power-of-2). Default is 'Grass'.

    Returns:
        Flow direction array converted to MOBIDIC notation (1-8).

    Notes:
        Transformation mappings:
            - GRASS -> MOBIDIC: 1->5, 2->6, 3->7, 4->8, 5->1, 6->2, 7->3, 8->4
            - Arc values are first converted to Grass, then to MOBIDIC

        Notation comparison:
            GRASS:              MOBIDIC:
              7 6 5               3 2 1
              8   4               4   8
              1 2 3               5 6 7

        MOBIDIC directions:
            1: up-right      (row -1, col +1)
            2: up            (row -1, col  0)
            3: up-left       (row -1, col -1)
            4: left          (row  0, col -1)
            5: down-left     (row +1, col -1)
            6: down          (row +1, col  0)
            7: down-right    (row +1, col +1)
            8: right         (row  0, col +1)

    Examples:
        >>> flow_dir_grass = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 1]])
        >>> flow_dir_mobidic = convert_to_mobidic_notation(flow_dir_grass, 'Grass')
        >>> flow_dir_mobidic
        array([[5, 6, 7],
               [8, 1, 2],
               [3, 4, 5]])
    """
    logger.info(f"Converting flow direction from {from_notation} to MOBIDIC notation")

    # GRASS to MOBIDIC transformation (from buildgis_mysql_include.m)
    grass_to_mobidic = {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4}

    # Arc to GRASS mapping
    arc_to_grass = {1: 8, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7}

    # Create output array
    converted = flow_dir.copy()

    if from_notation == "Arc":
        # Arc -> Grass -> MOBIDIC (two-step conversion)
        logger.debug("Converting Arc -> Grass -> MOBIDIC")
        for arc_val, grass_val in arc_to_grass.items():
            mask = flow_dir == arc_val
            converted[mask] = grass_to_mobidic[grass_val]
    elif from_notation == "Grass":
        # Grass -> MOBIDIC (direct conversion)
        logger.debug("Converting Grass -> MOBIDIC")
        for grass_val, mobidic_val in grass_to_mobidic.items():
            mask = flow_dir == grass_val
            converted[mask] = mobidic_val
    else:
        raise ValueError(f"Invalid from_notation: {from_notation}. Must be 'Grass' or 'Arc'")

    valid_cells = np.sum(np.isfinite(converted))
    logger.success(f"Flow direction conversion to MOBIDIC complete: {valid_cells} cells converted")

    return converted

