"""Hillslope-reach mapping module for MOBIDIC.

This module provides functions to map hillslope cells to their corresponding
river reaches based on flow direction and river network geometry.
"""

import numpy as np
import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString
from pathlib import Path

from mobidic.preprocessing.grid_operations import convert_to_mobidic_notation
from mobidic.preprocessing.gis_reader import grid_to_matrix


def compute_hillslope_cells(
    network: gpd.GeoDataFrame,
    grid_path: str | Path,
    densify_step: float = 10.0,
) -> gpd.GeoDataFrame:
    """Compute hillslope cells for each reach in the river network.

    This function rasterizes each reach geometry onto the grid to identify
    which grid cells are directly occupied by the channel. These cells are
    stored as linear indices in the 'hillslope_cells' column.

    The densification uses iterative midpoint subdivision (matching MATLAB's
    densify.m) to ensure exact compatibility with the original implementation.

    Args:
        network: GeoDataFrame with processed river network (must contain geometry column)
        grid_path: Path to reference grid raster (used to get grid parameters)
        densify_step: Maximum distance [m] between points when densifying reach geometries (default: 10.0)

    Returns:
        GeoDataFrame with added 'hillslope_cells' column containing list of linear indices

    Examples:
        >>> from mobidic import process_river_network
        >>> network = process_river_network("river_network.shp")
        >>> flowdir_path = "flowdir.tif"
        >>> network = compute_hillslope_cells(network, flowdir_path)
    """
    logger.info(f"Computing hillslope cells for {len(network)} reaches")

    # Read reference grid
    grid_result = grid_to_matrix(grid_path)
    grid = grid_result["data"]
    xllcorner = grid_result["xllcorner"]
    yllcorner = grid_result["yllcorner"]
    cellsize = grid_result["cellsize"]
    nrows, ncols = grid.shape

    logger.debug(f"Grid parameters: xllcorner={xllcorner}, yllcorner={yllcorner}, cellsize={cellsize}")

    # Initialize column for storing hillslope cells
    hillslope_cells = []

    for idx in network.index:
        geom = network.loc[idx, "geometry"]

        if geom.is_empty:
            hillslope_cells.append([])
            continue

        # Densify the LineString geometry
        coords = _densify_linestring(geom, densify_step)

        if len(coords) == 0:
            hillslope_cells.append([])
            continue

        # Convert geographic coordinates to grid indices
        xx0 = coords[:, 0]  # x coordinates
        yy0 = coords[:, 1]  # y coordinates

        # Convert to grid indices
        col_indices = np.round((xx0 - xllcorner) / cellsize).astype(int)
        row_indices = np.round((yy0 - yllcorner) / cellsize).astype(int)

        # Filter out-of-bounds indices
        valid_mask = (row_indices >= 0) & (row_indices < nrows) & (col_indices >= 0) & (col_indices < ncols)
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]

        if len(row_indices) == 0:
            hillslope_cells.append([])
            continue

        # Convert to linear indices
        linear_indices = col_indices * nrows + row_indices

        # Get unique indices
        unique_indices = np.unique(linear_indices)

        hillslope_cells.append(unique_indices.tolist())

    # Add column to network
    network = network.copy()
    network["hillslope_cells"] = hillslope_cells

    total_cells = sum(len(cells) for cells in hillslope_cells)
    logger.success(f"Computed hillslope cells: {total_cells} total cells across {len(network)} reaches")

    return network


def map_hillslope_to_reach(
    network: gpd.GeoDataFrame,
    flowdir_path: str | Path,
    flow_dir_type: str = "Grass",
) -> np.ndarray:
    """Map each hillslope cell to its downstream river reach.

    For each grid cell, this function follows the flow direction path until
    it reaches a cell that belongs to a river reach (hillslope cell), then
    assigns the reach's mobidic_id to the original cell.

    This function translates MATLAB's hill2chan.m algorithm:
    - Builds a lookup table of reach cells from network.hillslope_cells
    - For each valid cell, follows flow direction until reaching a reach cell
    - Assigns the reach's mobidic_id to all cells along the path
    - Handles edge cases: out-of-bounds, invalid directions, loops

    Args:
        network: GeoDataFrame with river network (must have 'hillslope_cells' and 'mobidic_id' columns)
        flowdir_path: Path to flow direction raster file
        flow_dir_type: Flow direction notation ('Grass' for 1-8 or 'Arc' for power-of-2)

    Returns:
        2D array with reach assignment for each cell (mobidic_id or -9999 for unassigned)

    Raises:
        ValueError: If network doesn't have required columns or flow_dir_type is invalid

    Examples:
        >>> from mobidic import process_river_network, compute_hillslope_cells
        >>> network = process_river_network("river_network.shp")
        >>> network = compute_hillslope_cells(network, "flow_dir.tif")
        >>> reach_map = map_hillslope_to_reach(network, "flow_dir.tif")
    """
    logger.info("Mapping hillslope cells to river reaches")

    # Validate network has required columns
    if "hillslope_cells" not in network.columns:
        logger.error("Network must have 'hillslope_cells' column. Run compute_hillslope_cells() first.")
        raise ValueError("Network must have 'hillslope_cells' column")
    if "mobidic_id" not in network.columns:
        logger.error("Network must have 'mobidic_id' column")
        raise ValueError("Network must have 'mobidic_id' column")

    # Read flow direction grid
    flowdir_result = grid_to_matrix(flowdir_path)
    flowdir = flowdir_result["data"]

    # Transform flow direction from GRASS/Arc to MOBIDIC notation
    flowdir = convert_to_mobidic_notation(flowdir, from_notation=flow_dir_type)

    nrows, ncols = flowdir.shape

    # Initialize output array with NaN
    reach_map = np.full((nrows, ncols), np.nan, dtype=float)

    # Build lookup table: linear_index -> mobidic_id
    # This replaces MATLAB's ri/rv arrays with a dictionary for O(1) lookup
    logger.debug("Building reach cell lookup table")
    cell_to_reach = {}
    for idx in network.index:
        mobidic_id = network.loc[idx, "mobidic_id"]
        hillslope_cells = network.loc[idx, "hillslope_cells"]

        if hillslope_cells is None or len(hillslope_cells) == 0:
            continue

        for cell_idx in hillslope_cells:
            # Store first occurrence (matches MATLAB h(1) behavior)
            if cell_idx not in cell_to_reach:
                cell_to_reach[cell_idx] = mobidic_id

    logger.debug(f"Lookup table built: {len(cell_to_reach)} reach cells")

    # Direction offsets for MOBIDIC notation (1-8)
    di = np.array([-1, -1, -1, 0, 1, 1, 1, 0])  # row offset (matches i8)
    dj = np.array([-1, 0, 1, 1, 1, 0, -1, -1])  # column offset (matches j8)

    # Get valid cells (non-NaN flow direction)
    valid_cells = np.where(np.isfinite(flowdir))
    n_valid = len(valid_cells[0])

    logger.info(f"Processing {n_valid} valid cells")

    # Process each valid cell
    processed_count = 0
    unassigned_count = 0

    for k in range(n_valid):
        i0, j0 = valid_cells[0][k], valid_cells[1][k]

        # Skip if already assigned
        if not np.isnan(reach_map[i0, j0]):
            continue

        # Track path for this cell
        path_i = [i0]
        path_j = [j0]
        ic, jc = i0, j0

        # Follow flow direction until reaching a reach cell or error
        while True:
            # Convert current position to linear index
            kc = jc * nrows + ic

            # Check if current cell is a reach cell
            if kc in cell_to_reach:
                # Assign reach to original cell
                reach_map[i0, j0] = cell_to_reach[kc]
                processed_count += 1
                break

            # Check for invalid flow direction
            direction = flowdir[ic, jc]
            if not (1 <= direction <= 8) or np.isnan(direction):
                reach_map[i0, j0] = -9999
                unassigned_count += 1
                break

            # Move to next cell
            direction_idx = int(direction) - 1
            ic_next = ic + di[direction_idx]
            jc_next = jc + dj[direction_idx]

            # Check bounds
            if ic_next < 0 or ic_next >= nrows or jc_next < 0 or jc_next >= ncols:
                reach_map[i0, j0] = -9999
                unassigned_count += 1
                break

            # MATLAB loop detection: if find(sub2ind([n,m],i0,j0) == sub2ind([n,m],ic,jc))
            # Check if next position (ic_next, jc_next) is already in the path
            # This detects if we're revisiting any cell in the current path
            if any(ic_next == pi and jc_next == pj for pi, pj in zip(path_i, path_j)):
                reach_map[i0, j0] = -9999
                unassigned_count += 1
                break

            # Append to path and continue (MATLAB: i0=[i0 ic]; j0=[j0 jc];)
            path_i.append(ic_next)
            path_j.append(jc_next)
            ic, jc = ic_next, jc_next

    logger.success(
        f"Hillslope-to-reach mapping complete: {processed_count} cells assigned, "
        f"{unassigned_count} cells unassigned (out-of-bounds, loops, or invalid flow)"
    )

    return reach_map


def _densify_linestring(geom: LineString, step: float) -> np.ndarray:
    """Densify a LineString geometry using iterative midpoint subdivision.

    This function iteratively subdivides line segments by adding midpoints
    until all segments are shorter than the specified step size.

    Args:
        geom: LineString geometry
        step: Maximum distance between consecutive points [m]

    Returns:
        Array of shape (n, 2) with densified coordinates
    """
    if geom.is_empty or geom.length == 0:
        return np.array([])

    # Get original coordinates
    coords = np.array(geom.coords)
    xx = coords[:, 0]
    yy = coords[:, 1]
    n = len(xx)

    # Compute segment lengths between consecutive vertices
    ss = np.zeros(n)
    if n > 1:
        ss[1:] = np.sqrt((xx[1:] - xx[:-1]) ** 2 + (yy[1:] - yy[:-1]) ** 2)

    # Remove duplicate points (segments with length < epsilon)
    eps = 1e-10
    kg_mask = np.ones(n, dtype=bool)
    if n > 1:
        kg_mask[1:] = ss[1:] >= eps

    kg = np.where(kg_mask)[0]

    if len(kg) == 0:
        return np.array([])

    # Cumulative distance along original vertices (excluding duplicates)
    ss_cum = np.cumsum(ss[kg])

    # Start with distances at original vertices
    sd = ss_cum.copy()

    if len(kg) > 1:
        # Iteratively subdivide by interpolating at midpoints until max segment <= step
        # MATLAB: sd = interp1(1:length(sd), sd, 1:0.5:length(sd))
        # This doubles the number of points each iteration by adding midpoints
        while np.max(np.diff(sd)) > step:
            indices_old = np.arange(1, len(sd) + 1)  # 1, 2, 3, ..., n
            indices_new = np.arange(1, len(sd) + 0.5, 0.5)  # 1, 1.5, 2, 2.5, ..., n
            sd = np.interp(indices_new, indices_old, sd)

        # Interpolate x,y coordinates at the densified distances
        xd = np.interp(sd, ss_cum, xx[kg])
        yd = np.interp(sd, ss_cum, yy[kg])
    else:
        # Only one vertex after removing duplicates
        xd = xx[kg]
        yd = yy[kg]

    return np.column_stack([xd, yd])
