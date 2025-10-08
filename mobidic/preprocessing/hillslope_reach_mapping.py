"""Hillslope-reach mapping module for MOBIDIC.

This module provides functions to map hillslope cells to their corresponding
river reaches based on flow direction and river network geometry.
"""

import numpy as np
import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString

from mobidic.preprocessing.grid_operations import convert_to_mobidic_notation


def compute_hillslope_cells(
    network: gpd.GeoDataFrame,
    transform: tuple,
    shape: tuple,
    densify_step: float = 10.0,
) -> gpd.GeoDataFrame:
    """Compute hillslope cells for each reach in the river network.

    This function rasterizes each reach geometry onto the grid to identify
    which grid cells are directly occupied by the channel. These cells are
    stored as linear indices in the 'hillslope_cells' column.

    Args:
        network: GeoDataFrame with processed river network (must contain geometry column)
        transform: Affine transform from rasterio (contains origin and resolution)
        shape: Tuple (nrows, ncols) of the grid
        densify_step: Distance [m] between points when densifying reach geometries (default: 10.0)

    Returns:
        GeoDataFrame with added 'hillslope_cells' column containing list of linear indices

    Examples:
        >>> from mobidic import process_river_network, read_raster
        >>> network = process_river_network("river_network.shp")
        >>> raster = read_raster("dtm.tif")
        >>> network = compute_hillslope_cells(network, raster['transform'], raster['data'].shape)
    """
    logger.info(f"Computing hillslope cells for {len(network)} reaches")

    # Extract grid parameters from transform
    # Transform is an Affine object: (a, b, c, d, e, f) representing
    # x = a * col + b * row + c
    # y = d * col + e * row + f
    # For standard north-up raster: a=cellsize, e=-cellsize, b=d=0, c=xllcorner, f=ytop
    xllcorner = transform[2]  # x-coordinate of left edge
    ytop = transform[5]  # y-coordinate of top edge
    cellsize_x = transform[0]  # cell width
    cellsize_y = abs(transform[4])  # cell height (absolute value)

    nrows, ncols = shape

    # Calculate yllcorner (bottom-left y-coordinate) to match MATLAB convention
    # yllcorner = ytop - nrows * cellsize_y
    yllcorner = ytop + nrows * transform[4]  # transform[4] is negative

    # MATLAB's grid2mat.m adds 0.5*cellsize to shift from edge to center coordinates
    # (see grid2mat.m lines 29-30). This ensures consistent cell assignment with MATLAB.
    xllcorner = xllcorner + 0.5 * cellsize_x
    yllcorner = yllcorner + 0.5 * cellsize_y

    logger.debug(f"Grid parameters: xllcorner={xllcorner}, yllcorner={yllcorner}, cellsize={cellsize_x}")

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
        # MATLAB code: cc=round((xx0-xllcorner)./cellsize)+1; rr=round((yy0-yllcorner)./cellsize)+1;
        # Note: MATLAB uses 1-based indexing, Python uses 0-based
        xx0 = coords[:, 0]  # x coordinates
        yy0 = coords[:, 1]  # y coordinates

        # Convert to grid indices (0-based, matching MATLAB's 1-based minus 1)
        col_indices = np.round((xx0 - xllcorner) / cellsize_x).astype(int)
        row_indices = np.round((yy0 - yllcorner) / cellsize_y).astype(int)

        # Filter out-of-bounds indices
        valid_mask = (row_indices >= 0) & (row_indices < nrows) & (col_indices >= 0) & (col_indices < ncols)
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]

        if len(row_indices) == 0:
            hillslope_cells.append([])
            continue

        # Convert to linear indices (column-major order to match MATLAB)
        # MATLAB: linear_idx = col * nrows + row + 1 (1-based)
        # Python: linear_idx = col * nrows + row (0-based)
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
    flow_dir: np.ndarray,
    network: gpd.GeoDataFrame,
    flow_dir_type: str = "Grass",
) -> np.ndarray:
    """Map each hillslope cell to its downstream river reach.

    For each grid cell, this function follows the flow direction path until
    it reaches a cell that belongs to a river reach (hillslope cell), then
    assigns the reach's mobidic_id to the original cell.

    Args:
        flow_dir: Flow direction raster (2D array)
        network: GeoDataFrame with river network (must have 'hillslope_cells' and 'mobidic_id' columns)
        flow_dir_type: Flow direction notation ('Grass' for 1-8 or 'Arc' for power-of-2)

    Returns:
        2D array with reach assignment for each cell (mobidic_id or -9999 for unassigned)

    Raises:
        ValueError: If network doesn't have required columns or flow_dir_type is invalid

    Examples:
        >>> from mobidic import read_raster, process_river_network, compute_hillslope_cells
        >>> flow_dir_data = read_raster("flow_dir.tif")
        >>> network = process_river_network("river_network.shp")
        >>> network = compute_hillslope_cells(network, flow_dir_data['transform'], flow_dir_data['data'].shape)
        >>> reach_map = map_hillslope_to_reach(flow_dir_data['data'], network)
    """
    logger.info("Mapping hillslope cells to river reaches")

    # Validate inputs
    if "hillslope_cells" not in network.columns:
        raise ValueError("Network must have 'hillslope_cells' column. Run compute_hillslope_cells() first.")

    if "mobidic_id" not in network.columns:
        raise ValueError("Network must have 'mobidic_id' column")

    if flow_dir_type not in ["Grass", "Arc"]:
        raise ValueError(f"Invalid flow_dir_type: {flow_dir_type}. Must be 'Grass' or 'Arc'")

    # Transform flow direction from GRASS/Arc to MOBIDIC notation
    flow_dir = convert_to_mobidic_notation(flow_dir, from_notation=flow_dir_type)

    # Row and column offsets for MOBIDIC directions (1-8)
    # IMPORTANT: flow_dir has already been transformed to MOBIDIC notation above
    #
    # MOBIDIC is GRASS rotated 180 degrees (from buildgis_mysql_include.m):
    # GRASS → MOBIDIC transformation: 1→5, 2→6, 3→7, 4→8, 5→1, 6→2, 7→3, 8→4
    #
    # GRASS notation:          MOBIDIC notation:
    #   7 6 5                    3 2 1
    #   8   4                    4   8
    #   1 2 3                    5 6 7
    #
    # MATLAB hill2chan.m uses these offsets with the ALREADY-TRANSFORMED direction values:
    #   i8=[-1 -1 -1 0 1 1 1 0]; j8=[-1 0 1 1 1 0 -1 -1];
    #
    # MATLAB arrays are flipped vertically (flipud in grid2mat.m), so row increases UPWARD.
    # Python arrays have row increasing DOWNWARD. Flow direction values match between systems,
    # but we need to NEGATE row offsets to account for opposite row direction:
    i8 = np.array([+1, +1, +1, 0, -1, -1, -1, 0])  # row offsets (NEGATED from MATLAB)
    j8 = np.array([-1, 0, +1, +1, +1, 0, -1, -1])  # column offsets (same as MATLAB)

    nrows, ncols = flow_dir.shape

    # Create a mapping from linear indices to reach IDs
    # Build arrays of all reach cells and their corresponding reach IDs
    reach_indices = []
    reach_ids = []

    for idx in network.index:
        cells = network.loc[idx, "hillslope_cells"]
        reach_id = network.loc[idx, "mobidic_id"]

        if isinstance(cells, (list, np.ndarray)) and len(cells) > 0:
            reach_indices.extend(cells)
            reach_ids.extend([reach_id] * len(cells))

    # Convert to numpy arrays for faster lookup
    reach_indices = np.array(reach_indices, dtype=int)
    reach_ids = np.array(reach_ids, dtype=int)

    # Initialize output array with NaN
    ch = np.full(flow_dir.shape, np.nan, dtype=float)

    # First, assign reach cells to themselves (even if flow_dir is NaN at those locations)
    logger.debug(f"Assigning {len(reach_indices)} reach cells to their reaches")
    for i in range(len(reach_indices)):
        lin_idx = reach_indices[i]
        reach_id = reach_ids[i]
        # Convert linear index (column-major) to 2D indices (row-major)
        row_idx = lin_idx % nrows
        col_idx = lin_idx // nrows
        if 0 <= row_idx < nrows and 0 <= col_idx < ncols:
            ch[row_idx, col_idx] = reach_id

    # Get all valid (non-NaN) cells in flow direction grid that aren't already assigned
    valid_cells = np.isfinite(flow_dir) & np.isnan(ch)
    valid_linear_indices_2d = np.where(valid_cells)

    logger.info(f"Processing {len(valid_linear_indices_2d[0])} valid cells (excluding reach cells)")

    # Process each valid cell
    for idx in range(len(valid_linear_indices_2d[0])):
        i0 = valid_linear_indices_2d[0][idx]
        j0 = valid_linear_indices_2d[1][idx]

        # Skip if already assigned
        if np.isfinite(ch[i0, j0]):
            continue

        # Trace flow path until reaching a channel cell or a cell that has already been assigned
        ic = i0
        jc = j0
        flow_path = [(i0, j0)]  # Track flow path for batch assignment

        while True:
            # Check if current cell is already assigned
            if np.isfinite(ch[ic, jc]):
                # Current cell already has an assignment, use it for the entire flow path
                assigned_reach = ch[ic, jc]
                # Assign all cells in the flow path to this reach
                for (pi, pj) in flow_path:
                    ch[pi, pj] = assigned_reach
                break

            # Compute linear index (column-major order to match MATLAB)
            kc = jc * nrows + ic

            # Check if current cell is a channel cell
            h = np.where(reach_indices == kc)[0]
            if len(h) > 0:
                # Found a channel cell, assign reach ID to entire flow path
                assigned_reach = reach_ids[h[0]]
                for (pi, pj) in flow_path:
                    ch[pi, pj] = assigned_reach
                break

            # Get flow direction
            fd = flow_dir[ic, jc]

            # Check for invalid flow direction
            if fd < 1 or fd > 8 or np.isnan(fd):
                # Assign -9999 to entire flow path
                for (pi, pj) in flow_path:
                    ch[pi, pj] = -9999
                break

            # Move to next cell following flow direction
            direction_idx = int(fd) - 1  # Convert to 0-based index
            ic_next = ic + i8[direction_idx]
            jc_next = jc + j8[direction_idx]

            # Check bounds
            if ic_next < 0 or ic_next >= nrows or jc_next < 0 or jc_next >= ncols:
                # Assign -9999 to entire flow path
                for (pi, pj) in flow_path:
                    ch[pi, pj] = -9999
                break

            # Check for loops (cell appears twice in flow path)
            if (ic_next, jc_next) in flow_path:
                # Assign -9999 to entire flow path
                for (pi, pj) in flow_path:
                    ch[pi, pj] = -9999
                break

            # Move to next cell
            ic = ic_next
            jc = jc_next
            flow_path.append((ic, jc))

    # Statistics
    assigned_cells = np.sum(ch >= 0)
    unassigned_cells = np.sum(ch == -9999)
    nodata_cells = np.sum(np.isnan(ch))

    logger.success(
        f"Hillslope-to-reach mapping complete: "
        f"{assigned_cells} assigned, {unassigned_cells} unassigned, {nodata_cells} nodata"
    )

    return ch



def _densify_linestring(geom: LineString, step: float) -> np.ndarray:
    """Densify a LineString geometry by adding points at regular intervals.

    Args:
        geom: LineString geometry
        step: Distance between points [m]

    Returns:
        Array of shape (n, 2) with densified coordinates
    """
    if geom.is_empty or geom.length == 0:
        return np.array([])

    coords = []
    total_length = geom.length
    num_points = max(2, int(np.ceil(total_length / step)))

    for i in range(num_points):
        distance = (i / (num_points - 1)) * total_length
        point = geom.interpolate(distance)
        coords.append([point.x, point.y])

    return np.array(coords)
