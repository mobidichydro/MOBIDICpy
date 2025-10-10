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
        >>> from mobidic import process_river_network, read_raster
        >>> network = process_river_network("river_network.shp")
        >>> flowdir_path = "flowdir.tif"
        >>> network = compute_hillslope_cells(network, flowdir_path)
    """
    logger.info(f"Computing hillslope cells for {len(network)} reaches")

    # Read reference grid
    grid, xllcorner, yllcorner, cellsize = grid_to_matrix(grid_path)
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
    flowdir_path: np.ndarray,
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
        >>> network = process_river_network("river_network.shp")
        >>> network = compute_hillslope_cells(network, "flow_dir.tif")
        >>> reach_map = map_hillslope_to_reach(network, "flow_dir.tif")
    """
    logger.info("Mapping hillslope cells to river reaches")

    flowdir, xllcorner, yllcorner, cellsize = grid_to_matrix(flowdir_path)

    # Transform flow direction from GRASS/Arc to MOBIDIC notation
    flowdir = convert_to_mobidic_notation(flowdir, from_notation=flow_dir_type)

    # TODO: Implement hill2chan.m functionality


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
