"""GIS data readers for raster and vector files.

This module provides functions to read GIS data files (rasters and shapefiles)
using rasterio and geopandas, respectively.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger


def grid_to_matrix(gridname: str | Path) -> dict[str, Any]:
    """Read raster grid file and return data array with corner coordinates. This function
    replicates the behavior of MATLAB's function for GeoTIFF files.

    Reads raster files in GeoTIFF (.tif, .tiff)
    format. Returns the data array and corner coordinates adjusted to cell centers.

    Args:
        gridname: Path to the grid file (.tif, .tiff).

    Returns:
        Dictionary containing:
            - 'data': 2D numpy array with raster values (NaN for nodata)
            - 'xllcorner': X coordinate of lower-left corner (cell center)
            - 'yllcorner': Y coordinate of lower-left corner (cell center)
            - 'cellsize': Cell size in map units

    Raises:
        FileNotFoundError: If the grid file does not exist.
        ValueError: If the file format is not supported.
        RuntimeError: If there are errors reading the file.

    Notes:
        - Corner coordinates are adjusted to cell centers (0.5 * cellsize offset)
        - Data is flipped vertically to match MATLAB convention
        - Very small values (< -1e32) are converted to NaN

    Examples:
        >>> result = grid_to_matrix('elevation.tif')
        >>> data = result['data']
        >>> print(f"Shape: {data.shape}, Resolution: {result['cellsize']}m")
    """
    gridname = Path(gridname)
    suffix_lower = gridname.suffix.lower()

    logger.info(f"Reading grid file: {gridname}")

    # GeoTIFF format
    if suffix_lower in [".tif", ".tiff"]:
        if not gridname.exists():
            logger.error(f"File not found: {gridname}")
            raise FileNotFoundError(f"File not found: {gridname}")

        try:
            with rasterio.open(gridname) as src:
                # Read data and convert to float
                matgr = src.read(1).astype(float)

                # Get nodata value and convert to NaN
                nodata = src.nodata
                if nodata is not None:
                    matgr[matgr == nodata] = np.nan

                # Get spatial information
                transform = src.transform
                cellsize = transform[0]  # Assuming square pixels
                xllcorner = src.bounds.left
                yllcorner = src.bounds.bottom

                # Get CRS
                crs = src.crs

        except Exception as e:
            logger.error(f"Error reading GeoTIFF file {gridname}: {e}")
            raise RuntimeError(f"Error reading GeoTIFF file: {e}") from e

        # Flip vertically (to match MATLAB convention)
        matgr = np.flipud(matgr)

        # Filter nodata values (e.g., -3.4e38)
        matgr[matgr < -1e32] = np.nan

        # Adjust to cell center
        xllcorner += 0.5 * cellsize
        yllcorner += 0.5 * cellsize

        logger.success(f"GeoTIFF read: shape={matgr.shape}, cellsize={cellsize}")

        return {
            "data": matgr,
            "xllcorner": xllcorner,
            "yllcorner": yllcorner,
            "cellsize": cellsize,
            "crs": crs,
        }

    else:
        logger.error(f"Unsupported file format: {suffix_lower}")
        raise ValueError(f"Unsupported file format: {suffix_lower}. Must be .tif or .tiff")


def read_shapefile(
    filepath: str | Path,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Read a shapefile and return a GeoDataFrame.

    Args:
        filepath: Path to the shapefile (.shp).
        crs: Optional CRS to reproject the data. If None, keeps original CRS.

    Returns:
        GeoDataFrame containing the shapefile data.

    Raises:
        FileNotFoundError: If the shapefile does not exist.
        Exception: If the file cannot be read by geopandas.

    Examples:
        >>> river_network = read_shapefile("example/shp/Arno_river_network.shp")
        >>> reach_ids = river_network['REACH_ID']
    """
    filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        logger.error(f"Shapefile not found: {filepath}")
        raise FileNotFoundError(f"Shapefile not found: {filepath}")

    logger.info(f"Reading shapefile: {filepath}")

    try:
        # Read shapefile
        gdf = gpd.read_file(filepath)

        logger.debug(f"Shapefile loaded: {len(gdf)} features, CRS={gdf.crs}")

        # Reproject if requested
        if crs is not None:
            logger.debug(f"Reprojecting from {gdf.crs} to {crs}")
            gdf = gdf.to_crs(crs)

        logger.success(f"Successfully read shapefile: {len(gdf)} features, CRS={gdf.crs}")

        return gdf

    except Exception as e:
        logger.error(f"Failed to read shapefile {filepath}: {e}")
        raise RuntimeError(f"Failed to read shapefile {filepath}: {e}") from e
