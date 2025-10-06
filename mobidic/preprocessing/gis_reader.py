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
from rasterio.errors import RasterioIOError


def read_raster(
    filepath: str | Path,
    band: int = 1,
    nodata_value: float | None = None,
) -> dict[str, Any]:
    """Read a raster file and return data with metadata.

    Args:
        filepath: Path to the raster file (e.g., GeoTIFF).
        band: Band number to read (default: 1).
        nodata_value: Optional nodata value to use. If None, uses the file's nodata value.

    Returns:
        Dictionary containing:
            - 'data': 2D numpy array with raster values
            - 'transform': Affine transformation matrix
            - 'crs': Coordinate reference system
            - 'nodata': Nodata value
            - 'bounds': Bounding box (left, bottom, right, top)
            - 'shape': Shape of the raster (rows, cols)
            - 'resolution': Pixel resolution (x_res, y_res)

    Raises:
        FileNotFoundError: If the raster file does not exist.
        RasterioIOError: If the file cannot be read by rasterio.
        ValueError: If the band number is invalid.

    Examples:
        >>> raster = read_raster("example/raster/dtm.tif")
        >>> elevation_data = raster['data']
        >>> crs = raster['crs']
    """
    filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        logger.error(f"Raster file not found: {filepath}")
        raise FileNotFoundError(f"Raster file not found: {filepath}")

    logger.info(f"Reading raster file: {filepath}")

    try:
        with rasterio.open(filepath) as src:
            # Validate band number
            if band < 1 or band > src.count:
                logger.error(f"Invalid band number {band}. File has {src.count} band(s).")
                raise ValueError(f"Invalid band number {band}. File has {src.count} band(s).")

            # Read raster data
            logger.debug(f"Reading band {band} from {filepath}")
            data = src.read(band)

            # Get nodata value
            file_nodata = src.nodata
            nodata = nodata_value if nodata_value is not None else file_nodata

            # Replace nodata values with NaN
            if nodata is not None:
                data = data.astype(float)
                data[data == nodata] = np.nan

            # Extract metadata
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            shape = (src.height, src.width)
            resolution = (src.res[0], src.res[1])

            logger.success(f"Successfully read raster: shape={shape}, crs={crs}, resolution={resolution}")

            return {
                "data": data,
                "transform": transform,
                "crs": crs,
                "nodata": nodata,
                "bounds": bounds,
                "shape": shape,
                "resolution": resolution,
            }

    except RasterioIOError as e:
        logger.error(f"Failed to read raster file {filepath}: {e}")
        raise RasterioIOError(f"Failed to read raster file {filepath}: {e}") from e


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
