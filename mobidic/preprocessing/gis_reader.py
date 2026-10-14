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

from mobidic.utils.crs import crs_equals, get_epsg_code, parse_crs


class GridAlignmentError(ValueError):
    """Raised when an input file does not match the DTM grid or CRS, or has a geographic CRS."""


# Values commonly used as nodata, checked when a raster declares no nodata value
_NODATA_TYPICAL = (-9999.0, -99999.0, -32768.0, -32767.0)


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
                else:
                    # Without a declared nodata value, common sentinels are read as valid data
                    for sentinel in _NODATA_TYPICAL:
                        n_sentinel = int(np.sum(matgr == sentinel))
                        if n_sentinel > 0:
                            logger.warning(
                                f"{gridname} declares no nodata value but has {n_sentinel} cells equal to "
                                f"{sentinel:g}: they are read as valid data. If they are nodata, set the "
                                "nodata value of the file."
                            )

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

        logger.info(f"GeoTIFF read: shape={matgr.shape}, cellsize={cellsize}")

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


def _describe_crs(crs: Any) -> str:
    """Return a short, human-readable label for a CRS (EPSG code when available)."""
    epsg = get_epsg_code(crs)
    if epsg is not None:
        return f"EPSG:{epsg}"
    parsed = parse_crs(crs)
    return parsed.name if parsed is not None else str(crs)


def check_crs(crs: Any, source: str | Path) -> bool:
    """Check the CRS of an input file.

    Warns when the file has no CRS and rejects geographic CRSs, since MOBIDIC
    assumes coordinates, lengths and cell sizes in metres.

    Args:
        crs: CRS of the file (rasterio/pyproj CRS, WKT, EPSG code) or None.
        source: Path of the file, used in messages.

    Returns:
        True if the file has a CRS, False otherwise.

    Raises:
        GridAlignmentError: If the CRS is geographic.
    """
    parsed = parse_crs(crs)
    if parsed is None:
        logger.warning(f"{source} has no CRS: assuming it is in the same CRS as the DTM")
        return False

    if parsed.is_geographic:
        msg = (
            f"{source} has a geographic CRS ({_describe_crs(crs)}). "
            "MOBIDIC requires a projected CRS in metres: reproject the file."
        )
        logger.error(msg)
        raise GridAlignmentError(msg)

    return True


def check_raster_alignment(
    raster: dict[str, Any],
    reference: dict[str, Any],
    raster_path: str | Path,
    reference_path: str | Path,
) -> None:
    """Check that a raster has the same shape, origin, cell size and CRS as the reference (DTM).

    No resampling or reprojection is performed: mismatching rasters must be fixed by the user.

    Args:
        raster: Result of grid_to_matrix() for the raster to check.
        reference: Result of grid_to_matrix() for the reference raster (DTM).
        raster_path: Path of the raster to check, used in messages.
        reference_path: Path of the reference raster, used in messages.

    Raises:
        GridAlignmentError: If the raster has a geographic CRS or does not match the reference grid.
    """
    has_crs = check_crs(raster["crs"], raster_path)

    errors = []
    shape = raster["data"].shape
    ref_shape = reference["data"].shape
    if shape != ref_shape:
        errors.append(f"shape {shape} != {ref_shape} (rows, cols)")

    cellsize = reference["cellsize"]
    if not np.isclose(raster["cellsize"], cellsize, rtol=1e-6):
        errors.append(f"cell size {raster['cellsize']} != {cellsize}")

    # Origins are compared at cell centers, with a tolerance for floating point noise
    tol = 1e-3 * cellsize
    if not (
        np.isclose(raster["xllcorner"], reference["xllcorner"], rtol=0, atol=tol)
        and np.isclose(raster["yllcorner"], reference["yllcorner"], rtol=0, atol=tol)
    ):
        errors.append(
            f"lower-left cell center ({raster['xllcorner']}, {raster['yllcorner']}) "
            f"!= ({reference['xllcorner']}, {reference['yllcorner']})"
        )

    ref_crs = reference["crs"]
    if has_crs and parse_crs(ref_crs) is not None and not crs_equals(raster["crs"], ref_crs):
        errors.append(f"CRS {_describe_crs(raster['crs'])} != {_describe_crs(ref_crs)}")

    if errors:
        msg = (
            f"Raster {raster_path} does not match the DTM grid ({reference_path}):\n  - "
            + "\n  - ".join(errors)
            + "\nAll rasters must share the DTM's extent, resolution and CRS."
        )
        logger.error(msg)
        raise GridAlignmentError(msg)


def check_vector_crs(gdf: gpd.GeoDataFrame, source: str | Path, reference_crs: Any = None) -> None:
    """Check the CRS of a vector dataset against the reference (DTM) CRS.

    Warns when the dataset has no CRS. No reprojection is performed.

    Args:
        gdf: Vector dataset to check.
        source: Path of the file, used in messages.
        reference_crs: CRS of the DTM. If None, only the geographic CRS check is performed.

    Raises:
        GridAlignmentError: If the CRS is geographic or differs from the reference CRS.
    """
    if not check_crs(gdf.crs, source):
        return

    if parse_crs(reference_crs) is not None and not crs_equals(gdf.crs, reference_crs):
        msg = (
            f"{source} has CRS {_describe_crs(gdf.crs)}, but the DTM has CRS {_describe_crs(reference_crs)}. "
            "Reproject the file to the DTM CRS."
        )
        logger.error(msg)
        raise GridAlignmentError(msg)
