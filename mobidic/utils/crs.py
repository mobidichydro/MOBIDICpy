"""CRS utilities for CF-compliant NetCDF encoding.

This module provides functions to convert CRS objects to CF-compliant
grid mapping attributes for NetCDF files, and to compare CRS objects.
"""

import re
from typing import Any
from pyproj import CRS
from loguru import logger


def crs_to_cf_attrs(crs_input: Any) -> dict[str, Any]:
    """Convert a CRS to CF-compliant grid mapping attributes.

    This function generates a dictionary of attributes suitable for a NetCDF
    grid mapping variable, following CF Conventions (CF-1.12+).

    The function:
    1. Extracts the correct grid_mapping_name (e.g., 'transverse_mercator')
    2. Includes all projection-specific scalar attributes
    3. Adds the crs_wkt attribute with OGC WKT1 format (for broad compatibility)
    4. Adds epsg_code for human readability (non-standard but common)

    Args:
        crs_input: A CRS object (pyproj.CRS, rasterio.crs.CRS, or any input
            accepted by pyproj.CRS.from_user_input), or a CRS string/code.

    Returns:
        Dictionary of CF-compliant grid mapping attributes including:
        - grid_mapping_name: Valid CF projection name
        - Projection-specific parameters (e.g., longitude_of_central_meridian)
        - crs_wkt: Full OGC WKT1 Well-Known Text representation
        - epsg_code: EPSG code string if available (e.g., "EPSG:3003")

    Examples:
        >>> from pyproj import CRS
        >>> crs = CRS.from_epsg(3003)
        >>> attrs = crs_to_cf_attrs(crs)
        >>> attrs['grid_mapping_name']
        'transverse_mercator'
        >>> attrs['epsg_code']
        'EPSG:3003'
        >>> 'crs_wkt' in attrs
        True

    Notes:
        - If the CRS cannot be converted to CF format, falls back to
          latitude_longitude with a warning.
        - The crs_wkt uses WKT1 (GDAL) format for broad compatibility with
          older software and GIS tools.
        - epsg_code is non-standard but widely used by tools like ADAGUC.
    """
    # Handle None or empty input
    if crs_input is None or (isinstance(crs_input, str) and not crs_input.strip()):
        logger.warning("Empty or None CRS provided, using default latitude_longitude")
        return {
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": "",
        }

    try:
        # Convert to pyproj CRS if needed
        if isinstance(crs_input, CRS):
            crs = crs_input
        else:
            crs = CRS.from_user_input(crs_input)

        # Get CF-compliant attributes using pyproj's to_cf() method
        cf_attrs = crs.to_cf()

        # Get proper WKT string (WKT1 GDAL format for broad compatibility)
        wkt_string = crs.to_wkt(version="WKT1_GDAL")

        # Add WKT attribute
        cf_attrs["crs_wkt"] = wkt_string

        # Add EPSG code if available (non-standard but widely used)
        epsg_code = crs.to_epsg()
        if epsg_code is not None:
            cf_attrs["epsg_code"] = f"EPSG:{epsg_code}"

        logger.debug(f"CRS converted to CF: grid_mapping_name={cf_attrs.get('grid_mapping_name', 'unknown')}")

        return cf_attrs

    except Exception as e:
        logger.warning(f"Failed to convert CRS to CF format: {e}. Using fallback.")

        # Fallback: try to at least get the WKT
        try:
            if not isinstance(crs_input, CRS):
                crs = CRS.from_user_input(crs_input)
            else:
                crs = crs_input
            wkt_string = crs.to_wkt(version="WKT1_GDAL")
        except Exception:
            wkt_string = str(crs_input) if crs_input else ""

        return {
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": wkt_string,
        }


def _extract_epsg_from_wkt(wkt_string: str) -> int | None:
    """Extract EPSG code from WKT string by parsing AUTHORITY tag.

    This is a fallback for when pyproj's to_epsg() returns None, which can
    happen when the WKT includes TOWGS84 parameters or other modifications
    that make it not exactly match the official EPSG definition.

    Args:
        wkt_string: A WKT1 or WKT2 string.

    Returns:
        EPSG code as integer, or None if not found.
    """
    if not wkt_string or not isinstance(wkt_string, str):
        return None

    # WKT1 format: AUTHORITY["EPSG","3003"] at the end of PROJCS/GEOGCS
    # Find the AUTHORITY tag right before the final closing brackets
    match = re.search(r'AUTHORITY\["EPSG",\s*"(\d+)"\]\s*\]\s*$', wkt_string.strip())
    if match:
        return int(match.group(1))

    # WKT2 format: ID["EPSG",3003] at the end
    match = re.search(r'ID\["EPSG",\s*(\d+)\]\s*\]\s*$', wkt_string.strip())
    if match:
        return int(match.group(1))

    return None


def get_epsg_code(crs_input: Any) -> int | None:
    """Extract EPSG code from a CRS input.

    Handles pyproj.CRS, rasterio.crs.CRS, WKT strings, EPSG codes,
    PROJ strings, and authority codes.

    When pyproj's to_epsg() returns None (e.g., for WKT strings with TOWGS84
    parameters that don't exactly match the official EPSG definition), this
    function falls back to parsing the AUTHORITY/ID tag directly from the
    WKT string.

    Args:
        crs_input: A CRS in any supported format (CRS object, WKT string,
            EPSG code like "EPSG:3003" or 3003, PROJ string, etc.)

    Returns:
        EPSG code as integer, or None if it cannot be determined.

    Examples:
        >>> get_epsg_code("EPSG:3003")
        3003
        >>> get_epsg_code(3003)
        3003
        >>> get_epsg_code(wkt_string)  # WKT with EPSG authority
        3003
    """
    if crs_input is None:
        return None

    # Already an integer
    if isinstance(crs_input, int):
        return crs_input

    # Empty string
    if isinstance(crs_input, str) and not crs_input.strip():
        return None

    # Try to parse with pyproj and extract EPSG
    try:
        if isinstance(crs_input, CRS):
            crs = crs_input
            wkt_string = crs.to_wkt()
        else:
            crs = CRS.from_user_input(crs_input)
            # Keep original string for WKT fallback parsing
            wkt_string = crs_input if isinstance(crs_input, str) else crs.to_wkt()

        epsg = crs.to_epsg()
        if epsg is not None:
            return epsg

        # Fallback: parse AUTHORITY/ID tag from WKT string
        epsg = _extract_epsg_from_wkt(wkt_string)
        if epsg is not None:
            logger.debug(f"EPSG {epsg} extracted from WKT AUTHORITY tag (pyproj.to_epsg() returned None)")
        return epsg

    except Exception as e:
        logger.debug(f"Failed to extract EPSG code: {e}")
        return None


def parse_crs(crs_input: Any) -> CRS | None:
    """Parse a CRS from various input formats.

    Handles pyproj.CRS, rasterio.crs.CRS, WKT strings, EPSG codes,
    PROJ strings, and authority codes.

    Args:
        crs_input: A CRS in any supported format (CRS object, WKT string,
            EPSG code like "EPSG:3003" or 3003, PROJ string, etc.)

    Returns:
        pyproj.CRS object, or None if parsing fails.

    Examples:
        >>> crs = parse_crs("EPSG:3003")
        >>> crs.to_epsg()
        3003
        >>> crs = parse_crs(wkt_string)
        >>> crs is not None
        True
    """
    if crs_input is None:
        return None

    # Already a pyproj CRS
    if isinstance(crs_input, CRS):
        return crs_input

    # Empty string
    if isinstance(crs_input, str) and not crs_input.strip():
        return None

    try:
        return CRS.from_user_input(crs_input)
    except Exception as e:
        logger.debug(f"Failed to parse CRS: {e}")
        return None


def crs_equals(crs1: Any, crs2: Any) -> bool:
    """Compare two CRS for equality by comparing their EPSG codes.

    Extracts the EPSG code from both inputs and compares them as integers.
    This is more reliable than comparing WKT strings or using pyproj.equals()
    which can fail due to minor differences in WKT versions or formatting.

    Args:
        crs1: First CRS (CRS object, WKT string, EPSG code, etc.)
        crs2: Second CRS (CRS object, WKT string, EPSG code, etc.)

    Returns:
        True if both CRS have the same EPSG code, False otherwise.
        Returns False if either EPSG code cannot be determined.

    Examples:
        >>> crs_equals("EPSG:3003", 3003)
        True
        >>> crs_equals(wkt_string, "EPSG:3003")
        True
        >>> crs_equals("EPSG:3003", "EPSG:4326")
        False
    """
    epsg1 = get_epsg_code(crs1)
    epsg2 = get_epsg_code(crs2)

    if epsg1 is None or epsg2 is None:
        logger.debug(f"Cannot compare CRS: epsg1={epsg1}, epsg2={epsg2}")
        return False

    return epsg1 == epsg2
