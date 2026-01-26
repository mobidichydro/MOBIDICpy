"""CRS utilities for CF-compliant NetCDF encoding.

This module provides functions to convert CRS objects to CF-compliant
grid mapping attributes for NetCDF files, and to compare CRS objects.
"""

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
    3. Adds the crs_wkt attribute with proper OGC WKT format
    4. Adds spatial_ref for GDAL/rasterio compatibility
    5. Adds epsg_code for human readability (non-standard but common)

    Args:
        crs_input: A CRS object (pyproj.CRS, rasterio.crs.CRS, or any input
            accepted by pyproj.CRS.from_user_input), or a CRS string/code.

    Returns:
        Dictionary of CF-compliant grid mapping attributes including:
        - grid_mapping_name: Valid CF projection name
        - Projection-specific parameters (e.g., longitude_of_central_meridian)
        - crs_wkt: Full OGC Well-Known Text representation
        - spatial_ref: Same as crs_wkt (for GDAL compatibility)
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
        - The crs_wkt uses WKT2:2019 format for maximum compatibility.
        - epsg_code is non-standard but widely used by tools like ADAGUC.
    """
    # Handle None or empty input
    if crs_input is None or (isinstance(crs_input, str) and not crs_input.strip()):
        logger.warning("Empty or None CRS provided, using default latitude_longitude")
        return {
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": "",
            "spatial_ref": "",
        }

    try:
        # Convert to pyproj CRS if needed
        if isinstance(crs_input, CRS):
            crs = crs_input
        else:
            crs = CRS.from_user_input(crs_input)

        # Get CF-compliant attributes using pyproj's to_cf() method
        cf_attrs = crs.to_cf()

        # Get proper WKT string (WKT2:2019 format)
        wkt_string = crs.to_wkt(version="WKT2_2019")

        # Add WKT attributes
        cf_attrs["crs_wkt"] = wkt_string
        cf_attrs["spatial_ref"] = wkt_string  # GDAL compatibility

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
            wkt_string = crs.to_wkt(version="WKT2_2019")
        except Exception:
            wkt_string = str(crs_input) if crs_input else ""

        return {
            "grid_mapping_name": "latitude_longitude",
            "crs_wkt": wkt_string,
            "spatial_ref": wkt_string,
        }


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
    """Compare two CRS for equality.

    Properly handles different CRS representations (WKT, EPSG codes,
    CRS objects) by parsing both and comparing their canonical forms.

    Args:
        crs1: First CRS (any format accepted by parse_crs)
        crs2: Second CRS (any format accepted by parse_crs)

    Returns:
        True if CRS are equivalent, False otherwise.
        Returns False if either CRS cannot be parsed.

    Examples:
        >>> crs_equals("EPSG:3003", 3003)
        True
        >>> crs_equals(wkt_string, "EPSG:3003")
        True
        >>> crs_equals("EPSG:3003", "EPSG:4326")
        False
    """
    parsed1 = parse_crs(crs1)
    parsed2 = parse_crs(crs2)

    if parsed1 is None or parsed2 is None:
        return False

    return parsed1.equals(parsed2)
