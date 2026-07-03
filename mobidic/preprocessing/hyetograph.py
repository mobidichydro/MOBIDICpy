"""Hyetograph construction module for IDF-based precipitation generation.

This module provides functionality to construct synthetic hyetographs (rainfall
time series) from Intensity-Duration-Frequency (IDF) parameters stored as
spatially distributed rasters.

The module supports:
- Reading IDF parameters (a, n, k) from GeoTIFF raster files
- Resampling IDF rasters to match a reference grid (e.g., DEM)
- Generating Chicago hyetographs with a configurable peak position (r_chicago) and
  constant-intensity rectangular hyetographs
- Outputting CF-1.12 compliant NetCDF files compatible with MeteoRaster

IDF formula: h = ka * k * a * t^n (precipitation depth as function of duration)
where:
- h is precipitation depth (mm)
- ka is the areal reduction factor (ARF) coefficient
- k is the return period factor
- a is the IDF scale parameter
- t is duration (hours)
- n is the IDF exponent parameter

The parameters a, n, k are spatially distributed and read from raster files.

"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rasterio
import xarray as xr
from loguru import logger
from rasterio.enums import Resampling
from rasterio.warp import reproject

from mobidic.preprocessing.gis_reader import grid_to_matrix
from mobidic.utils.crs import crs_to_cf_attrs, get_epsg_code

from mobidic import __version__

if TYPE_CHECKING:
    from mobidic.config.schema import MOBIDICConfig
    from mobidic.preprocessing.meteo_raster import MeteoRaster


@dataclass
class IDFParameters:
    """Container for IDF (Intensity-Duration-Frequency) raster parameters.

    Attributes:
        a: 2D array of IDF scale parameter (a) values
        n: 2D array of IDF exponent parameter (n) values
        k: 2D array of return period factor (k) values
        xllcorner: X coordinate of lower-left corner (cell center)
        yllcorner: Y coordinate of lower-left corner (cell center)
        cellsize: Grid cell size in map units (meters)
        crs: Coordinate reference system (pyproj CRS or WKT string)
        shape: Grid shape (nrows, ncols)
    """

    a: np.ndarray
    n: np.ndarray
    k: np.ndarray
    xllcorner: float
    yllcorner: float
    cellsize: float
    crs: object
    shape: tuple[int, int]


def resample_raster_to_grid(
    input_path: str | Path,
    ref_shape: tuple[int, int],
    ref_transform: rasterio.Affine,
    ref_crs: Any,
    ref_mask: np.ndarray | None = None,
    resampling_method: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Resample a raster to match a reference grid.

    Resamples the input raster to match the extent, resolution, and CRS of a
    reference grid using the specified resampling method. Handles coordinate
    system transformation if the rasters have different CRS.

    Args:
        input_path: Path to the input raster file (GeoTIFF)
        ref_shape: Shape of reference grid (nrows, ncols)
        ref_transform: Affine transform of reference grid
        ref_crs: CRS of reference grid (pyproj CRS, rasterio CRS, EPSG code, or WKT)
        ref_mask: Optional boolean mask for valid cells (True = valid).
            Invalid cells will be set to NaN.
        resampling_method: Resampling method (default: nearest neighbor)

    Returns:
        2D numpy array resampled to match the reference grid

    Raises:
        FileNotFoundError: If input raster file does not exist
        RuntimeError: If resampling fails

    Examples:
        >>> import rasterio
        >>> from rasterio.transform import from_bounds
        >>>
        >>> # Define reference grid from DEM
        >>> ref_shape = (253, 313)
        >>> ref_transform = from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)
        >>> ref_crs = "EPSG:32632"
        >>>
        >>> # Resample IDF parameter to match DEM grid
        >>> a_resampled = resample_raster_to_grid(
        ...     "idf/a.tif",
        ...     ref_shape, ref_transform, ref_crs
        ... )
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    logger.debug(f"Resampling raster to reference grid: {input_path}")

    try:
        with rasterio.open(input_path) as src:
            # Get source properties
            src_data = src.read(1).astype(np.float64)
            src_transform = src.transform
            src_crs = src.crs
            src_nodata = src.nodata

            # Convert nodata to NaN
            if src_nodata is not None:
                src_data[src_data == src_nodata] = np.nan

            # Check if CRS transformation is needed
            src_epsg = get_epsg_code(src_crs) if src_crs else None
            ref_epsg = get_epsg_code(ref_crs) if ref_crs else None

            need_crs_transform = src_epsg is not None and ref_epsg is not None and src_epsg != ref_epsg

            if need_crs_transform:
                logger.debug(f"CRS transformation required: EPSG:{src_epsg} -> EPSG:{ref_epsg}")

            # Always use reference transform for output
            dst_transform = ref_transform

            # Prepare output array
            dst_data = np.empty(ref_shape, dtype=np.float64)
            dst_data.fill(np.nan)

            # Perform reprojection/resampling
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs if src_crs else ref_crs,
                dst_transform=dst_transform,
                dst_crs=ref_crs,
                resampling=resampling_method,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

        # Apply mask if provided
        if ref_mask is not None:
            dst_data[~ref_mask] = np.nan

        # Filter very small nodata values
        dst_data[dst_data < -1e32] = np.nan

        logger.debug(
            f"Resampling complete: {src_data.shape} -> {dst_data.shape}, valid cells: {np.sum(~np.isnan(dst_data))}"
        )

        return dst_data

    except Exception as e:
        logger.error(f"Failed to resample raster {input_path}: {e}")
        raise RuntimeError(f"Failed to resample raster: {e}") from e


def read_idf_parameters_resampled(
    a_raster_path: str | Path,
    n_raster_path: str | Path,
    k_raster_path: str | Path,
    ref_raster_path: str | Path,
    ref_mask: np.ndarray | None = None,
) -> IDFParameters:
    """Read IDF parameters from rasters and resample to match a reference grid.

    Reads the three IDF parameters (a, n, k) from separate raster files and
    resamples them to match the extent, resolution, and CRS of a reference
    raster (typically the DEM).

    Uses nearest neighbor interpolation.

    Args:
        a_raster_path: Path to raster file containing IDF 'a' parameter
        n_raster_path: Path to raster file containing IDF 'n' parameter
        k_raster_path: Path to raster file containing IDF 'k' parameter
        ref_raster_path: Path to reference raster (e.g., DEM) that defines
            the target grid extent, resolution, and CRS
        ref_mask: Optional boolean mask for valid cells (True = valid).
            If None, derived from non-NaN cells in reference raster.

    Returns:
        IDFParameters object containing resampled parameter grids and metadata
        from the reference raster

    Raises:
        FileNotFoundError: If any raster file does not exist
        RuntimeError: If resampling fails

    Examples:
        >>> # Read IDF parameters resampled to DEM grid
        >>> params = read_idf_parameters_resampled(
        ...     a_raster_path="idf/a.tif",
        ...     n_raster_path="idf/n.tif",
        ...     k_raster_path="idf/k30.tif",
        ...     ref_raster_path="dem.tif"
        ... )
        >>> print(f"Grid shape: {params.shape}")  # Same as DEM shape
    """
    logger.info("Reading IDF parameters with resampling to reference grid")

    # Read reference raster to get grid properties
    ref_raster_path = Path(ref_raster_path)
    if not ref_raster_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_raster_path}")

    logger.debug(f"Reading reference grid from: {ref_raster_path}")

    with rasterio.open(ref_raster_path) as ref_src:
        ref_shape = (ref_src.height, ref_src.width)
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_bounds = ref_src.bounds
        cellsize = ref_transform[0]  # Assuming square pixels

        # Create mask from reference raster if not provided
        if ref_mask is None:
            ref_data = ref_src.read(1).astype(np.float64)
            nodata = ref_src.nodata
            if nodata is not None:
                ref_data[ref_data == nodata] = np.nan
            ref_mask = ~np.isnan(ref_data)
            # Flip mask to match grid_to_matrix convention
            ref_mask = np.flipud(ref_mask)

    logger.debug(f"Reference grid: shape={ref_shape}, cellsize={cellsize}m")

    # Resample each IDF parameter
    logger.debug(f"Resampling 'a' parameter from: {a_raster_path}")
    a_resampled = resample_raster_to_grid(a_raster_path, ref_shape, ref_transform, ref_crs, ref_mask=None)

    logger.debug(f"Resampling 'n' parameter from: {n_raster_path}")
    n_resampled = resample_raster_to_grid(n_raster_path, ref_shape, ref_transform, ref_crs, ref_mask=None)

    logger.debug(f"Resampling 'k' parameter from: {k_raster_path}")
    k_resampled = resample_raster_to_grid(k_raster_path, ref_shape, ref_transform, ref_crs, ref_mask=None)

    # Flip arrays to match grid_to_matrix convention (y increasing from south)
    a_resampled = np.flipud(a_resampled)
    n_resampled = np.flipud(n_resampled)
    k_resampled = np.flipud(k_resampled)

    # Apply mask after flipping
    if ref_mask is not None:
        a_resampled[~ref_mask] = np.nan
        n_resampled[~ref_mask] = np.nan
        k_resampled[~ref_mask] = np.nan

    # Validate sufficient valid cells (minimum 10%)
    _validate_valid_cells_threshold(a_resampled, n_resampled, k_resampled)

    # Calculate corner coordinates (cell center) matching grid_to_matrix convention
    xllcorner = ref_bounds.left + 0.5 * cellsize
    yllcorner = ref_bounds.bottom + 0.5 * cellsize

    params = IDFParameters(
        a=a_resampled,
        n=n_resampled,
        k=k_resampled,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        cellsize=cellsize,
        crs=ref_crs,
        shape=ref_shape,
    )

    logger.success(
        f"IDF parameters resampled to reference grid: shape={params.shape}, "
        f"cellsize={params.cellsize}m, "
        f"a range=[{np.nanmin(params.a):.2f}, {np.nanmax(params.a):.2f}], "
        f"n range=[{np.nanmin(params.n):.3f}, {np.nanmax(params.n):.3f}], "
        f"k range=[{np.nanmin(params.k):.2f}, {np.nanmax(params.k):.2f}]"
    )

    return params


def read_idf_parameters(
    a_raster_path: str | Path,
    n_raster_path: str | Path,
    k_raster_path: str | Path,
) -> IDFParameters:
    """Read IDF parameters from GeoTIFF raster files.

    Reads the three IDF parameters (a, n, k) from separate raster files and
    validates that they have consistent spatial properties (shape, resolution,
    extent, CRS).

    Args:
        a_raster_path: Path to raster file containing IDF 'a' parameter
        n_raster_path: Path to raster file containing IDF 'n' parameter
        k_raster_path: Path to raster file containing IDF 'k' parameter
            (return period factor)

    Returns:
        IDFParameters object containing all three parameter grids and metadata

    Raises:
        FileNotFoundError: If any raster file does not exist
        ValueError: If rasters have inconsistent spatial properties

    Examples:
        >>> params = read_idf_parameters(
        ...     "idf/a.tif",
        ...     "idf/n.tif",
        ...     "idf/k100.tif"
        ... )
        >>> print(f"Grid shape: {params.shape}")
    """
    logger.info("Reading IDF parameters from raster files")

    # Read all three rasters
    logger.debug(f"Reading 'a' parameter from: {a_raster_path}")
    a_data = grid_to_matrix(a_raster_path)

    logger.debug(f"Reading 'n' parameter from: {n_raster_path}")
    n_data = grid_to_matrix(n_raster_path)

    logger.debug(f"Reading 'k' parameter from: {k_raster_path}")
    k_data = grid_to_matrix(k_raster_path)

    # Validate spatial consistency
    _validate_raster_consistency(a_data, n_data, k_data)

    # Validate sufficient valid cells (minimum 10%)
    _validate_valid_cells_threshold(a_data["data"], n_data["data"], k_data["data"])

    # Extract common metadata from 'a' raster (already validated to be consistent)
    params = IDFParameters(
        a=a_data["data"],
        n=n_data["data"],
        k=k_data["data"],
        xllcorner=a_data["xllcorner"],
        yllcorner=a_data["yllcorner"],
        cellsize=a_data["cellsize"],
        crs=a_data.get("crs"),
        shape=a_data["data"].shape,
    )

    logger.success(
        f"IDF parameters loaded: shape={params.shape}, "
        f"cellsize={params.cellsize}m, "
        f"a range=[{np.nanmin(params.a):.2f}, {np.nanmax(params.a):.2f}], "
        f"n range=[{np.nanmin(params.n):.3f}, {np.nanmax(params.n):.3f}], "
        f"k range=[{np.nanmin(params.k):.2f}, {np.nanmax(params.k):.2f}]"
    )

    return params


def _validate_raster_consistency(
    a_data: dict,
    n_data: dict,
    k_data: dict,
) -> None:
    """Validate that IDF rasters have consistent spatial properties.

    Args:
        a_data: Dictionary from grid_to_matrix for 'a' parameter
        n_data: Dictionary from grid_to_matrix for 'n' parameter
        k_data: Dictionary from grid_to_matrix for 'k' parameter

    Raises:
        ValueError: If rasters have inconsistent properties
    """
    errors = []

    # Check shapes
    shapes = [a_data["data"].shape, n_data["data"].shape, k_data["data"].shape]
    if not all(s == shapes[0] for s in shapes):
        errors.append(f"Shape mismatch: a={shapes[0]}, n={shapes[1]}, k={shapes[2]}")

    # Check cell sizes (with tolerance for floating point)
    cellsizes = [a_data["cellsize"], n_data["cellsize"], k_data["cellsize"]]
    if not all(np.isclose(cs, cellsizes[0], rtol=1e-6) for cs in cellsizes):
        errors.append(f"Cellsize mismatch: a={cellsizes[0]}, n={cellsizes[1]}, k={cellsizes[2]}")

    # Check origins (with tolerance)
    xll = [a_data["xllcorner"], n_data["xllcorner"], k_data["xllcorner"]]
    yll = [a_data["yllcorner"], n_data["yllcorner"], k_data["yllcorner"]]

    if not all(np.isclose(x, xll[0], atol=1e-3) for x in xll):
        errors.append(f"X origin mismatch: a={xll[0]}, n={xll[1]}, k={xll[2]}")

    if not all(np.isclose(y, yll[0], atol=1e-3) for y in yll):
        errors.append(f"Y origin mismatch: a={yll[0]}, n={yll[1]}, k={yll[2]}")

    if errors:
        error_msg = "IDF rasters have inconsistent spatial properties:\n  - " + "\n  - ".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("IDF rasters validated: consistent spatial properties")


def _validate_valid_cells_threshold(
    a: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    min_valid_fraction: float = 0.1,
) -> None:
    """Validate that IDF rasters have sufficient valid (non-NaN) cells.

    Args:
        a: 2D array of IDF 'a' parameter
        n: 2D array of IDF 'n' parameter
        k: 2D array of IDF 'k' parameter
        min_valid_fraction: Minimum required fraction of valid cells (default: 0.1 = 10%)

    Raises:
        ValueError: If any raster has less than the minimum fraction of valid cells
    """
    total_cells = a.size
    errors = []

    # Check each parameter
    for param_name, param_data in [("a", a), ("n", n), ("k", k)]:
        valid_cells = np.sum(~np.isnan(param_data))
        valid_fraction = valid_cells / total_cells

        if valid_fraction < min_valid_fraction:
            errors.append(
                f"Parameter '{param_name}': {valid_fraction * 100:.1f}% valid cells "
                f"({valid_cells}/{total_cells}), minimum required: {min_valid_fraction * 100:.0f}%"
            )

    if errors:
        error_msg = (
            f"IDF rasters have insufficient valid cells (minimum {min_valid_fraction * 100:.0f}% required):\n  - "
            + "\n  - ".join(errors)
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"IDF rasters validated: sufficient valid cells (minimum {min_valid_fraction * 100:.0f}% required)")


def idf_depth(a: np.ndarray, n: np.ndarray, t: float) -> np.ndarray:
    """Calculate distributed precipitation depth from IDF formula.

    Computes h = a * t^n where t is duration in hours.

    Args:
        a: 2D array of IDF scale parameter values
        n: 2D array of IDF exponent parameter values
        t: Duration in hours

    Returns:
        2D array of precipitation depth (mm)

    Notes:
        - This is the base IDF formula without return period factor (k) or
          areal reduction factor (ka)
        - NaN values in input arrays propagate to output
    """
    return a * np.power(t, n)


class HyetographGenerator:
    """Generator for synthetic hyetographs from IDF parameters.

    This class constructs synthetic rainfall time series (hyetographs) from
    spatially distributed IDF parameters.

    Two construction methods are available:

    - ``chicago``: Chicago hyetograph with a configurable peak position
      (``r_chicago`` in [0, 1]); ``r_chicago=0`` gives a decreasing storm with the
      peak at the start, ``0.5`` a centered peak, and ``1`` an increasing storm with
      the peak at the end.
    - ``rectangular``: constant-intensity storm over the rainfall duration.

    The rainfall event spans ``rainfall_duration`` hours; if the total
    ``duration_hours`` is longer, the remaining timesteps are dry (zero precipitation).

    The total precipitation depth for a given duration is computed as:
        DDF(t) = ka * k * a * t^n

    where:
        - ka is the areal reduction factor (ARF) coefficient
        - k is the return period factor (spatially distributed)
        - a is the IDF scale parameter (spatially distributed)
        - n is the IDF exponent parameter (spatially distributed)
        - t is duration in hours

    Attributes:
        idf_params: IDFParameters object with a, n, k grids
        ka: Areal reduction factor coefficient

    Examples:
        >>> # Simplest workflow: generate from configuration (recommended)
        >>> from mobidic import load_config, load_gisdata, Simulation
        >>> config = load_config("basin_hyetograph.yaml")
        >>> gisdata = load_gisdata(config.paths.gisdata, config.paths.network)
        >>> forcing = HyetographGenerator.from_config(
        ...     config=config,
        ...     base_path="basin_dir",
        ...     start_time=datetime(2000, 1, 1)
        ... )
        >>> sim = Simulation(gisdata, forcing, config)
        >>> results = sim.run(forcing.start_date, forcing.end_date)
        >>>
        >>> # Alternative: generate forcing with manual parameters
        >>> generator = HyetographGenerator.from_rasters(
        ...     a_raster="idf/a.tif",
        ...     n_raster="idf/n.tif",
        ...     k_raster="idf/k30.tif",
        ...     ka=0.8,
        ...     ref_raster="dem.tif"
        ... )
        >>> forcing = generator.generate_forcing(
        ...     duration_hours=48,
        ...     start_time=datetime(2023, 11, 1),
        ...     output_path="design_storm.nc",
        ...     add_metadata={"return_period": "30 years"}
        ... )
        >>>
        >>> # Advanced workflow: manual control over generation and export
        >>> times, precip = generator.generate(
        ...     duration_hours=48,
        ...     start_time=datetime(2023, 11, 1),
        ...     rainfall_duration=24,
        ...     method="chicago",
        ...     r_chicago=0.5,
        ... )
        >>> generator.to_netcdf(
        ...     "hyetograph.nc",
        ...     times=times,
        ...     precipitation=precip,
        ...     add_metadata={"event": "design_storm_30yr"}
        ... )
    """

    def __init__(self, idf_params: IDFParameters, ka: float = 1.0):
        """Initialize HyetographGenerator with IDF parameters.

        Args:
            idf_params: IDFParameters object containing a, n, k grids
            ka: Areal reduction factor (ARF) coefficient (default: 1.0)
        """
        self.idf_params = idf_params
        self.ka = ka

        logger.debug(f"HyetographGenerator initialized: ka={ka}, shape={idf_params.shape}")

    @classmethod
    def from_rasters(
        cls,
        a_raster: str | Path,
        n_raster: str | Path,
        k_raster: str | Path,
        ka: float = 1.0,
        ref_raster: str | Path | None = None,
    ) -> "HyetographGenerator":
        """Create HyetographGenerator by loading IDF parameters from raster files.

        If a reference raster is provided, the IDF parameters will be resampled
        to match its extent, resolution, and CRS using nearest neighbor interpolation.
        This is the typical workflow when the IDF rasters have different resolution
        than the model grid (e.g., DEM).

        Args:
            a_raster: Path to raster file containing IDF 'a' parameter
            n_raster: Path to raster file containing IDF 'n' parameter
            k_raster: Path to raster file containing IDF 'k' parameter
            ka: Areal reduction factor (ARF) coefficient (default: 1.0)
            ref_raster: Optional path to reference raster (e.g., DEM) for resampling.
                If provided, IDF parameters will be resampled to match this grid.

        Returns:
            HyetographGenerator instance

        Examples:
            >>> # Without resampling (IDF rasters already aligned)
            >>> generator = HyetographGenerator.from_rasters(
            ...     a_raster="idf/a.tif",
            ...     n_raster="idf/n.tif",
            ...     k_raster="idf/k30.tif",
            ...     ka=0.8
            ... )
            >>>
            >>> # With resampling to DEM grid
            >>> generator = HyetographGenerator.from_rasters(
            ...     a_raster="idf/a.tif",
            ...     n_raster="idf/n.tif",
            ...     k_raster="idf/k30.tif",
            ...     ka=0.8,
            ...     ref_raster="dem.tif"
            ... )
        """
        if ref_raster is not None:
            idf_params = read_idf_parameters_resampled(a_raster, n_raster, k_raster, ref_raster)
        else:
            idf_params = read_idf_parameters(a_raster, n_raster, k_raster)
        return cls(idf_params, ka=ka)

    @classmethod
    def from_config(
        cls,
        config: "MOBIDICConfig",
        base_path: str | Path,
        start_time: datetime,
        preload: bool = True,
    ) -> "MeteoRaster":
        """Create hyetograph forcing from parameters specified in configuration file.

        Convenience method that reads hyetograph parameters from a yaml
        configuration object, generates the hyetograph, saves to NetCDF, and
        returns a MeteoRaster ready for simulation.

        All parameters (duration, timestep, method, IDF rasters, output path) are
        read from the configuration file. Only the start time needs to be specified.
        The start time is a reference datetime for the hyetograph event.

        Args:
            config: MOBIDICConfig object with hyetograph configuration section
            base_path: Base path for resolving relative paths in config (typically
                the directory containing the config file)
            start_time: Start datetime for the hyetograph
            preload: If True, preload all data into memory for fast access
                (default: True, recommended for normal use)

        Returns:
            MeteoRaster object ready for use in Simulation

        Raises:
            AttributeError: If config does not have a hyetograph section
            ValueError: If required configuration parameters are missing

        Examples:
            >>> from mobidic import load_config, load_gisdata, Simulation
            >>> from mobidic.preprocessing.hyetograph import HyetographGenerator
            >>> from datetime import datetime
            >>> from pathlib import Path
            >>>
            >>> # Load configuration
            >>> config_file = Path("basin_hyetograph.yaml")
            >>> config = load_config(config_file)
            >>> gisdata = load_gisdata(config.paths.gisdata, config.paths.network)
            >>>
            >>> # Generate hyetograph
            >>> forcing = HyetographGenerator.from_config(
            ...     config=config,
            ...     base_path=config_file.parent,
            ...     start_time=datetime(2000, 1, 1)
            ... )
            >>>
            >>> # Run simulation
            >>> sim = Simulation(gisdata, forcing, config)
            >>> results = sim.run(forcing.start_date, forcing.end_date)

        Notes:
            - Automatically resamples IDF parameters to DEM grid
            - Uses all hyetograph parameters from config (duration, timestep, method, ka)
            - Output path read from config.paths.hyetograph
            - Creates metadata from config basin and hyetograph sections
            - Returns MeteoRaster ready for simulation with proper date range
        """
        # Import here to avoid circular dependency

        base_path = Path(base_path)

        # Validate config has hyetograph section
        if not hasattr(config, "hyetograph"):
            raise AttributeError("Configuration does not have a 'hyetograph' section")

        # Validate config has paths.hyetograph
        if not hasattr(config.paths, "hyetograph") or config.paths.hyetograph is None:
            raise ValueError(
                "Configuration must specify 'paths.hyetograph' for the output NetCDF file. "
                "Add 'hyetograph: path/to/output.nc' to the 'paths' section in the config file."
            )

        hyeto_config = config.hyetograph

        # Resolve paths relative to base_path
        a_raster_path = base_path / hyeto_config.a_raster
        n_raster_path = base_path / hyeto_config.n_raster
        k_raster_path = base_path / hyeto_config.k_raster
        ref_raster_path = base_path / config.raster_files.dtm
        output_path = base_path / config.paths.hyetograph

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Generating hyetograph forcing from configuration")
        logger.debug(f"  Base path: {base_path}")
        logger.debug(f"  Duration: {hyeto_config.duration_hours} hours")
        logger.debug(f"  Rainfall duration: {hyeto_config.rainfall_duration} hours")
        logger.debug(f"  Timestep: {hyeto_config.timestep_hours} hour(s)")
        logger.debug(f"  Method: {hyeto_config.hyetograph_type}")
        logger.debug(f"  r_chicago: {hyeto_config.r_chicago}")
        logger.debug(f"  Output: {output_path}")

        # Create generator with IDF parameters resampled to DEM grid
        generator = cls.from_rasters(
            a_raster=a_raster_path,
            n_raster=n_raster_path,
            k_raster=k_raster_path,
            ka=hyeto_config.ka,
            ref_raster=ref_raster_path,
        )

        # Prepare metadata from config
        add_metadata = {
            "hyetograph_method": hyeto_config.hyetograph_type,
            "duration_hours": hyeto_config.duration_hours,
            "rainfall_duration_hours": hyeto_config.rainfall_duration,
            "timestep_hours": hyeto_config.timestep_hours,
            "areal_reduction_factor": hyeto_config.ka,
            "k_raster": str(hyeto_config.k_raster),
        }
        if hyeto_config.r_chicago is not None:
            add_metadata["r_chicago"] = hyeto_config.r_chicago

        # Add basin metadata if available
        if hasattr(config, "basin"):
            if hasattr(config.basin, "id") and config.basin.id:
                add_metadata["basin"] = config.basin.id
            if hasattr(config.basin, "paramset_id") and config.basin.paramset_id:
                add_metadata["scenario"] = config.basin.paramset_id

        # Generate forcing using generate_forcing method
        forcing = generator.generate_forcing(
            duration_hours=hyeto_config.duration_hours,
            start_time=start_time,
            output_path=output_path,
            rainfall_duration=hyeto_config.rainfall_duration,
            method=hyeto_config.hyetograph_type,
            timestep_hours=hyeto_config.timestep_hours,
            r_chicago=hyeto_config.r_chicago,
            add_metadata=add_metadata,
            preload=preload,
        )

        return forcing

    def generate(
        self,
        duration_hours: int,
        start_time: datetime,
        rainfall_duration: float | None = None,
        method: Literal["chicago", "rectangular"] = "chicago",
        timestep_hours: int = 1,
        r_chicago: float | None = None,
    ) -> tuple[list[datetime], np.ndarray]:
        """Generate hyetograph precipitation time series.

        The rainfall event spans ``rainfall_duration`` hours. If ``duration_hours``
        is longer, the remaining timesteps are dry (zero precipitation), which is
        useful to let the flood wave propagate through the network after the storm.

        Args:
            duration_hours: Total duration of the output time series in hours
            start_time: Start datetime for the hyetograph
            rainfall_duration: Duration of the rainfall event in hours. Must be
                <= duration_hours. If None, defaults to duration_hours (rain for
                the whole series).
            method: Hyetograph construction method, either "chicago" or "rectangular".
            timestep_hours: Time step in hours (default: 1)
            r_chicago: Position of the peak within the rainfall duration for the
                Chicago method, in [0, 1] (0 = peak at start / decreasing,
                0.5 = centered, 1 = peak at end / increasing). Required when
                method="chicago", ignored for "rectangular".

        Returns:
            Tuple of (times, precipitation) where:
                - times: List of datetime objects for each timestep
                - precipitation: 3D array (time, y, x) of precipitation (mm/h)

        Raises:
            ValueError: If method is not supported, if r_chicago is missing or out
                of range for the Chicago method, or if rainfall_duration exceeds
                duration_hours.

        Notes:
            - Precipitation values are in mm/h (intensity)
            - NaN values in IDF parameters propagate to all output timesteps
            - If NaN values are > 90% in any parameter, a ValueError is raised during
              initialization of the HyetographGenerator
        """
        if method not in ("chicago", "rectangular"):
            raise ValueError(f"Unsupported hyetograph method: {method}. Use 'chicago' or 'rectangular'.")

        if rainfall_duration is None:
            rainfall_duration = duration_hours

        if rainfall_duration <= 0:
            raise ValueError("rainfall_duration must be positive.")

        if rainfall_duration > duration_hours:
            raise ValueError(
                f"rainfall_duration ({rainfall_duration} h) cannot exceed duration_hours ({duration_hours} h)."
            )

        if method == "chicago":
            if r_chicago is None:
                raise ValueError("r_chicago is required for the 'chicago' method.")
            if r_chicago < 0 or r_chicago > 1:
                raise ValueError("r_chicago must be in range [0, 1].")

        logger.info(
            f"Generating {method} hyetograph: duration={duration_hours}h, "
            f"rainfall_duration={rainfall_duration}h, timestep={timestep_hours}h, "
            f"r_chicago={r_chicago}, start={start_time}"
        )

        return self._build_hyetograph(
            duration_hours=duration_hours,
            start_time=start_time,
            rainfall_duration=rainfall_duration,
            method=method,
            timestep_hours=timestep_hours,
            r_chicago=r_chicago,
        )

    def _ddf_depth(self, d: float) -> np.ndarray:
        """Distributed DDF depth h = ka * k * a * d^n at duration d [hours].

        Args:
            d: Duration in hours (>= 0)

        Returns:
            2D array of cumulated precipitation depth [mm]. Returns zeros for d <= 0.
        """
        if d <= 0:
            return np.zeros(self.idf_params.shape)
        return self.ka * self.idf_params.k * (self.idf_params.a * np.power(d, self.idf_params.n))

    def _chicago_cumulative(self, t: float, rainfall_duration: float, r: float, ddf_total: np.ndarray) -> np.ndarray:
        """Cumulated rainfall depth of the Chicago curve at time t.

        Implements the Chicago hyetograph cumulative curve with a peak located at
        ``r * rainfall_duration``. The limiting cases r=0 (decreasing) and r=1
        (increasing) are handled explicitly; 0 < r < 1 blends the pre-peak and
        post-peak branches.

        Args:
            t: Time within the rainfall event [hours], 0 <= t <= rainfall_duration
            rainfall_duration: Total rainfall duration [hours]
            r: Peak position within the rainfall duration, in [0, 1]
            ddf_total: Distributed total depth at rainfall_duration [mm]

        Returns:
            2D array of cumulated depth at time t [mm]
        """
        if r == 0:
            # Peak at the start: decreasing hyetograph
            return self._ddf_depth(t)
        if r == 1:
            # Peak at the end: increasing hyetograph
            return ddf_total - self._ddf_depth(rainfall_duration - t)

        # General case: 0 < r < 1
        tp = r * rainfall_duration  # time of the peak [h]
        if t <= tp:
            d_eff = (tp - t) / r
            return r * (ddf_total - self._ddf_depth(d_eff))
        d_eff = (t - tp) / (1 - r)
        return r * ddf_total + (1 - r) * self._ddf_depth(d_eff)

    def _build_hyetograph(
        self,
        duration_hours: int,
        start_time: datetime,
        rainfall_duration: float,
        method: str,
        timestep_hours: int,
        r_chicago: float | None,
    ) -> tuple[list[datetime], np.ndarray]:
        """Construct the hyetograph intensity time series.

        Rainfall increments are computed over intervals of width ``timestep_hours``
        spanning ``rainfall_duration`` (the last interval is clipped to the exact
        rainfall duration), then padded with dry timesteps up to ``duration_hours``.

        Args:
            duration_hours: Total duration of the output series [hours]
            start_time: Start datetime
            rainfall_duration: Rainfall event duration [hours]
            method: "chicago" or "rectangular"
            timestep_hours: Time step [hours]
            r_chicago: Peak position for the Chicago method (ignored for rectangular)

        Returns:
            Tuple of (times, precipitation) where precipitation is in mm/h
        """
        nrows, ncols = self.idf_params.shape

        # Total number of output timesteps and rainfall timesteps
        n_total = int(duration_hours // timestep_hours)
        n_rain = int(np.ceil(rainfall_duration / timestep_hours - 1e-9))
        n_rain = min(n_rain, n_total)

        # Interval edges within the rainfall event (last edge clipped to rainfall_duration)
        t_edges = np.minimum(np.arange(n_rain + 1) * timestep_hours, rainfall_duration)

        # Total depth at the rainfall duration
        ddf_total = self._ddf_depth(rainfall_duration)

        # Incremental precipitation depth [mm] per timestep
        increments = np.zeros((n_total, nrows, ncols))

        if method == "rectangular":
            # Constant intensity over the rainfall duration
            intensity_const = ddf_total / rainfall_duration  # [mm/h]
            for j in range(n_rain):
                dt_j = t_edges[j + 1] - t_edges[j]
                increments[j, :, :] = intensity_const * dt_j
        else:
            # Chicago: increments are differences of the cumulative curve
            c_prev = self._chicago_cumulative(t_edges[0], rainfall_duration, r_chicago, ddf_total)
            for j in range(n_rain):
                c_next = self._chicago_cumulative(t_edges[j + 1], rainfall_duration, r_chicago, ddf_total)
                increments[j, :, :] = c_next - c_prev
                c_prev = c_next

        # Numerical cleanup: remove tiny negative values from rounding (leaves NaN untouched)
        increments[increments < 0] = 0.0

        # Convert from mm/timestep to mm/h (intensity); dividing by the nominal
        # timestep preserves the depth integrated by the simulation over each slot.
        precip_intensity = increments / timestep_hours

        # Propagate the basin mask (NaN in IDF parameters) to every timestep
        mask = np.isnan(self.idf_params.a) | np.isnan(self.idf_params.n) | np.isnan(self.idf_params.k)
        precip_intensity[:, mask] = np.nan

        # Generate times
        times = [start_time + timedelta(hours=i * timestep_hours) for i in range(n_total)]

        peak = precip_intensity[:n_rain] if n_rain > 0 else precip_intensity
        logger.success(
            f"Hyetograph generated: {n_total} timesteps ({n_rain} wet), "
            f"total depth range=[{np.nanmin(ddf_total):.1f}, {np.nanmax(ddf_total):.1f}] mm, "
            f"peak intensity range=[{np.nanmin(peak):.2f}, {np.nanmax(peak):.2f}] mm/h"
        )

        return times, precip_intensity

    def generate_forcing(
        self,
        duration_hours: int,
        start_time: datetime,
        output_path: str | Path,
        rainfall_duration: float | None = None,
        method: Literal["chicago", "rectangular"] = "chicago",
        timestep_hours: int = 1,
        r_chicago: float | None = None,
        add_metadata: dict | None = None,
        preload: bool = True,
    ) -> "MeteoRaster":
        """Generate hyetograph and return as MeteoRaster ready for simulation.

        Convenience method that combines generate(), to_netcdf(), and
        MeteoRaster.from_netcdf() into a single call. This simplifies the
        workflow for design storm simulations.

        Args:
            duration_hours: Total duration of the output series in hours
            start_time: Start datetime for the hyetograph
            output_path: Path for output NetCDF file
            rainfall_duration: Duration of the rainfall event in hours (<= duration_hours).
                If None, defaults to duration_hours.
            method: Hyetograph construction method, "chicago" or "rectangular"
                (default: "chicago")
            timestep_hours: Time step in hours (default: 1)
            r_chicago: Peak position for the Chicago method, in [0, 1]. Required
                when method="chicago".
            add_metadata: Optional dictionary of additional global attributes
            preload: If True, preload all data into memory for fast access
                (default: True, recommended for normal use)

        Returns:
            MeteoRaster object ready for use in Simulation

        Examples:
            >>> # Create generator from IDF rasters
            >>> generator = HyetographGenerator.from_rasters(
            ...     a_raster="idf/a.tif",
            ...     n_raster="idf/n.tif",
            ...     k_raster="idf/k30.tif",
            ...     ka=0.8,
            ...     ref_raster="dem.tif"
            ... )
            >>>
            >>> # Generate forcing and get MeteoRaster in one call
            >>> forcing = generator.generate_forcing(
            ...     duration_hours=48,
            ...     start_time=datetime(2023, 11, 1),
            ...     output_path="design_hyetograph.nc",
            ...     add_metadata={"return_period": "30 years"}
            ... )
            >>>
            >>> # Use directly in simulation
            >>> sim = Simulation(gisdata, forcing, config)
            >>> results = sim.run(forcing.start_date, forcing.end_date)

        Notes:
            - This method is equivalent to calling generate(), to_netcdf(),
              and MeteoRaster.from_netcdf() sequentially
            - The NetCDF file is still created at output_path for later use
        """
        # Import here to avoid circular dependency
        from mobidic.preprocessing.meteo_raster import MeteoRaster

        # Generate hyetograph
        times, precipitation = self.generate(
            duration_hours=duration_hours,
            start_time=start_time,
            rainfall_duration=rainfall_duration,
            method=method,
            timestep_hours=timestep_hours,
            r_chicago=r_chicago,
        )

        # Save to NetCDF
        self.to_netcdf(
            output_path=output_path,
            times=times,
            precipitation=precipitation,
            method=method,
            add_metadata=add_metadata,
        )

        # Load as MeteoRaster
        forcing = MeteoRaster.from_netcdf(output_path, preload=preload)

        logger.success(f"Forcing data ready for simulation: {forcing.start_date} to {forcing.end_date}")

        return forcing

    def to_netcdf(
        self,
        output_path: str | Path,
        times: list[datetime],
        precipitation: np.ndarray,
        method: str = "chicago",
        add_metadata: dict | None = None,
    ) -> None:
        """Export hyetograph to CF-compliant NetCDF file.

        Creates a NetCDF file compatible with MeteoRaster.from_netcdf() for
        use as meteorological forcing in MOBIDIC simulations.

        Args:
            output_path: Path for output NetCDF file
            times: List of datetime objects for each timestep
            precipitation: 3D array (time, y, x) of precipitation [mm/h]
            method: Hyetograph construction method recorded in the file metadata
                (default: "chicago")
            add_metadata: Optional dictionary of additional global attributes

        Notes:
            - Output follows CF-1.12 conventions
            - Includes CRS information as grid mapping variable
            - Precipitation units are mm/h (compatible with MeteoRaster)

        Examples:
            >>> times, precip = generator.generate(48, datetime(2023, 11, 1), method="chicago", r_chicago=0)
            >>> generator.to_netcdf(
            ...     "design_storm.nc",
            ...     times=times,
            ...     precipitation=precip,
            ...     add_metadata={"return_period": "30 years"}
            ... )
        """
        output_path = Path(output_path)
        logger.info(f"Writing hyetograph to NetCDF: {output_path}")

        # Build coordinate arrays
        nrows, ncols = self.idf_params.shape
        cellsize = self.idf_params.cellsize

        # X coordinates (cell centers, west to east)
        x = np.arange(ncols) * cellsize + self.idf_params.xllcorner

        # Y coordinates (cell centers, south to north)
        # Note: Data is stored with y increasing (south to north) after flipud in grid_to_matrix
        y = np.arange(nrows) * cellsize + self.idf_params.yllcorner

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                "precipitation": (
                    ["time", "y", "x"],
                    precipitation,
                    {
                        "units": "mm h-1",
                        "long_name": "Precipitation rate",
                        "grid_mapping": "crs",
                    },
                ),
            },
            coords={
                "time": times,
                "y": y,
                "x": x,
            },
            attrs={
                "Conventions": "CF-1.12",
                "title": "Synthetic hyetograph from IDF parameters",
                "source": f"MOBIDICpy version {__version__}",
                "history": f"Created {datetime.now().isoformat()}",
                "hyetograph_method": method,
                "areal_reduction_factor": self.ka,
            },
        )

        # Add coordinate attributes
        ds.x.attrs = {
            "units": "m",
            "long_name": "x coordinate",
            "standard_name": "projection_x_coordinate",
        }
        ds.y.attrs = {
            "units": "m",
            "long_name": "y coordinate",
            "standard_name": "projection_y_coordinate",
        }
        ds.time.attrs = {
            "long_name": "time",
            "standard_name": "time",
        }

        # Add CRS variable
        if self.idf_params.crs is not None:
            crs_attrs = crs_to_cf_attrs(self.idf_params.crs)
            # Create scalar CRS variable
            ds["crs"] = xr.DataArray(
                0,  # Scalar placeholder
                attrs=crs_attrs,
            )

        # Add custom metadata
        if add_metadata:
            ds.attrs.update(add_metadata)

        # Write to NetCDF with compression
        encoding = {
            "precipitation": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            },
            "time": {"dtype": "float64"},
        }

        ds.to_netcdf(output_path, encoding=encoding)

        logger.success(f"Hyetograph written to: {output_path} ({precipitation.shape[0]} timesteps)")
