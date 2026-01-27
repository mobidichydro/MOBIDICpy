"""Hyetograph construction module for IDF-based precipitation generation.

This module provides functionality to construct synthetic hyetographs (rainfall
time series) from Intensity-Duration-Frequency (IDF) parameters stored as
spatially distributed rasters.

The module supports:
- Reading IDF parameters (a, n, k) from GeoTIFF raster files
- Resampling IDF rasters to match a reference grid (e.g., DEM)
- Generating Chicago-type hyetograph, currently only after-peak curve ('decreasing') only
- Outputting CF-1.12 compliant NetCDF files compatible with MeteoRaster

IDF formula: h = a * t^n (precipitation depth as function of duration)
where:
- h is precipitation depth [mm]
- a is the IDF scale parameter
- n is the IDF shape parameter (typically <= 1)
- t is duration [hours]

The return period information is encoded in the spatially distributed k parameter,
and the areal reduction factor (ARF) is applied via the ka coefficient.

"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import xarray as xr
from loguru import logger
from rasterio.enums import Resampling
from rasterio.warp import reproject

from mobidic.preprocessing.gis_reader import grid_to_matrix
from mobidic.utils.crs import crs_to_cf_attrs, get_epsg_code

from mobidic import __version__

@dataclass
class IDFParameters:
    """Container for IDF (Intensity-Duration-Frequency) raster parameters.

    Attributes:
        a: 2D array of IDF scale parameter (a) values
        n: 2D array of IDF shape parameter (n) values
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

    This function replicates the behavior of MATLAB's resample_grid.m function.

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
    raster (typically the DEM). Uses nearest neighbor interpolation.

    This function replicates the workflow in MATLAB hyetograph_M.m where IDF
    parameters are resampled to match the DEM grid using resample_grid.m.

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


def idf_depth(a: np.ndarray, n: np.ndarray, t: float) -> np.ndarray:
    """Calculate precipitation depth from IDF formula.

    Computes h = a * t^n where t is duration in hours.

    Args:
        a: 2D array of IDF scale parameter values
        n: 2D array of IDF shape parameter values
        t: Duration in hours

    Returns:
        2D array of precipitation depth [mm]

    Notes:
        - This is the base IDF formula without return period factor (k) or
          areal reduction factor (ka)
        - NaN values in input arrays propagate to output
    """
    return a * np.power(t, n)


class HyetographGenerator:
    """Generator for synthetic hyetographs from IDF parameters.

    This class constructs synthetic rainfall time series (hyetographs) from
    spatially distributed IDF parameters. Currently implements the Chicago
    method (decreasing branch only).

    The total precipitation depth for a given duration is computed as:
        DDF(t) = ka * k * a * t^n

    where:
        - ka is the areal reduction factor (ARF) coefficient
        - k is the return period factor (spatially distributed)
        - a is the IDF scale parameter (spatially distributed)
        - n is the IDF shape parameter (spatially distributed)
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
        ...     method="chicago_decreasing"
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
        config,
        base_path: str | Path,
        start_time: datetime,
        preload: bool = True,
    ):
        """Create hyetograph forcing from MOBIDIC configuration.

        Convenience method that reads hyetograph parameters from a MOBIDIC
        configuration object, generates the hyetograph, saves to NetCDF, and
        returns a MeteoRaster ready for simulation. This is the most streamlined
        workflow for design storm simulations.

        All parameters (duration, timestep, method, IDF rasters, output path) are
        read from the configuration file. Only the start time needs to be specified.

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
                "Add 'hyetograph: path/to/output.nc' to the 'paths' section in your config."
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
        logger.debug(f"  Timestep: {hyeto_config.timestep_hours} hour(s)")
        logger.debug(f"  Method: {hyeto_config.hyetograph_type}")
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
            "timestep_hours": hyeto_config.timestep_hours,
            "areal_reduction_factor": hyeto_config.ka,
            "k_raster": str(hyeto_config.k_raster),
        }

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
            method=hyeto_config.hyetograph_type,
            timestep_hours=hyeto_config.timestep_hours,
            add_metadata=add_metadata,
            preload=preload,
        )

        return forcing

    def generate(
        self,
        duration_hours: int,
        start_time: datetime,
        method: Literal["chicago_decreasing"] = "chicago_decreasing",
        timestep_hours: int = 1,
    ) -> tuple[list[datetime], np.ndarray]:
        """Generate hyetograph precipitation time series.

        Args:
            duration_hours: Total duration of the hyetograph in hours
            start_time: Start datetime for the hyetograph
            method: Hyetograph construction method. Currently only
                "chicago_decreasing" is implemented.
            timestep_hours: Time step in hours (default: 1)

        Returns:
            Tuple of (times, precipitation) where:
                - times: List of datetime objects for each timestep
                - precipitation: 3D array (time, y, x) of precipitation [mm/h]

        Raises:
            ValueError: If method is not supported

        Notes:
            - The Chicago decreasing method generates only the falling limb
              of the Chicago hyetograph (after the peak)
            - Precipitation values are in mm/h (intensity, not depth)
            - NaN values in IDF parameters propagate to output
        """
        if method != "chicago_decreasing":
            raise ValueError(f"Unsupported hyetograph method: {method}. Only 'chicago_decreasing' is implemented.")

        logger.info(
            f"Generating {method} hyetograph: duration={duration_hours}h, "
            f"timestep={timestep_hours}h, start={start_time}"
        )

        return self._chicago_decreasing(duration_hours, start_time, timestep_hours)

    def _chicago_decreasing(
        self,
        duration_hours: int,
        start_time: datetime,
        timestep_hours: int,
    ) -> tuple[list[datetime], np.ndarray]:
        """Generate Chicago decreasing hyetograph.

        The Chicago method constructs a hyetograph where precipitation intensity
        decreases monotonically from the peak. This implementation generates only
        the descending part (after peak).

        Args:
            duration_hours: Total duration in hours
            start_time: Start datetime
            timestep_hours: Time step in hours

        Returns:
            Tuple of (times, precipitation) where precipitation is in mm/h
        """
        n_steps = duration_hours // timestep_hours
        nrows, ncols = self.idf_params.shape

        # Extract parameters
        a = self.idf_params.a
        n = self.idf_params.n
        k = self.idf_params.k
        ka = self.ka

        # Initialize arrays
        # DDF: Depth-Duration-Frequency (cumulative depth) [mm]
        # P: Incremental precipitation [mm per timestep]
        ddf = np.zeros((n_steps, nrows, ncols))
        precip = np.zeros((n_steps, nrows, ncols))

        # Generate times
        times = [start_time + timedelta(hours=i * timestep_hours) for i in range(n_steps)]

        # Chicago hyetograph calculation
        # Note: t is 1-indexed in the formula (t=1 for first hour)
        for i in range(n_steps):
            t = (i + 1) * timestep_hours  # Duration in hours (1, 2, 3, ...)

            # DDF calculation: h = ka * k * a * t^n [mm]
            ddf[i, :, :] = ka * k * (a * np.power(t, n))

            if i > 0:
                # Rainfall increment (Chicago decreasing)
                precip[i, :, :] = ddf[i, :, :] - ddf[i - 1, :, :]
            else:
                # First timestep: P = DDF
                precip[i, :, :] = ddf[i, :, :]

        # Convert from mm/timestep to mm/h (intensity)
        precip_intensity = precip / timestep_hours

        logger.success(
            f"Hyetograph generated: {n_steps} timesteps, "
            f"total depth range=[{np.nanmin(ddf[-1]):.1f}, {np.nanmax(ddf[-1]):.1f}] mm, "
            f"peak intensity range=[{np.nanmin(precip_intensity[0]):.2f}, {np.nanmax(precip_intensity[0]):.2f}] mm/h"
        )

        return times, precip_intensity

    def generate_forcing(
        self,
        duration_hours: int,
        start_time: datetime,
        output_path: str | Path,
        method: Literal["chicago_decreasing"] = "chicago_decreasing",
        timestep_hours: int = 1,
        add_metadata: dict | None = None,
        preload: bool = True,
    ):
        """Generate hyetograph and return as MeteoRaster ready for simulation.

        Convenience method that combines generate(), to_netcdf(), and
        MeteoRaster.from_netcdf() into a single call. This simplifies the
        workflow for design storm simulations.

        Args:
            duration_hours: Total duration of the hyetograph in hours
            start_time: Start datetime for the hyetograph
            output_path: Path for output NetCDF file
            method: Hyetograph construction method (default: "chicago_decreasing")
            timestep_hours: Time step in hours (default: 1)
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
            - By default, data is preloaded into memory for optimal performance
        """
        # Import here to avoid circular dependency
        from mobidic.preprocessing.meteo_raster import MeteoRaster

        # Generate hyetograph
        times, precipitation = self.generate(
            duration_hours=duration_hours,
            start_time=start_time,
            method=method,
            timestep_hours=timestep_hours,
        )

        # Save to NetCDF
        self.to_netcdf(
            output_path=output_path,
            times=times,
            precipitation=precipitation,
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
        add_metadata: dict | None = None,
    ) -> None:
        """Export hyetograph to CF-compliant NetCDF file.

        Creates a NetCDF file compatible with MeteoRaster.from_netcdf() for
        use as meteorological forcing in MOBIDIC simulations.

        Args:
            output_path: Path for output NetCDF file
            times: List of datetime objects for each timestep
            precipitation: 3D array (time, y, x) of precipitation [mm/h]
            add_metadata: Optional dictionary of additional global attributes

        Notes:
            - Output follows CF-1.12 conventions
            - Includes CRS information as grid mapping variable
            - Precipitation units are mm/h (compatible with MeteoRaster)

        Examples:
            >>> times, precip = generator.generate(48, datetime(2023, 11, 1))
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
                "hyetograph_method": "chicago_decreasing",
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
