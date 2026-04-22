"""Main preprocessing orchestrator for MOBIDIC GIS data.

This module coordinates the complete preprocessing workflow:
1. Read configuration
2. Load and process raster grids (DTM, flow direction, soil parameters, etc.)
3. Process river network
4. Compute hillslope-reach mapping
5. Save consolidated preprocessed data
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np
import geopandas as gpd
from loguru import logger

from mobidic import __version__
from mobidic.config.schema import MOBIDICConfig
from mobidic.preprocessing.gis_reader import grid_to_matrix
from mobidic.preprocessing.grid_operations import (
    decimate_raster,
    decimate_flow_direction,
    convert_to_mobidic_notation,
)
from mobidic.preprocessing.river_network import process_river_network
from mobidic.preprocessing.hillslope_reach_mapping import (
    compute_hillslope_cells,
    map_hillslope_to_reach,
)
from mobidic.preprocessing.reservoirs import Reservoirs, process_reservoirs


def _calculate_slopes(dtm: np.ndarray, grid_size: float, pmin: float = 1e-5) -> np.ndarray:
    """Calculate slope from DTM using finite differences.

    Translated from MATLAB: pendenzez.m

    Args:
        dtm: Digital terrain model [m]
        grid_size: Grid cell size [m]
        pmin: Minimum slope [-]

    Returns:
        Slope grid [-]
    """
    n, m = dtm.shape
    slopes = np.full_like(dtm, np.nan)

    # Pad DTM with NaN around edges
    dtm_padded = np.full((n + 2, m + 2), np.nan)
    dtm_padded[1 : n + 1, 1 : m + 1] = dtm

    # Calculate slopes for each cell
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if np.isfinite(dtm_padded[i, j]):
                # Find valid neighboring cells in x direction
                if np.isfinite(dtm_padded[i, j + 1]):
                    jp = j + 1
                else:
                    jp = j

                if np.isfinite(dtm_padded[i, j - 1]):
                    jm = j - 1
                else:
                    jm = j

                # Find valid neighboring cells in y direction
                if np.isfinite(dtm_padded[i + 1, j]):
                    ip = i + 1
                else:
                    ip = i

                if np.isfinite(dtm_padded[i - 1, j]):
                    im = i - 1
                else:
                    im = i

                # Calculate x gradient
                dzx = dtm_padded[i, jp] - dtm_padded[i, jm]
                if abs(dzx) < pmin * 2 * grid_size:
                    dzx = pmin
                else:
                    dzx = dzx / grid_size / (jp - jm)

                # Calculate y gradient
                dzy = dtm_padded[ip, j] - dtm_padded[im, j]
                if abs(dzy) < pmin * 2 * grid_size:
                    dzy = pmin
                else:
                    dzy = dzy / grid_size / (ip - im)

                # Combined slope
                slopes[i - 1, j - 1] = np.sqrt(dzx**2 + dzy**2)

    return slopes


class GISData:
    """Container for preprocessed GIS data.

    This class holds all preprocessed spatial data including grids, river network,
    reservoirs, and hillslope-reach mapping. It provides methods to save/load
    consolidated preprocessed data.

    Attributes:
        grids: Dictionary of 2D numpy arrays containing raster data
        metadata: Dictionary containing grid metadata (transform, CRS, resolution, etc.)
        network: GeoDataFrame with processed river network
        reservoirs: Reservoirs object with reservoir data (optional)
        hillslope_reach_map: 2D array mapping each cell to its downstream reach
        config: MOBIDIC configuration used for preprocessing
    """

    def __init__(
        self,
        grids: dict[str, np.ndarray],
        metadata: dict[str, Any],
        network: gpd.GeoDataFrame,
        hillslope_reach_map: np.ndarray,
        config: MOBIDICConfig,
        reservoirs: Optional[Reservoirs] = None,
    ):
        """Initialize GISData container.

        Args:
            grids: Dictionary of 2D numpy arrays with grid data
            metadata: Dictionary with grid metadata
            network: Processed river network GeoDataFrame
            hillslope_reach_map: 2D array with reach assignments
            config: MOBIDIC configuration
            reservoirs: Optional Reservoirs object
        """
        self.grids = grids
        self.metadata = metadata
        self.network = network
        self.reservoirs = reservoirs
        self.hillslope_reach_map = hillslope_reach_map
        self.config = config

    def save(
        self, gisdata_path: str | Path, network_path: str | Path, reservoirs_path: Optional[str | Path] = None
    ) -> None:
        """Save preprocessed data to files.

        Args:
            gisdata_path: Path to save gridded data (NetCDF format)
            network_path: Path to save river network (GeoParquet format)
            reservoirs_path: Optional path to save reservoirs (GeoParquet format)
        """
        from mobidic.preprocessing.io import save_gisdata, save_network, save_reservoirs

        save_gisdata(self, gisdata_path)
        save_network(self.network, network_path, format="parquet")

        if self.reservoirs is not None and reservoirs_path is not None:
            save_reservoirs(self.reservoirs, reservoirs_path, format="parquet")

    @classmethod
    def load(
        cls, gisdata_path: str | Path, network_path: str | Path, reservoirs_path: Optional[str | Path] = None
    ) -> "GISData":
        """Load preprocessed data from files.

        Args:
            gisdata_path: Path to gridded data file (NetCDF)
            network_path: Path to river network file (GeoParquet)
            reservoirs_path: Optional path to reservoirs file (GeoParquet)

        Returns:
            GISData object with loaded data
        """
        from mobidic.preprocessing.io import load_gisdata, load_reservoirs

        gisdata = load_gisdata(gisdata_path, network_path)

        # Load reservoirs if path provided
        if reservoirs_path is not None:
            reservoirs_path = Path(reservoirs_path)
            if reservoirs_path.exists():
                gisdata.reservoirs = load_reservoirs(reservoirs_path)
            else:
                logger.warning(f"Reservoirs file not found: {reservoirs_path}")
                gisdata.reservoirs = None

        return gisdata


def run_preprocessing(config: MOBIDICConfig) -> GISData:
    """Run complete preprocessing workflow.

    This function orchestrates the entire preprocessing pipeline:
    1. Load raster grids (DTM, flow direction, soil parameters, etc.)
    2. Apply grid decimation if needed (decimation factor > 1)
    3. Convert flow direction to MOBIDIC notation
    4. Process river network (topology, Strahler ordering, routing parameters)
    5. Compute hillslope cells for each reach
    6. Map hillslope cells to reaches

    Args:
        config: MOBIDIC configuration with preprocessing settings

    Returns:
        GISData object containing all preprocessed spatial data

    Examples:
        >>> from mobidic import load_config, run_preprocessing
        >>> config = load_config("Arno.yaml")
        >>> gisdata = run_preprocessing(config)
        >>> gisdata.save("Arno_gisdata.nc", "Arno_network.parquet")
    """
    logger.info("=" * 80)
    logger.info(f"MOBIDICpy v{__version__} - PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Basin: {config.basin.id}")
    logger.info(f"Parameter set: {config.basin.paramset_id}")
    logger.info("")

    # Step 1: Load required raster grids
    logger.info("Step 1/7: Loading raster grids")
    logger.info("-" * 80)

    grids = {}
    metadata = {}

    # Load DTM (required)
    logger.debug(f"Loading DTM from {config.raster_files.dtm}")
    dtm_result = grid_to_matrix(config.raster_files.dtm)
    grids["dtm"] = dtm_result["data"]
    xllcorner = dtm_result["xllcorner"]
    yllcorner = dtm_result["yllcorner"]
    cellsize = dtm_result["cellsize"]
    crs = dtm_result["crs"]

    # Store metadata extracted from DTM
    metadata["xllcorner"] = xllcorner
    metadata["yllcorner"] = yllcorner
    metadata["cellsize"] = cellsize
    metadata["shape"] = grids["dtm"].shape
    metadata["resolution"] = (cellsize, cellsize)
    metadata["nodata"] = np.nan
    metadata["crs"] = crs

    # Load flow direction (required)
    logger.debug(f"Loading flow direction from {config.raster_files.flow_dir}")
    flow_dir_result = grid_to_matrix(config.raster_files.flow_dir)
    grids["flow_dir"] = flow_dir_result["data"]

    # Load flow accumulation (required)
    logger.debug(f"Loading flow accumulation from {config.raster_files.flow_acc}")
    flow_acc_result = grid_to_matrix(config.raster_files.flow_acc)
    flow_acc_data = flow_acc_result["data"]
    if np.any((flow_acc_data[~np.isnan(flow_acc_data)] < 1)):
        flow_acc_data[~np.isnan(flow_acc_data)] += 1
        logger.info("Flow accumulation values were < 1 and have been incremented by 1.")
    grids["flow_acc"] = flow_acc_data

    # Get index of the cell with maximum flow accumulation
    max_acc_idx = np.unravel_index(np.nanargmax(flow_acc_data), flow_acc_data.shape)

    # Set outlet value to -1 in the flow direction grid
    grids["flow_dir"][max_acc_idx] = -1

    # Load soil parameters (required)
    logger.debug(f"Loading Wc0 from {config.raster_files.Wc0}")
    wc0_result = grid_to_matrix(config.raster_files.Wc0)
    grids["Wc0"] = wc0_result["data"] * 0.001  # Convert from mm to m

    logger.debug(f"Loading Wg0 from {config.raster_files.Wg0}")
    wg0_result = grid_to_matrix(config.raster_files.Wg0)
    grids["Wg0"] = wg0_result["data"] * 0.001  # Convert from mm to m

    logger.debug(f"Loading ks from {config.raster_files.ks}")
    ks_result = grid_to_matrix(config.raster_files.ks)
    grids["ks"] = ks_result["data"] * 0.001 / 3600  # Convert from mm/h to m/s

    # Check ranges of ks values (ks_min and ks_max in mm/h from config)
    ks_min = config.parameters.soil.ks_min
    ks_max = config.parameters.soil.ks_max
    if ks_min is not None:
        ks_min = ks_min * 0.001 / 3600
        if np.any(grids["ks"] < ks_min):
            logger.warning(f"Some ks values are below the minimum threshold: {ks_min} m/s")

    if ks_max is not None:
        ks_max = ks_max * 0.001 / 3600
        if np.any(grids["ks"] > ks_max):
            logger.warning(f"Some ks values are above the maximum threshold: {ks_max} m/s")

    if ks_min is not None and ks_max is not None:
        grids["ks"] = np.clip(grids["ks"], ks_min, ks_max)

    # Load optional raster files
    optional_rasters = {
        "kf": config.raster_files.kf,
        "CH": config.raster_files.CH,
        "Alb": config.raster_files.Alb,
        "Ma": config.raster_files.Ma,
        "Mf": config.raster_files.Mf,
        "gamma": config.raster_files.gamma,
        "kappa": config.raster_files.kappa,
        "beta": config.raster_files.beta,
        "alpha": config.raster_files.alpha,
    }

    for name, path in optional_rasters.items():
        if path is not None:
            logger.debug(f"Loading {name} from {path}")
            result = grid_to_matrix(path)
            grids[name] = result["data"]
        else:
            logger.debug(f"Using default value for {name} (no raster provided)")
            # Create grid filled with default parameter value
            default_value = _get_default_parameter_value(name, config)
            grids[name] = np.full_like(grids["dtm"], default_value, dtype=float)

    # Corine Land Cover (CLC) raster: optional categorical grid; when absent the
    # simulation falls back to the scalar parameters.soil.Kc. We keep it out of
    # the numeric optional_rasters loop because it carries class codes and must
    # not be filled with a default flux value nor averaged during decimation.
    if config.raster_files.CLC is not None:
        logger.debug(f"Loading CLC from {config.raster_files.CLC}")
        clc_result = grid_to_matrix(config.raster_files.CLC)
        grids["CLC"] = clc_result["data"]
    else:
        logger.debug("No CLC raster provided; Kc will default to parameters.soil.Kc")

    # Calculate alpsur (surface routing parameter based on slope)
    # From buildgis_mysql_include.m lines 647-651
    logger.debug("Calculating alpsur from DTM slopes")
    slopes = _calculate_slopes(grids["dtm"], cellsize, pmin=1e-5)
    alpsur = np.sqrt(slopes)
    alpsur = alpsur / np.nanmean(alpsur)
    grids["alpsur"] = alpsur

    logger.success(f"Loaded {len(grids)} raster grids. Shape: {grids['dtm'].shape}, cellsize: {cellsize} m")
    logger.info("")

    # Step 2: Apply grid decimation if needed
    logger.info("Step 2/7: Applying grid decimation")
    logger.info("-" * 80)

    decimation_factor = config.simulation.decimation

    if decimation_factor > 1:
        logger.info(f"Decimating grids by factor {decimation_factor}")

        # Decimate most grids using simple averaging. flow_dir/flow_acc follow
        # a dedicated path below; CLC holds discrete class codes so averaging
        # is skipped and the upper-left sub-cell of each block is taken.
        categorical_grids = {"CLC"}
        for name in grids.keys():
            if name in ("flow_dir", "flow_acc"):
                continue
            if name in categorical_grids:
                logger.debug(f"Decimating {name} (nearest)")
                grids[name] = grids[name][::decimation_factor, ::decimation_factor]
                continue
            logger.debug(f"Decimating {name}")
            grids[name] = decimate_raster(grids[name], decimation_factor)

        # Decimate flow direction and accumulation
        logger.debug("Decimating flow_dir and flow_acc")
        grids["flow_dir"], grids["flow_acc"] = decimate_flow_direction(
            grids["flow_dir"], grids["flow_acc"], decimation_factor
        )

        # Update metadata
        metadata["shape"] = grids["dtm"].shape
        metadata["resolution"] = (
            metadata["resolution"][0] * decimation_factor,
            metadata["resolution"][1] * decimation_factor,
        )

        logger.success(f"Grid decimation complete. New shape: {metadata['shape']}")
    else:
        logger.info("No grid decimation applied (decimation factor = 1)")

    logger.info("")

    # Step 3: Convert flow direction to MOBIDIC notation
    logger.info("Step 3/7: Converting flow direction to MOBIDIC notation")
    logger.info("-" * 80)

    flow_dir_type = config.raster_settings.flow_dir_type

    grids["flow_dir"] = convert_to_mobidic_notation(grids["flow_dir"], from_notation=flow_dir_type)

    logger.success("Flow direction conversion complete")
    logger.info("")

    # Step 4: Process river network
    logger.info("Step 4/7: Processing river network")
    logger.info("-" * 80)

    shapefile_path = config.vector_files.river_network.shp

    routing_params = {
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    }

    network = process_river_network(
        shapefile_path=shapefile_path,
        join_single_tributaries=True,
        routing_params=routing_params,
    )

    logger.info("")

    # Step 5: Compute hillslope cells for each reach
    logger.info("Step 5/7: Computing hillslope cells")
    logger.info("-" * 80)

    network = compute_hillslope_cells(
        network=network,
        grid_array=grids["flow_dir"],
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        cellsize=metadata["resolution"][0],
        densify_step=10.0,
    )

    logger.info("")

    # Step 6: Map hillslope cells to reaches
    logger.info("Step 6/7: Mapping hillslope to reaches")
    logger.info("-" * 80)

    hillslope_reach_map = map_hillslope_to_reach(
        network=network,
        flowdir_array=grids["flow_dir"],
    )

    logger.info("")

    # Step 7: Process reservoirs (optional)
    logger.info("Step 7/7: Processing reservoirs")
    logger.info("-" * 80)

    reservoirs = None
    if config.parameters.reservoirs.res_shape is not None:
        try:
            reservoirs = process_reservoirs(
                shapefile_path=config.parameters.reservoirs.res_shape,
                stage_storage_path=config.parameters.reservoirs.stage_storage,
                regulation_curves_path=config.parameters.reservoirs.regulation_curves,
                regulation_schedule_path=config.parameters.reservoirs.regulation_schedule,
                initial_volumes_path=config.initial_conditions.reservoir_volumes,
                network=network,
                grid_shape=grids["dtm"].shape,
                xllcorner=xllcorner,
                yllcorner=yllcorner,
                cellsize=metadata["resolution"][0],
            )
        except Exception as e:
            logger.error(f"Failed to process reservoirs: {e}")
            logger.warning("Continuing without reservoirs")
            reservoirs = None
    else:
        logger.info("No reservoirs configured, skipping")

    gisdata = GISData(
        grids=grids,
        metadata=metadata,
        network=network,
        hillslope_reach_map=hillslope_reach_map,
        config=config,
        reservoirs=reservoirs,
    )

    logger.info("")
    logger.success("Preprocessing completed successfully")
    logger.info("")

    return gisdata


def _get_default_parameter_value(param_name: str, config: MOBIDICConfig) -> float:
    """Get default parameter value from configuration.

    Args:
        param_name: Name of parameter (e.g., 'kf', 'CH', 'Alb')
        config: MOBIDIC configuration

    Returns:
        Default parameter value
    """
    param_map = {
        "kf": config.parameters.soil.kf,
        "CH": config.parameters.energy.CH,
        "Alb": config.parameters.energy.Alb,
        "gamma": config.parameters.soil.gamma,
        "kappa": config.parameters.soil.kappa,
        "beta": config.parameters.soil.beta,
        "alpha": config.parameters.soil.alpha,
        "Ma": 0.0,  # Binary mask, default 0 (no artesian aquifer)
        "Mf": 1.0,  # Binary mask, default 1 (freatic aquifer everywhere)
    }

    return param_map.get(param_name, 0.0)
