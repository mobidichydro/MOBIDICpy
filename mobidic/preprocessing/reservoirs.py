"""Reservoir preprocessing for MOBIDIC.

This module handles reservoir data preprocessing including:
1. Reading reservoir polygon features from shapefile
2. Processing stage-storage curves
3. Processing regulation curves and schedules (stage-discharge relationships)
4. Mapping reservoirs to river network and grid cells

Translated from MATLAB: buildgis_mysql_include.m (lines 594-740)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from loguru import logger


@dataclass
class Reservoir:
    """Single reservoir data structure.

    This is similar to the MATLAB 'reserv' structure

    Attributes:
        id: Reservoir ID (from shapefile)
        z_max: Maximum stage [m]
        name: Reservoir name
        basin_pixels: Linear indices of cells in reservoir basin (from polygon)
        inlet_reaches: Reach IDs flowing into reservoir (upstream of outlet)
        outlet_reach: Reach ID where reservoir discharges
        stage_storage_curve: DataFrame with stage-storage curve (without reservoir_id column)
        period_times: Dict with start dates for each regulation period {"000": date, "001": date, ...}
        stage_discharge_h: Dict with stage values [m] {"000": [stage values], "001": [...], ...}
        stage_discharge_q: Dict with discharge values [m³/s] {"000": [discharge values], "001": [...], ...}
        initial_volume: Initial volume [m³]
        date_start: Start date of reservoir operation (pandas datetime)
        geometry: Shapely polygon geometry of reservoir

    Note:
        Dictionary keys are zero-padded strings (not integers) for Parquet compatibility.
    """

    id: int
    z_max: float
    name: str = ""
    basin_pixels: Optional[np.ndarray] = None
    inlet_reaches: Optional[np.ndarray] = None
    outlet_reach: Optional[int] = None
    stage_storage_curve: Optional[pd.DataFrame] = None
    period_times: Optional[dict[str, Any]] = None
    stage_discharge_h: Optional[dict[str, list]] = None
    stage_discharge_q: Optional[dict[str, list]] = None
    initial_volume: Optional[float] = None
    date_start: Optional[pd.Timestamp] = None
    geometry: Optional[Any] = None


class Reservoirs:
    """Container for multiple reservoirs.

    This class holds all reservoir data and provides methods to save/load
    to/from GeoParquet format.

    Attributes:
        reservoirs: List of Reservoir objects
        metadata: Dictionary containing processing metadata
    """

    def __init__(self, reservoirs: list[Reservoir], metadata: Optional[dict[str, Any]] = None):
        """Initialize Reservoirs container.

        Args:
            reservoirs: List of Reservoir objects
            metadata: Optional metadata dictionary
        """
        self.reservoirs = reservoirs
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """Return number of reservoirs."""
        return len(self.reservoirs)

    def __getitem__(self, idx: int) -> Reservoir:
        """Get reservoir by index."""
        return self.reservoirs[idx]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert reservoirs to GeoDataFrame for export.

        Returns:
            GeoDataFrame with reservoir data and polygon geometries
        """
        if not self.reservoirs:
            # Return empty GeoDataFrame with expected schema
            return gpd.GeoDataFrame(
                columns=[
                    "id",
                    "name",
                    "z_max",
                    "basin_pixels",
                    "inlet_reaches",
                    "outlet_reach",
                    "period_times",
                    "stage_discharge_h",
                    "stage_discharge_q",
                    "initial_volume",
                    "stage_storage_curve",
                    "geometry",
                ]
            )

        data = []
        for res in self.reservoirs:
            row = {
                "id": res.id,
                "name": res.name,
                "z_max": res.z_max,
                "basin_pixels": res.basin_pixels.tolist() if res.basin_pixels is not None else None,
                "inlet_reaches": res.inlet_reaches.tolist() if res.inlet_reaches is not None else None,
                "outlet_reach": res.outlet_reach,
                "period_times": res.period_times,
                "stage_discharge_h": res.stage_discharge_h,
                "stage_discharge_q": res.stage_discharge_q,
                "initial_volume": res.initial_volume,
                "stage_storage_curve": (
                    res.stage_storage_curve.to_dict(orient="records") if res.stage_storage_curve is not None else None
                ),
                "geometry": res.geometry,
            }
            data.append(row)

        return gpd.GeoDataFrame(data, crs=self.metadata.get("crs"))

    def save(self, output_path: str | Path, format: str = "parquet") -> None:
        """Save reservoirs to file.

        Args:
            output_path: Path to output file
            format: Output format (only 'parquet' is supported)
        """
        gdf = self.to_geodataframe()

        if format.lower() == "parquet":
            gdf.to_parquet(output_path)
            logger.success(f"Saved {len(self)} reservoirs to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}. Only 'parquet' is supported.")

    @classmethod
    def load(cls, input_path: str | Path) -> "Reservoirs":
        """Load reservoirs from file.

        Args:
            input_path: Path to input Parquet file

        Returns:
            Reservoirs object with loaded data
        """
        input_path = Path(input_path)

        # Read Parquet file
        gdf = gpd.read_parquet(input_path)

        # Convert to Reservoir objects
        reservoirs = []
        for _, row in gdf.iterrows():
            res = Reservoir(
                id=row["id"],
                z_max=row["z_max"],
                basin_pixels=np.array(row["basin_pixels"]) if row["basin_pixels"] is not None else None,
                inlet_reaches=np.array(row["inlet_reaches"]) if row["inlet_reaches"] is not None else None,
                outlet_reach=row["outlet_reach"],
                period_times=row.get("period_times"),
                stage_discharge_h=row.get("stage_discharge_h"),
                stage_discharge_q=row.get("stage_discharge_q"),
                initial_volume=row["initial_volume"],
                geometry=row["geometry"],
                name=row.get("name", ""),
                stage_storage_curve=(
                    pd.DataFrame(row["stage_storage_curve"]) if row.get("stage_storage_curve") is not None else None
                ),
            )
            reservoirs.append(res)

        metadata = {"crs": gdf.crs}
        logger.success(f"Loaded {len(reservoirs)} reservoirs from {input_path}")

        return cls(reservoirs, metadata)


def _rasterize_reservoir_polygon(
    polygon,
    grid_shape: tuple[int, int],
    xllcorner: float,
    yllcorner: float,
    cellsize: float,
) -> np.ndarray:
    """Rasterize reservoir polygon to get basin pixels.

    Identifies all grid cells that intersect (even partially) with the reservoir polygon.

    Args:
        polygon: Shapely polygon geometry
        grid_shape: Shape of grid (nrows, ncols)
        xllcorner: X coordinate of lower-left corner [m]
        yllcorner: Y coordinate of lower-left corner [m]
        cellsize: Grid cell size [m]

    Returns:
        Array of linear indices of cells intersecting the polygon
    """
    from rasterio import Affine

    nrows, ncols = grid_shape

    # Create affine transform
    # Upper-left corner is at (xllcorner, yllcorner + nrows * cellsize)
    transform = Affine.translation(xllcorner, yllcorner + nrows * cellsize) * Affine.scale(cellsize, -cellsize)

    # Rasterize polygon
    mask = rasterio.features.rasterize(
        [(polygon, 1)],
        out_shape=(nrows, ncols),
        transform=transform,
        fill=0,
        all_touched=True,  # Include cells that are partially intersected
        dtype=np.uint8,
    )

    # Get linear indices of cells inside polygon. (Fortran order to match MATLAB)
    basin_pixels = np.where(mask.flatten(order="F") == 1)[0]

    return basin_pixels.astype(int)


def _find_reservoir_reaches(
    reservoir: Reservoir,
    network: gpd.GeoDataFrame,
) -> tuple[np.ndarray, int]:
    """Find inlet and outlet reaches for a reservoir.

    The outlet is the reach with the highest calc_order among those intersecting
    the reservoir polygon. The inlets are the upstream reaches of the outlet.

    Args:
        reservoir: Reservoir object with polygon geometry
        network: River network GeoDataFrame

    Returns:
        Tuple of (inlet_reaches, outlet_reach)
    """
    # Find reaches that intersect the reservoir polygon
    intersecting = network[network.intersects(reservoir.geometry)]

    if len(intersecting) == 0:
        logger.warning(f"No reaches intersect reservoir {reservoir.id}")
        return np.array([], dtype=int), -1

    # Find outlet: reach with highest calc_order
    outlet_idx = intersecting["calc_order"].idxmax()
    outlet_row = network.loc[outlet_idx]
    outlet_reach = outlet_row["mobidic_id"]

    # Find inlets: upstream reaches of the outlet
    inlet_reaches = []
    if pd.notna(outlet_row["upstream_1"]) and outlet_row["upstream_1"] >= 0:
        inlet_reaches.append(outlet_row["upstream_1"])
    if pd.notna(outlet_row["upstream_2"]) and outlet_row["upstream_2"] >= 0:
        inlet_reaches.append(outlet_row["upstream_2"])

    return np.array(inlet_reaches, dtype=int), outlet_reach


def _interpolate_volume_at_stage(
    stage_storage_curve: pd.DataFrame,
    target_stage: float,
) -> float:
    """Interpolate volume from stage-storage curve at a target stage.

    Uses numpy linear interpolation with clamping to curve bounds.

    Args:
        stage_storage_curve: DataFrame with 'stage_m' and 'volume_m3' columns
        target_stage: Target stage elevation [m]

    Returns:
        Interpolated volume [m³], clamped to curve min/max if target is out of bounds
    """
    stages = stage_storage_curve["stage_m"].values
    volumes = stage_storage_curve["volume_m3"].values

    # Sort by stage if not already sorted
    if not np.all(stages[:-1] <= stages[1:]):
        sort_idx = np.argsort(stages)
        stages = stages[sort_idx]
        volumes = volumes[sort_idx]

    # Interpolate with clamping (numpy.interp automatically clamps to bounds)
    interpolated_volume = np.interp(target_stage, stages, volumes)

    # Log if clamping occurred
    if target_stage > stages[-1]:
        logger.debug(
            f"Target stage {target_stage:.2f}m exceeds max curve stage {stages[-1]:.2f}m, "
            f"using max volume {volumes[-1]:.0f}m³"
        )
    elif target_stage < stages[0]:
        logger.debug(
            f"Target stage {target_stage:.2f}m below min curve stage {stages[0]:.2f}m, "
            f"using min volume {volumes[0]:.0f}m³"
        )

    return float(interpolated_volume)


def process_reservoirs(
    shapefile_path: str | Path,
    stage_storage_path: str | Path,
    regulation_curves_path: str | Path,
    regulation_schedule_path: str | Path,
    initial_volumes_path: Optional[str | Path],
    network: gpd.GeoDataFrame,
    grid_shape: tuple[int, int],
    xllcorner: float,
    yllcorner: float,
    cellsize: float,
) -> Reservoirs:
    """Process reservoir data from input files.

    This function orchestrates the complete reservoir preprocessing:
    1. Read reservoir polygons from shapefile
    2. Load stage-storage curves
    3. Load regulation curves
    4. Load regulation schedules
    5. Load initial volumes
    6. Map reservoirs to grid and network

    Translated from MATLAB: buildgis_mysql_include.m (lines 594-740)

    Args:
        shapefile_path: Path to reservoir polygon shapefile
        stage_storage_path: Path to stage-storage CSV
        regulation_curves_path: Path to regulation curves CSV
        regulation_schedule_path: Path to regulation schedule CSV
        initial_volumes_path: Path to CSV with initial volumes (columns: 'reservoir_id', 'volume_m3').
            If None, initial volumes are auto-calculated as 100% capacity (volume at z_max)
        network: Processed river network GeoDataFrame
        grid_shape: Shape of computational grid (nrows, ncols)
        xllcorner: X coordinate of lower-left corner [m]
        yllcorner: Y coordinate of lower-left corner [m]
        cellsize: Grid cell size [m]

    Returns:
        Reservoirs object with processed data
    """
    logger.info("Processing reservoir data")
    logger.info("-" * 80)

    # Read reservoir polygons
    logger.debug(f"Reading reservoir shapefile: {shapefile_path}")
    res_polygons = gpd.read_file(shapefile_path)
    logger.info(f"Found {len(res_polygons)} reservoirs in shapefile")

    if len(res_polygons) == 0:
        logger.warning("No reservoirs found in shapefile")
        return Reservoirs([], metadata={"crs": res_polygons.crs})

    # Read stage-storage curves
    logger.debug(f"Reading stage-storage curves: {stage_storage_path}")
    stage_storage = pd.read_csv(stage_storage_path)

    # Read regulation curves
    logger.debug(f"Reading regulation curves: {regulation_curves_path}")
    regulation_curves = pd.read_csv(regulation_curves_path)

    # Read regulation schedule
    logger.debug(f"Reading regulation schedule: {regulation_schedule_path}")
    regulation_schedule = pd.read_csv(regulation_schedule_path)
    regulation_schedule["start_date"] = pd.to_datetime(regulation_schedule["start_date"])
    regulation_schedule["end_date"] = pd.to_datetime(regulation_schedule["end_date"])

    # Read initial volumes (optional)
    if initial_volumes_path is not None:
        logger.debug(f"Reading initial volumes: {initial_volumes_path}")
        initial_volumes = pd.read_csv(initial_volumes_path)
        volumes_dict = dict(zip(initial_volumes["reservoir_id"], initial_volumes["volume_m3"]))
        logger.info(f"Loaded initial volumes for {len(volumes_dict)} reservoirs from CSV")
    else:
        volumes_dict = {}
        logger.info("No initial volumes CSV provided, will auto-calculate from stage-storage curves")

    # Process each reservoir
    reservoirs = []
    for _, res_row in res_polygons.iterrows():
        res_id = res_row["id"]
        logger.debug(f"Processing reservoir {res_id}: {res_row.get('name', 'Unnamed')}")

        # Get reservoir polygon geometry
        polygon = res_row.geometry

        # Get z_max from shapefile
        z_max = res_row["zmax"]

        # Get stage-storage curve for this reservoir
        ss_curve = stage_storage[stage_storage["reservoir_id"] == res_id].copy()
        ss_curve = ss_curve.sort_values("stage_m")

        if len(ss_curve) == 0:
            logger.warning(f"No stage-storage curve found for reservoir {res_id}, skipping")
            continue

        # Drop reservoir_id column as it's used only for mapping
        ss_curve = ss_curve.drop(columns=["reservoir_id"])

        # Get initial volume: try CSV first, then auto-calculate from curve
        initial_volume = volumes_dict.get(res_id)
        if initial_volume is None:
            # Auto-calculate from stage-storage curve
            if len(ss_curve) > 0:
                initial_volume = _interpolate_volume_at_stage(ss_curve, z_max)
                logger.debug(
                    f"Auto-calculated initial volume for reservoir {res_id}: "
                    f"{initial_volume:.0f}m³ at z_max={z_max:.2f}m"
                )
            else:
                logger.warning(f"No stage-storage curve available for reservoir {res_id}, using 0m³")
                initial_volume = 0.0

        # Get regulation curves for this reservoir
        reg_curves_data = regulation_curves[regulation_curves["reservoir_id"] == res_id]
        reg_names = reg_curves_data["regulation_name"].unique()

        # Get regulation schedule for this reservoir
        reg_schedule = regulation_schedule[regulation_schedule["reservoir_id"] == res_id].copy()
        reg_schedule = reg_schedule.sort_values("start_date")

        # Build regulation curves arrays
        n_periods = len(reg_schedule)
        if n_periods == 0:
            logger.warning(f"No regulation schedule found for reservoir {res_id}, skipping")
            continue

        # Find maximum number of points across all regulation curves
        max_points = 0
        reg_curves_dict = {}
        for reg_name in reg_names:
            curve = reg_curves_data[reg_curves_data["regulation_name"] == reg_name].copy()
            curve = curve.sort_values("stage_m")
            reg_curves_dict[reg_name] = curve
            max_points = max(max_points, len(curve))

        # Build lawH and lawQ arrays (n_periods × max_points)
        lawH = np.full((n_periods, max_points), np.nan)
        lawQ = np.full((n_periods, max_points), np.nan)
        period_times = []

        for i, (_, sched_row) in enumerate(reg_schedule.iterrows()):
            reg_name = sched_row["regulation_name"]
            period_times.append(sched_row["start_date"])

            if reg_name in reg_curves_dict:
                curve = reg_curves_dict[reg_name]
                n_pts = len(curve)
                lawH[i, :n_pts] = curve["stage_m"].values
                lawQ[i, :n_pts] = curve["discharge_m3s"].values
            else:
                logger.warning(f"Regulation curve '{reg_name}' not found for reservoir {res_id}")

        # Rasterize polygon to get basin pixels
        logger.debug(f"Rasterizing reservoir {res_id} polygon")
        basin_pixels = _rasterize_reservoir_polygon(polygon, grid_shape, xllcorner, yllcorner, cellsize)
        logger.debug(f"Reservoir {res_id} basin: {len(basin_pixels)} cells")

        # Create temporary reservoir object for reach finding
        temp_reservoir = Reservoir(id=res_id, z_max=z_max, geometry=polygon)

        # Find inlet and outlet reaches
        logger.debug(f"Finding inlet/outlet reaches for reservoir {res_id}")
        inlet_reaches, outlet_reach = _find_reservoir_reaches(temp_reservoir, network)

        if outlet_reach >= 0:
            logger.debug(f"Reservoir {res_id} outlet reach: {outlet_reach}")
            if len(inlet_reaches) > 0:
                logger.debug(f"Reservoir {res_id} inlet reaches: {inlet_reaches}")
        else:
            logger.warning(f"No outlet reach found for reservoir {res_id}")

        # Convert arrays to dicts for storage (use zero-padded string keys for Parquet compatibility)
        period_times_dict = {f"{i:03d}": pd.Timestamp(t).isoformat() for i, t in enumerate(period_times)}
        stage_discharge_h_dict = {f"{i:03d}": lawH[i, :].tolist() for i in range(n_periods)}
        stage_discharge_q_dict = {f"{i:03d}": lawQ[i, :].tolist() for i in range(n_periods)}

        # Create Reservoir object
        reservoir = Reservoir(
            id=res_id,  # Use original shapefile id
            z_max=z_max,
            basin_pixels=basin_pixels,
            inlet_reaches=inlet_reaches,
            outlet_reach=outlet_reach,
            period_times=period_times_dict,
            stage_discharge_h=stage_discharge_h_dict,
            stage_discharge_q=stage_discharge_q_dict,
            initial_volume=initial_volume,
            geometry=polygon,
            name=res_row.get("name", ""),
            date_start=pd.to_datetime(res_row.get("date_start", None)),
            stage_storage_curve=ss_curve,
        )

        reservoirs.append(reservoir)

    logger.success(f"Processed {len(reservoirs)} reservoirs")

    return Reservoirs(reservoirs, metadata={"crs": res_polygons.crs})
