"""State file I/O for MOBIDIC simulations.

This module provides functions to save and load simulation state variables
(soil water content, discharge, etc.) in NetCDF format.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mobidic import __version__


def save_state(
    state: "SimulationState",  # noqa: F821
    output_path: str | Path,
    time: datetime,
    grid_metadata: dict,
    network_size: int,
    add_metadata: dict | None = None,
) -> None:
    """
    Save simulation state to NetCDF file.

    Args:
        state: SimulationState object containing state variables
        output_path: Path to output NetCDF file
        time: Current simulation time
        grid_metadata: Dictionary with grid metadata (shape, resolution, crs, etc.)
        network_size: Number of reaches in network
        add_metadata: Additional global attributes (optional)

    Examples:
        >>> from mobidic import Simulation
        >>> sim = Simulation(gisdata, forcing, config)
        >>> # After running simulation
        >>> from mobidic.io import save_state
        >>> save_state(sim.state, "state_2020-01-01.nc", datetime(2020, 1, 1),
        ...            sim.gisdata.metadata, len(sim.network))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving simulation state to NetCDF: {output_path}")

    # Get grid dimensions
    nrows, ncols = grid_metadata["shape"]
    resolution = grid_metadata["resolution"]
    xllcorner = grid_metadata["xllcorner"]
    yllcorner = grid_metadata["yllcorner"]

    # Create coordinate arrays
    x = xllcorner + np.arange(ncols) * resolution[0]
    y = yllcorner + np.arange(nrows) * resolution[1]
    reaches = np.arange(network_size)

    # Create data variables
    data_vars = {
        "Wc": (["y", "x"], state.wc),
        "Wg": (["y", "x"], state.wg),
        "Ws": (["y", "x"], state.ws),
        "discharge": (["reach"], state.discharge),
    }

    # Add plant water if present
    if state.wp is not None:
        data_vars["Wp"] = (["y", "x"], state.wp)

    # Add grid mapping variable for CRS (CF-1.12 compliance)
    data_vars["crs"] = ([], 0)

    # Create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": (["x"], x),
            "y": (["y"], y),
            "reach": (["reach"], reaches),
            "time": time,
        },
    )

    # Add grid mapping attributes (CF-1.12 compliance)
    crs_string = str(grid_metadata.get("crs", ""))
    ds["crs"].attrs = {
        "grid_mapping_name": "spatial_ref",
        "crs_wkt": crs_string,
        "spatial_ref": crs_string,
    }

    # Add coordinate attributes
    ds["x"].attrs = {
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
        "units": "m",
        "axis": "X",
    }
    ds["y"].attrs = {
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "units": "m",
        "axis": "Y",
    }
    ds["reach"].attrs = {
        "long_name": "reach index",
        "description": "MOBIDIC reach ID (mobidic_id)",
        "units": "1",
    }
    ds["time"].attrs = {
        "long_name": "simulation time",
    }

    # Add variable metadata
    ds["Wc"].attrs = {
        "long_name": "Capillary Water Content",
        "units": "m",
        "description": "Water held by capillary forces in soil",
        "grid_mapping": "crs",
    }
    ds["Wg"].attrs = {
        "long_name": "Gravitational Water Content",
        "units": "m",
        "description": "Drainable water in soil large pores",
        "grid_mapping": "crs",
    }
    ds["Ws"].attrs = {
        "long_name": "Surface Water Content",
        "units": "m",
        "description": "Water in surface depressions",
        "grid_mapping": "crs",
    }
    ds["discharge"].attrs = {
        "long_name": "River Discharge",
        "units": "m3 s-1",
        "description": "Discharge at downstream end of each reach",
    }

    if "Wp" in ds:
        ds["Wp"].attrs = {
            "long_name": "Plant/Canopy Water Content",
            "units": "m",
            "description": "Water intercepted by vegetation canopy",
            "grid_mapping": "crs",
        }

    # Add global attributes
    ds.attrs["Conventions"] = "CF-1.12"
    ds.attrs["title"] = "MOBIDIC simulation state"
    ds.attrs["source"] = "MOBIDICpy simulation"
    ds.attrs["history"] = f"Created by MOBIDICpy version {__version__} at {datetime.now().isoformat()}"
    ds.attrs["simulation_time"] = time.isoformat()

    # Add custom metadata if provided
    if add_metadata:
        ds.attrs.update(add_metadata)

    # Save to NetCDF with compression
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars if var != "crs"}
    ds.to_netcdf(output_path, encoding=encoding, engine="netcdf4")

    logger.success(f"State saved to {output_path}")
    logger.debug(f"File size: {output_path.stat().st_size / 1024:.2f} KB")


def load_state(
    input_path: str | Path,
    network_size: int,
) -> tuple["SimulationState", datetime, dict]:  # noqa: F821
    """
    Load simulation state from NetCDF file.

    Args:
        input_path: Path to input NetCDF file
        network_size: Expected number of reaches in network

    Returns:
        Tuple of (state, time, metadata) where:
            - state: SimulationState object
            - time: Simulation time from file
            - metadata: Grid metadata dictionary

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If file format is invalid

    Examples:
        >>> from mobidic.io import load_state
        >>> state, time, metadata = load_state("state_2020-01-01.nc", 1235)
        >>> print(f"Loaded state at {time}")
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"State file not found: {input_path}")

    logger.info(f"Loading simulation state from NetCDF: {input_path}")

    # Load dataset
    ds = xr.open_dataset(input_path)

    # Extract state variables
    wc = ds["Wc"].values
    wg = ds["Wg"].values
    ws = ds["Ws"].values
    discharge = ds["discharge"].values

    # Check array shapes
    if discharge.shape[0] != network_size:
        logger.warning(
            f"Network size mismatch: expected {network_size}, got {discharge.shape[0]}. "
            "This may cause issues if network has changed."
        )

    # Extract plant water if present
    wp = ds["Wp"].values if "Wp" in ds else None

    # Extract time
    time = pd.Timestamp(ds["time"].values).to_pydatetime()

    # Extract grid metadata
    nrows, ncols = wc.shape
    resolution_x = float(ds.x.values[1] - ds.x.values[0]) if len(ds.x) > 1 else float(ds.attrs.get("resolution_x", 1.0))
    resolution_y = float(ds.y.values[1] - ds.y.values[0]) if len(ds.y) > 1 else float(ds.attrs.get("resolution_y", 1.0))
    xllcorner = float(ds.x.values[0])
    yllcorner = float(ds.y.values[0])
    crs = ds["crs"].attrs.get("crs_wkt", "") if "crs" in ds else ""

    metadata = {
        "shape": (nrows, ncols),
        "resolution": (resolution_x, resolution_y),
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "crs": crs,
    }

    # Close dataset
    ds.close()

    # Import here to avoid circular import
    from mobidic.core.simulation import SimulationState

    state = SimulationState(wc, wg, wp, ws, discharge)

    logger.success(f"State loaded: time={time.isoformat()}, grid={nrows}x{ncols}, reaches={len(discharge)}")

    return state, time, metadata
