"""I/O functions for saving and loading preprocessed GIS data.

This module provides functions to save/load preprocessed data to/from files:
- Gridded data: NetCDF4 format (xarray-compatible)
- River network: GeoParquet format (default) or Shapefile
"""

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import geopandas as gpd
import xarray as xr
from loguru import logger

if TYPE_CHECKING:
    from mobidic.preprocessing.preprocessor import GISData


def save_gisdata(gisdata: "GISData", output_path: str | Path) -> None:
    """Save preprocessed gridded data to NetCDF file.

    This function saves all grids (DTM, flow direction, soil parameters, etc.),
    metadata, and hillslope-reach mapping to a single NetCDF file using xarray.

    TODO: Check grid orientations (flip y-axis when writing/reading for better visualization in external tools)
    TODO: Check nodata handling (set nodata values in grids and metadata)

    Args:
        gisdata: GISData object containing preprocessed data
        output_path: Path to output NetCDF file

    Examples:
        >>> from mobidic import run_preprocessing, load_config
        >>> config = load_config("Arno.yaml")
        >>> gisdata = run_preprocessing(config)
        >>> from mobidic.preprocessing.io import save_gisdata
        >>> save_gisdata(gisdata, "Arno_gisdata.nc")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving gridded data to NetCDF: {output_path}")

    # Get dimensions from metadata
    nrows, ncols = gisdata.metadata["shape"]
    resolution = gisdata.metadata["resolution"]

    # Create coordinate arrays
    # Use cell centers following MOBIDIC convention (xllcorner and yllcorner are already at cell centers)
    x = gisdata.metadata["xllcorner"] + np.arange(ncols) * resolution[0]
    y = gisdata.metadata["yllcorner"] + np.arange(nrows) * resolution[1]

    # Create xarray Dataset
    data_vars = {}

    # Add all grids as data variables
    for name, grid in gisdata.grids.items():
        data_vars[name] = (["y", "x"], grid)

    # Add hillslope-reach mapping
    data_vars["hillslope_reach_map"] = (["y", "x"], gisdata.hillslope_reach_map)

    # Create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": (["x"], x),
            "y": (["y"], y),
        },
    )

    # Add metadata as attributes
    ds.attrs["basin_id"] = gisdata.config.basin.id
    ds.attrs["paramset_id"] = gisdata.config.basin.paramset_id
    ds.attrs["basin_baricenter_lon"] = gisdata.config.basin.baricenter.lon
    ds.attrs["basin_baricenter_lat"] = gisdata.config.basin.baricenter.lat
    ds.attrs["crs"] = str(gisdata.metadata["crs"])
    ds.attrs["resolution_x"] = resolution[0]
    ds.attrs["resolution_y"] = resolution[1]
    ds.attrs["nodata_value"] = np.nan
    ds.attrs["resample_factor"] = gisdata.config.simulation.resample
    ds.attrs["flow_dir_notation"] = "MOBIDIC"
    ds.attrs["conventions"] = "CF-1.12"
    ds.attrs["title"] = f"MOBIDIC preprocessed GIS data for {gisdata.config.basin.id}"
    ds.attrs["institution"] = "University of Florence, Department of Civil and Environmental Engineering"
    ds.attrs["source"] = "MOBIDICpy preprocessing"
    ds.attrs["history"] = f"Created by MOBIDICpy preprocessing for basin {gisdata.config.basin.id}"

    # Add variable metadata
    ds["dtm"].attrs = {
        "long_name": "Digital Terrain Model",
        "units": "m",
        "description": "Elevation above sea level",
    }
    ds["flow_dir"].attrs = {
        "long_name": "Flow Direction",
        "units": "1",
        "description": "Flow direction in MOBIDIC notation (1-8)",
        "notation": "MOBIDIC: 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE, 8=E",
    }
    ds["flow_acc"].attrs = {
        "long_name": "Flow Accumulation",
        "units": "cells",
        "description": "Number of upstream cells draining to each cell",
    }
    ds["Wc0"].attrs = {
        "long_name": "Capillary Water Holding Capacity",
        "units": "mm",
        "description": "Maximum water holding capacity in soil small pores",
    }
    ds["Wg0"].attrs = {
        "long_name": "Gravitational Water Holding Capacity",
        "units": "mm",
        "description": "Maximum water holding capacity in soil large pores",
    }
    ds["ks"].attrs = {
        "long_name": "Hydraulic Conductivity",
        "units": "mm/h",
        "description": "Soil hydraulic conductivity",
    }
    ds["hillslope_reach_map"].attrs = {
        "long_name": "Hillslope to Reach Mapping",
        "units": "1",
        "description": "MOBIDIC ID of downstream reach for each cell",
        "missing_value": -9999,
    }

    # Add optional parameter metadata
    if "kf" in gisdata.grids:
        ds["kf"].attrs = {
            "long_name": "Aquifer Conductivity",
            "units": "m/s",
            "description": "Aquifer hydraulic conductivity",
        }
    if "CH" in gisdata.grids:
        ds["CH"].attrs = {
            "long_name": "Heat Exchange Coefficient",
            "units": "1",
            "description": "Turbulent exchange coefficient for heat",
        }
    if "Alb" in gisdata.grids:
        ds["Alb"].attrs = {
            "long_name": "Albedo",
            "units": "1",
            "description": "Surface albedo",
        }
    if "gamma" in gisdata.grids:
        ds["gamma"].attrs = {
            "long_name": "Percolation Coefficient",
            "units": "1/s",
            "description": "Percolation coefficient from capillary to gravitational reservoir",
        }
    if "kappa" in gisdata.grids:
        ds["kappa"].attrs = {
            "long_name": "Adsorption Coefficient",
            "units": "1/s",
            "description": "Adsorption coefficient from gravitational to capillary reservoir",
        }
    if "beta" in gisdata.grids:
        ds["beta"].attrs = {
            "long_name": "Hypodermic Flow Coefficient",
            "units": "1/s",
            "description": "Hypodermic (subsurface) flow coefficient",
        }
    if "alpha" in gisdata.grids:
        ds["alpha"].attrs = {
            "long_name": "Hillslope Flow Coefficient",
            "units": "1/s",
            "description": "Surface hillslope flow coefficient",
        }
    if "Ma" in gisdata.grids:
        ds["Ma"].attrs = {
            "long_name": "Artesian Aquifer Mask",
            "units": "1",
            "description": "Binary mask defining artesian aquifer extension (0=no, 1=yes)",
        }
    if "Mf" in gisdata.grids:
        ds["Mf"].attrs = {
            "long_name": "Freatic Aquifer Mask",
            "units": "1",
            "description": "Binary mask defining freatic aquifer extension (0=no, 1=yes)",
        }

    # Save to NetCDF
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding, engine="netcdf4")

    logger.success(f"Gridded data saved to {output_path}")
    logger.debug(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def save_network(network: gpd.GeoDataFrame, output_path: str | Path, format: str = "parquet") -> None:
    """Save processed river network to file.

    Saves the river network to either GeoParquet (default, recommended) or Shapefile format.
    GeoParquet is more efficient, has no field name limitations, and preserves data types better.

    Args:
        network: Processed river network GeoDataFrame
        output_path: Path to output file
        format: Output format, either 'parquet' (default) or 'shapefile'

    Raises:
        ValueError: If format is not supported

    Examples:
        >>> from mobidic import process_river_network
        >>> network = process_river_network("river_network.shp")
        >>> from mobidic.preprocessing.io import save_network
        >>> save_network(network, "river_network.parquet")  # Default: parquet
        >>> save_network(network, "river_network.shp", format="shapefile")  # Shapefile
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        logger.info(f"Saving river network to GeoParquet: {output_path}")
        network.to_parquet(output_path)
        logger.success(f"Network saved to GeoParquet: {output_path}")
        logger.debug(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    elif format == "shapefile":
        import warnings

        logger.info(f"Saving river network to Shapefile: {output_path}")
        # Suppress shapefile field name truncation warnings (10 char limit is expected)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Column names longer than 10 characters.*")
            warnings.filterwarnings("ignore", message=".*Normalized/laundered field name.*")
            network.to_file(output_path)
        logger.success(f"Network saved to Shapefile: {output_path}")
        logger.debug(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'shapefile'")


def load_gisdata(gisdata_path: str | Path, network_path: str | Path) -> "GISData":
    """Load preprocessed GIS data from NetCDF and GeoParquet files.

    Args:
        gisdata_path: Path to gridded data NetCDF file
        network_path: Path to river network GeoParquet file

    Returns:
        GISData object containing loaded data

    Raises:
        FileNotFoundError: If either file does not exist

    Examples:
        >>> from mobidic.preprocessing.io import load_gisdata
        >>> gisdata = load_gisdata("Arno_gisdata.nc", "Arno_network.parquet")
    """
    from mobidic.preprocessing.preprocessor import GISData

    gisdata_path = Path(gisdata_path)
    network_path = Path(network_path)

    # Check files exist
    if not gisdata_path.exists():
        raise FileNotFoundError(f"GIS data file not found: {gisdata_path}")
    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    logger.info(f"Loading gridded data from NetCDF: {gisdata_path}")

    # Load NetCDF dataset
    ds = xr.open_dataset(gisdata_path)

    # Extract grids
    grids = {}
    grid_vars = [
        "dtm",
        "flow_dir",
        "flow_acc",
        "Wc0",
        "Wg0",
        "ks",
        "kf",
        "CH",
        "Alb",
        "Ma",
        "Mf",
        "gamma",
        "kappa",
        "beta",
        "alpha",
    ]

    for var in grid_vars:
        if var in ds:
            grids[var] = ds[var].values

    # Extract hillslope-reach mapping
    hillslope_reach_map = ds["hillslope_reach_map"].values

    # Extract metadata
    metadata = {
        "shape": (len(ds.y), len(ds.x)),
        "resolution": (ds.attrs.get("resolution_x"), ds.attrs.get("resolution_y")),
        "crs": ds.attrs.get("crs"),
        "nodata": ds.attrs.get("nodata_value", -9999.0),
        "bounds": None,  # Not stored, can reconstruct if needed
        "transform": None,  # Not stored, can reconstruct if needed
    }

    # Close dataset
    ds.close()

    logger.success(f"Gridded data loaded: {len(grids)} grids")

    # Load river network
    network = load_network(network_path)

    # Create GISData object
    # Note: config will be None since we only have partial config from attributes
    gisdata = GISData(
        grids=grids,
        metadata=metadata,
        network=network,
        hillslope_reach_map=hillslope_reach_map,
        config=None,  # type: ignore
    )

    return gisdata


def load_network(network_path: str | Path) -> gpd.GeoDataFrame:
    """Load processed river network from GeoParquet file.

    Loads river network data from GeoParquet format only (the recommended format).
    For loading shapefiles, use geopandas.read_file() directly.

    Args:
        network_path: Path to network GeoParquet file (.parquet)

    Returns:
        GeoDataFrame with river network

    Raises:
        FileNotFoundError: If network file does not exist
        ValueError: If file is not a .parquet file

    Examples:
        >>> from mobidic.preprocessing.io import load_network
        >>> network = load_network("river_network.parquet")
    """
    network_path = Path(network_path)

    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    if network_path.suffix != ".parquet":
        raise ValueError(
            f"Only .parquet files supported. Got: {network_path.suffix}. "
            "Use geopandas.read_file() to load other formats."
        )

    logger.info(f"Loading river network from GeoParquet: {network_path}")
    network = gpd.read_parquet(network_path)
    logger.success(f"River network loaded: {len(network)} reaches")

    return network
