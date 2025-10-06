"""Demo script for testing GIS data readers with real Arno river basin data.

This script demonstrates how to use the read_raster and read_shapefile
functions from the mobidic package with real GIS data.

"""

from pathlib import Path

import numpy as np
from loguru import logger

from mobidic import read_raster
from mobidic import read_shapefile

from mobidic.utils import configure_logger


def demo_read_rasters():
    """Demonstrate raster reading with example basin data."""
    logger.info("=" * 80)
    logger.info("DEMO: Reading Raster Files")
    logger.info("=" * 80)

    # Define paths to raster files
    raster_dir = Path("examples/Arno/raster")
    raster_files = {
        "dtm": "dtm.tif",
        "flowdir": "flowdir.tif",
        "flowacc": "flowacc.tif",
        "cap": "cap.tif",
        "grav": "grav.tif",
        "ks": "ks.tif",
    }

    # Read and display information for each raster
    for name, filename in raster_files.items():
        filepath = raster_dir / filename

        if not filepath.exists():
            logger.warning(f"Skipping {name}: file not found at {filepath}")
            continue

        logger.info(f"{'-' * 80}")
        logger.info(f"Reading {name.upper()} raster: {filename}")
        logger.info(f"{'-' * 80}")

        raster = read_raster(filepath)

        # Display raster information
        logger.info(f"  Shape: {raster['shape']} (rows × cols)")
        logger.info(f"  CRS: {raster['crs']}")
        logger.info(f"  Resolution: {raster['resolution']} (x, y)")
        logger.info(f"  Bounds: {raster['bounds']}")
        logger.info(f"  NoData value: {raster['nodata']}")
        logger.info(f"  Data type: {raster['data'].dtype}")

        # Calculate statistics (excluding NaN values)
        data = raster["data"]
        valid_data = data[~np.isnan(data)]

        if len(valid_data) > 0:
            logger.info(f"  Valid pixels: {len(valid_data)} / {data.size}")
            logger.info(f"  Min value: {valid_data.min():.6f}")
            logger.info(f"  Max value: {valid_data.max():.6f}")
            logger.info(f"  Mean value: {valid_data.mean():.6f}")
            logger.info(f"  Std dev: {valid_data.std():.6f}")
        else:
            logger.warning("  No valid data found in raster")


def demo_read_shapefile():
    """Demonstrate shapefile reading of river network data."""
    logger.info("=" * 80)
    logger.info("DEMO: Reading River Network Shapefile")
    logger.info("=" * 80)

    # Define path to shapefile
    shapefile_path = Path("examples/Arno/shp/Arno_river_network.shp")

    if not shapefile_path.exists():
        logger.error(f"Shapefile not found: {shapefile_path}")
        return

    logger.info(f"Reading shapefile: {shapefile_path}")

    # Read shapefile
    gdf = read_shapefile(shapefile_path)

    # Display shapefile information
    logger.info(f"{'-' * 80}")
    logger.info("Shapefile Information:")
    logger.info(f"{'-' * 80}")
    logger.info(f"  Number of features (reaches): {len(gdf)}")
    logger.info(f"  CRS: {gdf.crs}")
    logger.info(f"  Geometry type: {gdf.geometry.type.unique()[0]}")
    logger.info(f"  Columns: {list(gdf.columns)}")

    # Display bounds
    bounds = gdf.total_bounds
    logger.info(f"  Bounds (minx, miny, maxx, maxy): {bounds}")

    # Display first few features
    logger.info(f"{'-' * 80}")
    logger.info("Sample Features (first 5):")
    logger.info(f"{'-' * 80}")

    display_columns = [col for col in gdf.columns if col != "geometry"]
    for idx in range(min(5, len(gdf))):
        row = gdf.iloc[idx]
        logger.info(f"    Feature {idx + 1}:")
        for col in display_columns:
            logger.info(f"    {col}: {row[col]}")
        logger.info(f"    Geometry length: {row.geometry.length:.1f} meters")


def main():
    """Run all demos."""
    # Configure logger using centralized configuration
    configure_logger(level="INFO")

    logger.info("=" * 80)
    logger.info("MOBIDIC GIS Reader Demo - Arno River Basin")
    logger.info("=" * 80)

    # Run demos
    demo_read_rasters()
    demo_read_shapefile()

    logger.info("=" * 80)
    logger.info("All demos completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
