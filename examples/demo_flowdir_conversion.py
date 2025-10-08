"""Demo script for converting flow direction rasters to MOBIDIC notation.

This script demonstrates how to:
1. Load configuration to get flow direction file path
2. Read the flow direction raster
3. Convert from Grass/Arc notation to MOBIDIC notation
4. Export the converted raster to a new file

"""

from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from rasterio.transform import Affine

from mobidic import load_config
from mobidic import read_raster
from mobidic.preprocessing.grid_operations import convert_to_mobidic_notation
from mobidic.utils import configure_logger


def export_raster(
    data: np.ndarray,
    output_path: str | Path,
    transform: Affine,
    crs: str,
    nodata: float | None = None,
) -> None:
    """Export a numpy array as a GeoTIFF raster.

    Args:
        data: 2D numpy array to export
        output_path: Path where the raster will be saved
        transform: Affine transformation matrix
        crs: Coordinate reference system
        nodata: Value to use for nodata pixels

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)

    logger.success(f"Exported raster to {output_path}")


def demo_flowdir_conversion(config_path: str | Path):
    """Demonstrate flow direction conversion.

    Args:
        config_path: Path to MOBIDIC configuration file

    """
    logger.info("=" * 80)
    logger.info("DEMO: Flow Direction Conversion")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    logger.success("Configuration loaded successfully")

    # Get flow direction file path from config
    flowdir_path = Path(config.raster_files.flow_dir)
    current_notation = config.raster_settings.flow_dir_type

    logger.info(f"Flow direction file: {flowdir_path}")
    logger.info(f"Current notation: {current_notation}")

    if not flowdir_path.exists():
        logger.error(f"Flow direction file not found: {flowdir_path}")
        return

    # Read the flow direction raster
    logger.info("-" * 80)
    logger.info("Reading flow direction raster...")
    flowdir_raster = read_raster(flowdir_path)

    # Display raster information
    logger.info(f"  Shape: {flowdir_raster['shape']} (rows × cols)")
    logger.info(f"  CRS: {flowdir_raster['crs']}")
    logger.info(f"  Resolution: {flowdir_raster['resolution']} (x, y)")
    logger.info(f"  NoData value: {flowdir_raster['nodata']}")

    # Get the data
    flowdir_data = flowdir_raster["data"]
    valid_data = flowdir_data[~np.isnan(flowdir_data)]

    if len(valid_data) > 0:
        logger.info(f"  Valid pixels: {len(valid_data)} / {flowdir_data.size}")
        logger.info(f"  Unique values: {sorted(np.unique(valid_data).astype(int))}")

    logger.info("-" * 80)
    logger.info(f"Converting from {current_notation} to MOBIDIC notation...")

    # Convert flow direction to MOBIDIC notation
    converted_flowdir = convert_to_mobidic_notation(
        flow_dir=flowdir_data,
        from_notation=current_notation,
    )

    logger.success("Conversion completed successfully")

    # Display conversion statistics
    converted_valid = converted_flowdir[~np.isnan(converted_flowdir)]
    if len(converted_valid) > 0:
        logger.info(f"  Converted unique values: {sorted(np.unique(converted_valid).astype(int))}")

    # Create output path
    output_path = flowdir_path.parent / f"{flowdir_path.stem}_mobidic{flowdir_path.suffix}"

    logger.info("-" * 80)
    logger.info(f"Exporting converted raster to: {output_path}")

    # Export the converted raster
    export_raster(
        data=converted_flowdir,
        output_path=output_path,
        transform=flowdir_raster["transform"],
        crs=flowdir_raster["crs"],
        nodata=flowdir_raster["nodata"],
    )

    logger.info("-" * 80)
    logger.success("Flow direction conversion demo completed!")


def main():
    """Run the demo."""
    # Configure logger
    configure_logger(level="INFO")

    # Path to configuration file
    config_path = Path("examples/Arno/Arno.yaml")

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please ensure Arno.yaml exists in the examples/Arno directory")
        return

    # Run demo
    demo_flowdir_conversion(config_path)


if __name__ == "__main__":
    main()
