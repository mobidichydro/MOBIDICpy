"""Example script demonstrating the complete MOBIDIC preprocessing workflow.

This script shows how to:
1. Load a MOBIDIC configuration from YAML
2. Run the complete preprocessing pipeline
3. Save preprocessed data to files
4. Load preprocessed data from files

Usage:
    python examples/run_preprocessing.py

Requirements:
    - YAML configuration file (e.g., sample_config.yaml)
    - GIS input data (rasters and shapefiles) as specified in the config
"""

from pathlib import Path
from mobidic import load_config, run_preprocessing, GISData, configure_logger

# Configure logging
configure_logger(level="INFO")


def main():
    """Run the complete preprocessing workflow."""

    print("=" * 80)
    print("MOBIDIC PREPROCESSING EXAMPLE")
    print("=" * 80)

    # Step 1: Load configuration
    config_path = Path("examples/Arno/Arno.yaml")
    if not config_path.exists():
        return
    config = load_config(config_path)

    # Step 2: Run preprocessing
    gisdata = run_preprocessing(config)

    # Step 3: Save preprocessed data
    gisdata_path = Path(config.paths.gisdata)
    network_path = Path(config.paths.network)

    # Create output directories if needed
    gisdata_path.parent.mkdir(parents=True, exist_ok=True)
    network_path.parent.mkdir(parents=True, exist_ok=True)

    gisdata.save(gisdata_path, network_path)

    # Step 4: Load preprocessed data (demonstration)
    loaded_gisdata = GISData.load(gisdata_path, network_path)

    # Step 5: Display summary statistics
    print("=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print()

    print(f"Basin: {config.basin.id}")
    print(f"Parameter set: {config.basin.paramset_id}")
    print()

    print("Grid information:")
    print(f"  - Shape: {loaded_gisdata.metadata['shape']} (rows x cols)")
    print(f"  - Resolution: {loaded_gisdata.metadata['resolution']} m")
    print(f"  - CRS: {loaded_gisdata.metadata['crs']}")
    print()

    print("River network statistics:")
    print(f"  - Total reaches: {len(loaded_gisdata.network)}")
    print(f"  - Strahler orders: {sorted(loaded_gisdata.network['strahler_order'].unique().tolist())}")
    print(f"  - Total length: {loaded_gisdata.network['length_m'].sum() / 1000:.1f} km")
    print(f"  - Mean width: {loaded_gisdata.network['width_m'].mean():.2f} m")
    print()

    print("Grid variables:")
    for var_name in sorted(loaded_gisdata.grids.keys()):
        grid = loaded_gisdata.grids[var_name]
        import numpy as np

        print(f"  - {var_name}: min={np.nanmin(grid):.3f}, max={np.nanmax(grid):.3f}, mean={np.nanmean(grid):.3f}")
    print()

    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
