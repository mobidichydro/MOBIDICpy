"""Example script demonstrating the complete MOBIDIC preprocessing workflow.

This script shows how to:
1. Load a MOBIDIC configuration from YAML
2. Run the complete preprocessing pipeline (grids, network, reservoirs)
3. Save preprocessed data to files
4. Load preprocessed data from files
5. Display summary statistics

Usage:
    python examples/run_preprocessing.py

Requirements:
    - YAML configuration file (e.g., sample_config.yaml)
    - GIS input data (rasters and shapefiles) as specified in the config
    - Optional: reservoir data (shapefile, stage-storage curves, regulation rules)
"""

from pathlib import Path
from mobidic import load_config, run_preprocessing, GISData, configure_logger

# Get directory containing this script
SCRIPT_DIR = Path(__file__).parent
EXAMPLE_DIR = SCRIPT_DIR / "Arno"

# Configure logging
configure_logger(level="INFO")


def main():
    """Run the complete preprocessing workflow."""

    print("=" * 80)
    print("MOBIDIC PREPROCESSING EXAMPLE")
    print("=" * 80)

    # Step 1: Load configuration
    config_path = EXAMPLE_DIR / "Arno.yaml"
    if not config_path.exists():
        return
    config = load_config(config_path)

    # Step 2: Run preprocessing
    gisdata = run_preprocessing(config)

    # Step 3: Save preprocessed data
    gisdata_path = Path(config.paths.gisdata)
    network_path = Path(config.paths.network)
    reservoirs_path = Path(config.paths.reservoirs) if config.paths.reservoirs else None

    # Create output directories if needed
    gisdata_path.parent.mkdir(parents=True, exist_ok=True)
    network_path.parent.mkdir(parents=True, exist_ok=True)
    if reservoirs_path:
        reservoirs_path.parent.mkdir(parents=True, exist_ok=True)

    gisdata.save(gisdata_path, network_path, reservoirs_path)

    # Step 4: Load preprocessed data (demonstration)
    loaded_gisdata = GISData.load(gisdata_path, network_path, reservoirs_path)

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
    print(f"  - Resolution: {list(int(x) for x in loaded_gisdata.metadata['resolution'])} m")
    print(f"  - CRS: {loaded_gisdata.metadata['crs']}")
    print()

    print("River network statistics:")
    print(f"  - Total reaches: {len(loaded_gisdata.network)}")
    print(f"  - Strahler orders: {sorted(loaded_gisdata.network['strahler_order'].unique().tolist())}")
    print(f"  - Total length: {loaded_gisdata.network['length_m'].sum() / 1000:.1f} km")
    print(f"  - Mean width: {loaded_gisdata.network['width_m'].mean():.2f} m")
    print()

    if loaded_gisdata.reservoirs is not None and len(loaded_gisdata.reservoirs) > 0:
        print("Reservoir statistics:")
        print(f"  - Total reservoirs: {len(loaded_gisdata.reservoirs)}")
        for res in loaded_gisdata.reservoirs:
            print(f"  - {res.name}:")
            print(f"      Maximum stage: {res.z_max:.2f} m")
            print(f"      Initial volume: {res.initial_volume*1e-6:.0f} Mm³")
            print(f"      Number of regulation schedules: {len(res.period_times) if res.period_times else 0}")
            if res.basin_pixels is not None:
                print(f"      Basin cells: {len(res.basin_pixels)}")
            if res.outlet_reach is not None and res.outlet_reach >= 0:
                print(f"      Outlet reach: {res.outlet_reach}")
            if res.inlet_reaches is not None and len(res.inlet_reaches) > 0:
                print(f"      Inlet reaches: {list(int(x) for x in res.inlet_reaches)}")
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
