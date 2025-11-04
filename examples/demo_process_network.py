"""Test script to process the Arno river network."""

from pathlib import Path
from mobidic import load_config, process_river_network, save_network

# Get directory containing this script
SCRIPT_DIR = Path(__file__).parent
EXAMPLE_DIR = SCRIPT_DIR / "Arno"

# Load configuration
config = load_config(EXAMPLE_DIR / "Arno.yaml")

# Process river network
print("Processing Arno river network...")
network = process_river_network(
    shapefile_path=config.vector_files.river_network.shp,
    join_single_tributaries=True,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    },
)

print("\nNetwork summary:")
print(f"  Total reaches: {len(network)}")
print(f"  Strahler orders: {sorted(network['strahler_order'].unique())}")
print(f"  Max calculation order: {network['calc_order'].max()}")
print(f"  Total network length: {network['length_m'].sum():.1f} m")
print(f"  Mean channel width: {network['width_m'].mean():.2f} m")

# Export to parquet
print(f"\nExporting to {config.paths.network}...")
try:
    save_network(network, config.paths.network, format="parquet")
    print("Successfully exported to Parquet format")
except ImportError as e:
    print(f"Warning: {e}")
    print("Skipping Parquet export. Install pyarrow to enable Parquet support.")

print("\nDone!")
