"""Demo script for hillslope-reach mapping.

This script demonstrates how to:
1. Process a river network
2. Compute hillslope cells (rasterize reaches)
3. Map hillslope cells to reaches using flow direction
4. Export results
"""

import rasterio
import numpy as np
from mobidic import (
    load_config,
    configure_logger,
    process_river_network,
    compute_hillslope_cells,
    map_hillslope_to_reach,
)

# Configuration
config = load_config("examples/Arno/Arno.yaml")

# Configure logger
configure_logger(level="INFO")

# Step 1: Process river network
print("=" * 60)
print("STEP 1: Processing river network")
print("=" * 60)

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

# Step 2: Compute hillslope cells
print("\n" + "=" * 60)
print("STEP 2: Computing hillslope cells")
print("=" * 60)

network = compute_hillslope_cells(
    network=network,
    grid_path=config.raster_files.flow_dir,
)

# Step 3: Map hillslope cells to reaches
print("\n" + "=" * 60)
print("STEP 3: Mapping hillslope cells to reaches")
print("=" * 60)

reach_map = map_hillslope_to_reach(
    network=network,
    flowdir_path=config.raster_files.flow_dir,
    flow_dir_type=config.raster_settings.flow_dir_type,
)

print(f"\nReach map shape: {reach_map.shape}")
print(f"Unique reaches assigned: {len(set(reach_map[~np.isnan(reach_map)]))}")
print(f"Cells assigned to reaches: {np.sum(~np.isnan(reach_map))}")
print(f"Unassigned cells: {np.sum(reach_map == -9999)}")

# Step 4: Export results
print("\n" + "=" * 60)
print("STEP 4: Exporting results")
print("=" * 60)

reach_map_output_path = "examples/Arno/output/reach_map.tif"

with rasterio.open(config.raster_files.flow_dir) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open(reach_map_output_path, "w", **profile) as dst:
        dst.write(reach_map.astype(rasterio.float32), 1)
print(f"\nReach map exported to: {reach_map_output_path}")
