"""Demo script for hillslope-reach mapping.

This script demonstrates how to:
1. Process a river network
2. Compute hillslope cells (rasterize reaches)
3. Map hillslope cells to reaches using flow direction
4. Export results
"""

from mobidic import (
    load_config,
    configure_logger,
    process_river_network,
    compute_hillslope_cells,
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
    flowdir_path=config.raster_files.flow_dir,
)
