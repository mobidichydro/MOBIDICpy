"""Demo script for hillslope-reach mapping.

This script demonstrates how to:
1. Process a river network
2. Compute hillslope cells (rasterize reaches)
3. Map hillslope cells to reaches using flow direction
4. Export results
"""

import numpy as np
from pathlib import Path
from mobidic import (
    load_config,
    configure_logger,
    read_raster,
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
    join_single_tributaries=False,
    routing_params={
        "wcel": config.parameters.routing.wcel,
        "Br0": config.parameters.routing.Br0,
        "NBr": config.parameters.routing.NBr,
        "n_Man": config.parameters.routing.n_Man,
    },
)

print(f"\nNetwork processed: {len(network)} reaches")
print(f"Strahler orders: {sorted(network['strahler_order'].unique())}")

# Step 2: Read flow direction and DTM
print("\n" + "=" * 60)
print("STEP 2: Reading raster data")
print("=" * 60)

# Read flow direction
flow_dir_path = config.raster_files.flow_dir
if flow_dir_path:
    print(f"\nReading flow direction: {flow_dir_path}")
    flow_dir = read_raster(flow_dir_path)
    print(f"  Grid shape: {flow_dir['data'].shape}")
    print(f"  Resolution: {flow_dir['resolution']} m")
    print(f"  CRS: {flow_dir['crs']}")
else:
    raise ValueError("Flow direction raster not specified in configuration")

# Read DTM for grid properties
dtm_path = config.raster_files.dtm
if dtm_path:
    print(f"\nReading DTM: {dtm_path}")
    dtm = read_raster(dtm_path)
    print(f"  Grid shape: {dtm['data'].shape}")
    print(f"  Elevation range: {np.nanmin(dtm['data']):.1f} - {np.nanmax(dtm['data']):.1f} m")
else:
    raise ValueError("DTM raster not specified in configuration")

# Step 3: Compute hillslope cells
print("\n" + "=" * 60)
print("STEP 3: Computing hillslope cells")
print("=" * 60)

network = compute_hillslope_cells(
    network,
    transform=dtm["transform"],
    shape=dtm["data"].shape,
)

total_hillslope_cells = sum(len(cells) for cells in network["hillslope_cells"])
print(f"\nTotal hillslope cells: {total_hillslope_cells}")
print(f"Average cells per reach: {total_hillslope_cells / len(network):.1f}")

# Step 4: Map hillslope to reach
print("\n" + "=" * 60)
print("STEP 4: Mapping hillslope to reach")
print("=" * 60)

reach_map = map_hillslope_to_reach(
    flow_dir["data"],
    network,
    flow_dir_type=config.raster_settings.flow_dir_type,
)

# Step 5: Analyze results
print("\n" + "=" * 60)
print("STEP 5: Analyzing results")
print("=" * 60)

# Calculate cell size in m2
cell_area_m2 = dtm["resolution"][0] * dtm["resolution"][1]

# Summary statistics
assigned_cells = np.sum(reach_map >= 0)
unassigned_cells = np.sum(reach_map == -9999)
nodata_cells = np.sum(np.isnan(reach_map))
total_cells = reach_map.size

print("\nGrid summary:")
print(f"  Total cells: {total_cells:,}")
print(f"  Assigned to reaches: {assigned_cells:,} ({100 * assigned_cells / total_cells:.1f}%)")
print(f"  Unassigned: {unassigned_cells:,} ({100 * unassigned_cells / total_cells:.1f}%)")
print(f"  No data: {nodata_cells:,} ({100 * nodata_cells / total_cells:.1f}%)")

# Per-reach statistics
print("\nContributing area per reach:")
print(f"{'Reach ID':>10} {'Cells':>10} {'Area (km²)':>12} {'Order':>8}")
print("-" * 44)

# Sort by Strahler order and then by area (descending)
reach_stats = []
for idx in network.index:
    reach_id = network.loc[idx, "mobidic_id"]
    n_cells = np.sum(reach_map == reach_id)
    area_km2 = n_cells * cell_area_m2 / 1e6
    order = network.loc[idx, "strahler_order"]
    reach_stats.append((reach_id, n_cells, area_km2, order))

# Sort by order (descending) then area (descending)
reach_stats.sort(key=lambda x: (-x[3], -x[2]))

# Print top 10 reaches by area
for i, (reach_id, n_cells, area_km2, order) in enumerate(reach_stats[:10]):
    print(f"{reach_id:>10} {n_cells:>10,} {area_km2:>12.2f} {int(order):>8}")

print(f"\n... ({len(reach_stats) - 10} more reaches)")

# Statistics by Strahler order
print("\nStatistics by Strahler order:")
print(f"{'Order':>8} {'Reaches':>10} {'Total Area (km²)':>18} {'Avg Area (km²)':>16}")
print("-" * 56)

for order in sorted(network["strahler_order"].unique()):
    reaches_in_order = network[network["strahler_order"] == order]
    n_reaches = len(reaches_in_order)
    total_area = 0

    for idx in reaches_in_order.index:
        reach_id = network.loc[idx, "mobidic_id"]
        n_cells = np.sum(reach_map == reach_id)
        total_area += n_cells * cell_area_m2 / 1e6

    avg_area = total_area / n_reaches if n_reaches > 0 else 0
    print(f"{int(order):>8} {n_reaches:>10} {total_area:>18.2f} {avg_area:>16.2f}")

# Step 6: Save results (optional)
print("\n" + "=" * 60)
print("STEP 6: Saving results")
print("=" * 60)

# Create output directory if needed
output_dir = Path("examples/Arno/output")
output_dir.mkdir(exist_ok=True)

# Save reach map as GeoTIFF
try:
    import rasterio

    output_path = output_dir / "reach_map.tif"
    print(f"\nSaving reach map to: {output_path}")

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=reach_map.shape[0],
        width=reach_map.shape[1],
        count=1,
        dtype=reach_map.dtype,
        crs=dtm["crs"],
        transform=dtm["transform"],
        nodata=-9999,
        compress="lzw",
    ) as dst:
        dst.write(reach_map, 1)

    print("  Successfully saved reach map")

except Exception as e:
    print(f"  Warning: Could not save reach map: {e}")

# Save network with hillslope cells
try:
    from mobidic import export_network

    output_path = output_dir / "network_with_hillslope_cells.parquet"
    print(f"\nSaving network to: {output_path}")

    export_network(network, output_path, format="parquet")
    print("  Successfully saved network with hillslope cells")

except Exception as e:
    print(f"  Warning: Could not save network: {e}")


print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
