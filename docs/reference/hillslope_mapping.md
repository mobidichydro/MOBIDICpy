# Hillslope-Reach Mapping

The hillslope-reach mapping module connects hillslope grid cells to river reaches, enabling lateral flow routing from the distributed grid to the channel network.

## Overview

This module performs two key operations:

1. **Rasterization**: Identifies which grid cells are occupied by river reaches
2. **Flow routing**: Traces flow paths from hillslope cells to their corresponding reaches

These functions are essential for distributing lateral inflow from the hillslope water balance to the appropriate river reaches in the channel routing module.

## Functions

### Compute Hillslope Cells

::: mobidic.preprocessing.hillslope_reach_mapping.compute_hillslope_cells

### Map Hillslope to Reach

::: mobidic.preprocessing.hillslope_reach_mapping.map_hillslope_to_reach

## Workflow

The typical workflow for hillslope-reach mapping is:

```python
from mobidic import (
    process_river_network,
    compute_hillslope_cells,
    map_hillslope_to_reach,
)

# Step 1: Process river network
network = process_river_network(
    shapefile_path="data/river_network.shp",
    join_single_tributaries=True,
)

# Step 2: Rasterize reaches onto grid
network = compute_hillslope_cells(
    network=network,
    grid_path="data/flow_direction.tif",
)

# Step 3: Map hillslope cells to reaches
reach_map = map_hillslope_to_reach(
    network=network,
    flowdir_path="data/flow_direction.tif",
    flow_dir_type="Grass",  # or "Arc"
)

# Result: 2D array with reach ID for each cell
print(f"Grid shape: {reach_map.shape}")
print(f"Reaches identified: {len(set(reach_map[reach_map >= 0]))}")
```

## Algorithm Details

### Rasterization Algorithm

`compute_hillslope_cells()` performs the following steps:

1. Reads the reference grid to obtain spatial reference (transform, CRS, dimensions)
2. For each reach in the network:
   - Converts the reach geometry to a list of (x, y) coordinates
   - Transforms geographic coordinates to pixel indices using the grid's affine transform
   - Records all cells intersected by the reach geometry
3. Stores the list of cell indices (as linear indices) in a `hillslope_cells` column

**Linear indexing**: Cells are stored as linear indices computed from (row, col) as `index = row * ncols + col`. This format is efficient for sparse storage.

### Flow Path Tracing Algorithm

`map_hillslope_to_reach()` implements a flow path following algorithm:

1. For each grid cell:
   - If the cell belongs to a reach (has hillslope_cells), assign that reach ID
   - Otherwise, follow the flow direction downstream until reaching a reach
   - Assign the reached reach ID to the original cell
2. Cells that don't reach any reach (e.g., flowing out of basin) are assigned -9999

**Flow direction support**: Both Grass (1-8 notation) and Arc (power-of-2 notation) formats are supported.

## MATLAB Translation

This module translates two MATLAB functions:

- `inter_quote_ret.m` → `compute_hillslope_cells()`
- `hill2chan.m` → `map_hillslope_to_reach()`

The Python implementation provides identical functionality with improved performance through vectorized NumPy operations.

## Return Values

### compute_hillslope_cells()

Returns a modified GeoDataFrame with an additional column:

- `hillslope_cells` (list of int): Linear indices of cells occupied by each reach

### map_hillslope_to_reach()

Returns a 2D numpy array with:

- Values ≥ 0: `mobidic_id` of the reach draining this cell
- Value = -9999: Cell does not drain to any reach (e.g., basin outlet)
- NaN: Nodata cells outside the valid domain

## Performance Considerations

- **Rasterization**: O(n_reaches × avg_reach_length) - Linear in network size
- **Flow tracing**: O(n_cells × avg_path_length) - Can be slow for large grids with long flow paths

For large basins, consider:

1. Using coarser resolution grids
2. Degrading the flow direction grid before mapping
3. Caching the reach_map result for reuse

## Example Output

After processing, the network GeoDataFrame contains:

```python
network[['mobidic_id', 'length_m', 'hillslope_cells']].head()
#    mobidic_id  length_m  hillslope_cells
# 0           0   5432.1   [12045, 12046, 12147, ...]
# 1           1   3210.8   [15234, 15235, ...]
# 2           2   8765.4   [18902, 18903, 19004, ...]
```

And the reach_map can be visualized:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(reach_map, cmap='tab20', interpolation='nearest')
plt.colorbar(label='Reach ID')
plt.title('Hillslope-Reach Mapping')
plt.show()
```
