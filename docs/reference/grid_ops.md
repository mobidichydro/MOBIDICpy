# Grid Operations

The grid operations module provides functions for processing and transforming gridded raster data, including resolution degradation and flow direction conversions.

## Overview

Grid operations are essential for:

- **Multi-resolution modeling**: Coarsen high-resolution DEMs to computationally manageable resolutions
- **Flow direction processing**: Handle different flow direction notation systems (Grass vs Arc)
- **Preprocessing**: Prepare gridded data for hydrological modeling

All functions handle NaN values appropriately and provide comprehensive logging.

## Functions

### Resolution Degradation

::: mobidic.preprocessing.grid_operations.degrade_raster

::: mobidic.preprocessing.grid_operations.degrade_flow_direction

### Flow Direction Conversion

::: mobidic.preprocessing.grid_operations.convert_to_mobidic_notation

## Examples

### Coarsening a Raster

```python
from mobidic import grid_to_matrix, degrade_raster
import numpy as np

# Read high-resolution DTM (e.g., 10m)
dtm = grid_to_matrix("dtm_10m.tif")

# Degrade to 50m resolution (factor = 5)
dtm_degraded = degrade_raster(
    data=dtm['data'],
    factor=5,
    min_valid_fraction=0.125  # Require at least 1/8 valid cells
)

print(f"Original shape: {dtm['data'].shape}")
print(f"Degraded shape: {dtm_degraded.shape}")
print(f"Original cellsize: {dtm['cellsize']} m")
print(f"New cellsize: {dtm['cellsize'] * 5} m")
```

### Coarsening Flow Direction

```python
from mobidic import grid_to_matrix, degrade_flow_direction

# Read flow direction and accumulation grids
flow_dir_data = grid_to_matrix("flow_direction.tif")
flow_acc_data = grid_to_matrix("flow_accumulation.tif")

# Degrade both grids together
flow_dir_coarse, flow_acc_coarse = degrade_flow_direction(
    flow_dir=flow_dir_data['data'],
    flow_acc=flow_acc_data['data'],
    factor=5,
    min_valid_fraction=0.5
)

print(f"Original shape: {flow_dir_data['data'].shape}")
print(f"Degraded shape: {flow_dir_coarse.shape}")
```

### Converting Flow Direction Notation

```python
from mobidic import grid_to_matrix, convert_to_mobidic_notation

# Read flow direction in Grass notation (1-8)
flow_dir_grass = grid_to_matrix("flow_dir_grass.tif")

# Convert to MOBIDIC notation (used internally by the model)
flow_dir_mobidic = convert_to_mobidic_notation(
    flow_dir=flow_dir_grass['data'],
    from_notation="Grass"
)

# Or convert from Arc notation directly
flow_dir_arc = grid_to_matrix("flow_dir_arc.tif")
flow_dir_mobidic = convert_to_mobidic_notation(
    flow_dir=flow_dir_arc['data'],
    from_notation="Arc"
)
```

## Flow Direction Notations

MOBIDICpy supports three flow direction notation systems:

### Grass Notation (1-8)

Sequential numbering from 1 to 8, starting from East and going counter-clockwise:

```
┌───┬───┬───┐
│ 7 │ 6 │ 5 │
├───┼───┼───┤
│ 8 │ X │ 4 │
├───┼───┼───┤
│ 1 │ 2 │ 3 │
└───┴───┴───┘
```

### Arc Notation (Power of 2)

Powers of 2 from 1 to 128, starting from East and going counter-clockwise:

```
┌─────┬─────┬─────┐
│ 128 │  64 │ 32  │
├─────┼─────┼─────┤
│  1  │  X  │ 16  │
├─────┼─────┼─────┤
│  2  │  4  │  8  │
└─────┴─────┴─────┘
```

### MOBIDIC Notation (1-8)

MOBIDIC uses a transformed version of Grass notation with a 180-degree rotation. This is the notation used internally by the model:

```
┌───┬───┬───┐
│ 3 │ 2 │ 1 │
├───┼───┼───┤
│ 4 │ X │ 8 │
├───┼───┼───┤
│ 5 │ 6 │ 7 │
└───┴───┴───┘
```

**Mapping from Grass to MOBIDIC:**
- Grass 1→5, 2→6, 3→7, 4→8, 5→1, 6→2, 7→3, 8→4

**Direction meanings in MOBIDIC notation:**
- 1: up-right (row -1, col +1)
- 2: up (row -1, col 0)
- 3: up-left (row -1, col -1)
- 4: left (row 0, col -1)
- 5: down-left (row +1, col -1)
- 6: down (row +1, col 0)
- 7: down-right (row +1, col +1)
- 8: right (row 0, col +1)

## Technical Details

### Degradation Algorithm

1. Divides the input grid into blocks of size `factor × factor`
2. For regular rasters: computes mean of valid cells in each block
3. For flow direction: finds the cell with maximum flow accumulation in each block, determines the dominant flow direction
4. Applies `min_valid_fraction` threshold to avoid blocks with too few valid cells

### Flow Direction Degradation

The algorithm preserves drainage patterns by:

1. Finding the fine cell with maximum flow accumulation in each coarse block
2. Determining which coarse neighbor the drainage flows to
3. Assigning the appropriate flow direction code
4. Normalizing flow accumulation by `factor²` to maintain consistent scaling

## Notes

- All functions return new arrays and transforms without modifying inputs
- NaN values are properly propagated and excluded from calculations
- Flow direction values must be in the valid range for the specified notation
- Invalid flow direction values (e.g., not in Grass 1-8 or Arc powers-of-2) are converted to NaN
