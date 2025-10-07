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

::: mobidic.preprocessing.grid_operations.convert_flow_direction

## Examples

### Coarsening a Raster

```python
from mobidic import read_raster, degrade_raster

# Read high-resolution DTM (e.g., 10m)
dtm = read_raster("dtm_10m.tif")

# Degrade to 50m resolution (factor = 5)
dtm_50m = degrade_raster(
    dtm['data'],
    dtm['transform'],
    degradation_factor=5,
    min_valid_fraction=0.125  # Require at least 1/8 valid cells
)

print(f"Original shape: {dtm['data'].shape}")
print(f"Degraded shape: {dtm_50m['data'].shape}")
print(f"New resolution: {dtm_50m['resolution']}")
```

### Coarsening Flow Direction

```python
from mobidic import read_raster, degrade_flow_direction

# Read flow direction and accumulation grids
flow_dir = read_raster("flow_direction.tif")
flow_acc = read_raster("flow_accumulation.tif")

# Degrade both grids together
degraded = degrade_flow_direction(
    flow_dir['data'],
    flow_acc['data'],
    flow_dir['transform'],
    degradation_factor=5,
    min_valid_fraction=0.5
)

# Access degraded grids
flow_dir_coarse = degraded['flow_direction']
flow_acc_coarse = degraded['flow_accumulation']
new_transform = degraded['transform']
```

### Converting Flow Direction Notation

```python
from mobidic import read_raster, convert_flow_direction

# Read flow direction in Arc notation (powers of 2)
flow_dir_arc = read_raster("flow_dir_arc.tif")

# Convert to Grass notation (1-8)
flow_dir_grass = convert_flow_direction(
    flow_dir_arc['data'],
    from_notation="arc",
    to_notation="grass"
)

# Convert back to Arc notation
flow_dir_arc_2 = convert_flow_direction(
    flow_dir_grass,
    from_notation="grass",
    to_notation="arc"
)
```

## Flow Direction Notations

MOBIDICpy supports two common flow direction notation systems:

### Grass Notation (1-8)

Sequential numbering from 1 to 8, clockwise from East:

```
┌───┬───┬───┐
│ 8 │ 1 │ 2 │
├───┼───┼───┤
│ 7 │ X │ 3 │
├───┼───┼───┤
│ 6 │ 5 │ 4 │
└───┴───┴───┘
```

### Arc Notation (Power of 2)

Powers of 2 from 1 to 128, clockwise from East:

```
┌─────┬─────┬─────┐
│ 128 │  1  │  2  │
├─────┼─────┼─────┤
│ 64  │  X  │  4  │
├─────┼─────┼─────┤
│ 32  │ 16  │  8  │
└─────┴─────┴─────┘
```

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
