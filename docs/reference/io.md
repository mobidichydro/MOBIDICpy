# Data I/O and GISData Container

This module provides consolidated I/O for preprocessed MOBIDIC data and a container class for managing the complete preprocessed dataset.

## Overview

After preprocessing GIS data and processing the river network, MOBIDICpy consolidates all data into a single `GISData` object that can be saved to and loaded from disk. This approach:

- **Simplifies data management**: Single object contains all preprocessed data
- **Ensures consistency**: Grids, network, and metadata stay synchronized
- **Enables caching**: Save expensive preprocessing results for reuse
- **Facilitates sharing**: Package preprocessed data for model runs

## Classes

### GISData Container

::: mobidic.preprocessing.io.GISData

## Functions

### Save and Load Functions

::: mobidic.preprocessing.io.save_gisdata

::: mobidic.preprocessing.io.load_gisdata

::: mobidic.preprocessing.io.save_network

::: mobidic.preprocessing.io.load_network

## Usage Examples

### Example 1: Complete Preprocessing Workflow

```python
from mobidic import load_config, run_preprocessing, GISData

# Load configuration
config = load_config("config.yaml")

# Run preprocessing
gisdata = run_preprocessing(config)

# Save preprocessed data
gisdata.save(
    gisdata_path="output/gisdata.nc",
    network_path="output/network.parquet"
)

# Later, load preprocessed data
loaded_gisdata = GISData.load(
    gisdata_path="output/gisdata.nc",
    network_path="output/network.parquet"
)

# Access components
print(f"Grid variables: {list(loaded_gisdata.grids.keys())}")
print(f"Network reaches: {len(loaded_gisdata.network)}")
print(f"Grid shape: {loaded_gisdata.metadata['shape']}")
```

### Example 2: Working with GISData

```python
from mobidic import GISData
import numpy as np

# Create GISData object
gisdata = GISData()

# Add grid data
gisdata.grids['dtm'] = np.random.rand(100, 100)
gisdata.grids['flow_dir'] = np.random.randint(1, 9, (100, 100))
gisdata.grids['ks'] = np.random.rand(100, 100) * 10

# Add network
gisdata.network = network_gdf  # GeoDataFrame from process_river_network()

# Add metadata
gisdata.metadata = {
    'shape': (100, 100),
    'resolution': 100.0,
    'crs': 'EPSG:32632',
    'transform': affine_transform,
}

# Save
gisdata.save("output/gisdata.nc", "output/network.parquet")
```

### Example 3: Saving Network Only

```python
from mobidic import process_river_network, save_network, load_network

# Process network
network = process_river_network(
    shapefile_path="data/river_network.shp",
    join_single_tributaries=True,
)

# Save as GeoParquet (recommended)
save_network(network, "output/network.parquet", format="parquet")

# Or save as Shapefile
save_network(network, "output/network.shp", format="shapefile")

# Load network later
loaded_network = load_network("output/network.parquet")
```

## File Formats

### NetCDF for Grid Data

Grid data is saved in NetCDF4 format with:

**Structure:**
- Each grid variable is a 2D data variable (y, x dimensions)
- Coordinate variables for x and y
- Comprehensive metadata (CRS, transform, resolution)

**Compression:**
- zlib compression (level 4 by default)
- Chunking optimized for spatial access patterns

**Advantages:**
- Self-describing format with embedded metadata
- Efficient compression for large grids
- CF-compliant for interoperability
- Supports NaN for nodata values

### GeoParquet for Network Data

River network data is saved in GeoParquet format with:

**Advantages:**
- Very fast read/write performance
- Excellent compression ratios
- Preserves all attribute types (int, float, list, etc.)
- Native support for complex geometries
- Column-oriented storage for efficient queries

**Requirements:**
- Requires `pyarrow` package: `pip install pyarrow`

**Fallback:**
- If pyarrow not available, can use Shapefile format
- Shapefile has limitations (attribute names, data types)

## Data Consistency

The `GISData` class ensures consistency between components:

### Grid Validation

All grids must have the same shape:

```python
gisdata = GISData()
gisdata.grids['dtm'] = np.zeros((100, 100))
gisdata.grids['ks'] = np.zeros((100, 150))  # Different shape - will raise error on save
```

### Metadata Requirements

Required metadata fields:
- `shape`: Tuple (nrows, ncols)
- `resolution`: Float or tuple (x_res, y_res)
- `crs`: String (e.g., "EPSG:32632")
- `transform`: Affine transform object

### Network Validation

The network must be a GeoDataFrame with specific required columns:
- `mobidic_id`: Integer reach identifiers
- `geometry`: LineString geometries
- `upstream_1`, `upstream_2`, `downstream`: Topology references
- `strahler_order`, `calc_order`: Ordering information
- `length_m`, `width_m`: Geometric parameters

## Performance Considerations

### File Sizes

Typical file sizes for a medium-sized basin (1000×1000 grid, 1000 reaches):

| Component | Format | Compressed Size | Uncompressed Size |
|-----------|--------|-----------------|-------------------|
| Grid data (5 variables) | NetCDF | ~5-10 MB | ~20-40 MB |
| River network | GeoParquet | ~1-2 MB | ~5-10 MB |
| River network | Shapefile | ~3-5 MB | ~3-5 MB |

### Read/Write Speed

Approximate times on modern hardware:

| Operation | Format | Time |
|-----------|--------|------|
| Write grids | NetCDF | ~1-2 seconds |
| Read grids | NetCDF | ~0.5-1 seconds |
| Write network | GeoParquet | ~0.1-0.3 seconds |
| Read network | GeoParquet | ~0.1-0.2 seconds |
| Write network | Shapefile | ~1-3 seconds |
| Read network | Shapefile | ~0.5-1 seconds |

**Recommendation**: Use GeoParquet for best performance.

## Spatial Reference Handling

### CRS Representation

The CRS is stored in multiple places:

1. **NetCDF grids**: As global attribute `crs` (WKT or EPSG code)
2. **GeoDataFrame**: Native CRS property
3. **Metadata dict**: As string for convenience

These should all be consistent and are validated on save/load.

### Affine Transform

The affine transform maps pixel coordinates to geographic coordinates:

```python
from affine import Affine

# Example: 100m resolution, origin at (600000, 4800000)
transform = Affine(100.0, 0.0, 600000.0,
                   0.0, -100.0, 4900000.0)

# Transform pixel (row, col) to (x, y)
x, y = transform * (col, row)
```

The transform is stored in the metadata dict and embedded in NetCDF grid variables.

## Integration with Preprocessing

The preprocessing workflow automatically creates and populates GISData:

```python
from mobidic import load_config, run_preprocessing

config = load_config("config.yaml")

# This function internally:
# 1. Creates GISData object
# 2. Reads all raster files specified in config
# 3. Processes river network
# 4. Computes hillslope-reach mapping
# 5. Populates grids, network, and metadata
gisdata = run_preprocessing(config)

# Save for later use
gisdata.save(config.paths.gisdata, config.paths.network)
```

## API Stability

The GISData format and I/O functions are designed for long-term stability:

- NetCDF and GeoParquet are standard formats
- Metadata schema is versioned
- Future versions will maintain backward compatibility
- Migration tools will be provided if format changes are needed

## Error Handling

All I/O functions provide comprehensive error handling:

- **File not found**: Clear messages with expected paths
- **Format errors**: Validation of required fields and data types
- **CRS mismatches**: Warnings when CRS differs between components
- **Corruption detection**: Checksums and validation on load
- **Missing dependencies**: Helpful messages for optional packages (pyarrow)

All operations use loguru for structured logging.
