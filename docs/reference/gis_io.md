# GIS data I/O

The GIS I/O module provides functions for reading geospatial data in various formats, including raster (GeoTIFF) and vector (Shapefile) files.

## Overview

MOBIDICpy uses industry-standard libraries for geospatial data handling:

- **Rasterio** for raster data (GeoTIFF)
- **GeoPandas** for vector data (Shapefiles, GeoJSON, etc.)

All functions provide comprehensive logging and error handling, converting nodata values to NaN for consistent numerical processing.

## Functions

### Raster I/O

::: mobidic.preprocessing.gis_reader.grid_to_matrix

### Vector I/O

::: mobidic.preprocessing.gis_reader.read_shapefile

## Examples

### Reading a Raster

```python
from mobidic import grid_to_matrix
import numpy as np

# Read a Digital Terrain Model
dtm = grid_to_matrix("path/to/dtm.tif")

# Access raster data and metadata
print(f"Shape: {dtm['data'].shape}")
print(f"Cell size: {dtm['cellsize']} m")
print(f"Lower-left corner: ({dtm['xllcorner']}, {dtm['yllcorner']})")
print(f"CRS: {dtm['crs']}")

# Access the data array (nodata converted to NaN)
elevation = dtm['data']
mean_elevation = np.nanmean(elevation)
print(f"Mean elevation: {mean_elevation:.2f} m")
```

### Reading a shapefile

```python
from mobidic import read_shapefile

# Read river network shapefile
network = read_shapefile("path/to/river_network.shp")

# Access as GeoDataFrame
print(f"Number of reaches: {len(network)}")
print(f"CRS: {network.crs}")
print(network.head())

# Reproject to a different CRS
network_utm = read_shapefile(
    "path/to/river_network.shp",
    crs="EPSG:32632"  # WGS84 / UTM zone 32N
)
```

## Return values

### Raster data

`grid_to_matrix()` returns a dictionary with:

- `data` (numpy.ndarray): 2D array of raster values, nodata converted to NaN (flipped vertically to match MATLAB convention)
- `xllcorner` (float): X coordinate of lower-left corner (cell center)
- `yllcorner` (float): Y coordinate of lower-left corner (cell center)
- `cellsize` (float): Cell size in map units
- `crs` (rasterio.crs.CRS): Coordinate reference system

### Vector data

`read_shapefile()` returns a `geopandas.GeoDataFrame` with geometry and attribute columns.

## Error handling

Both functions provide detailed error messages and logging:

- **File not found**: `FileNotFoundError` with clear message
- **Invalid format**: `RasterioIOError` or `RuntimeError` with details
- **CRS issues**: Warnings when CRS is missing or reprojection fails
- **Logging**: All operations logged using loguru (DEBUG, INFO, SUCCESS, ERROR levels)
