# GIS Data I/O

The GIS I/O module provides functions for reading geospatial data in various formats, including raster (GeoTIFF) and vector (Shapefile) files.

## Overview

MOBIDICpy uses industry-standard libraries for geospatial data handling:

- **Rasterio** for raster data (GeoTIFF, ASCII grids, etc.)
- **GeoPandas** for vector data (Shapefiles, GeoJSON, etc.)

All functions provide comprehensive logging and error handling, converting nodata values to NaN for consistent numerical processing.

## Functions

### Raster I/O

::: mobidic.preprocessing.gis_reader.read_raster

### Vector I/O

::: mobidic.preprocessing.gis_reader.read_shapefile

## Examples

### Reading a Raster

```python
from mobidic import read_raster

# Read a Digital Terrain Model
dtm = read_raster("path/to/dtm.tif")

# Access raster data and metadata
print(f"Shape: {dtm['data'].shape}")
print(f"Resolution: {dtm['resolution']}")
print(f"CRS: {dtm['crs']}")
print(f"Bounds: {dtm['bounds']}")

# Access the data array (nodata converted to NaN)
elevation = dtm['data']
mean_elevation = elevation[~np.isnan(elevation)].mean()
print(f"Mean elevation: {mean_elevation:.2f} m")
```

### Reading a Shapefile

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
    target_crs="EPSG:32632"  # WGS84 / UTM zone 32N
)
```

## Return Values

### Raster Data

`read_raster()` returns a dictionary with:

- `data` (numpy.ndarray): 2D array of raster values, nodata converted to NaN
- `transform` (affine.Affine): Affine transformation matrix
- `crs` (rasterio.crs.CRS): Coordinate reference system
- `bounds` (tuple): Bounding box (left, bottom, right, top)
- `resolution` (tuple): Pixel size (x_res, y_res)

### Vector Data

`read_shapefile()` returns a `geopandas.GeoDataFrame` with geometry and attribute columns.

## Error Handling

Both functions provide detailed error messages and logging:

- **File not found**: `FileNotFoundError` with clear message
- **Invalid format**: `RasterioIOError` or `RuntimeError` with details
- **CRS issues**: Warnings when CRS is missing or reprojection fails
- **Logging**: All operations logged using loguru (DEBUG, INFO, SUCCESS, ERROR levels)
