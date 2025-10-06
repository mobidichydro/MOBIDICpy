"""Tests for GIS data readers."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from mobidic.preprocessing.gis_reader import read_raster, read_shapefile


class TestReadRaster:
    """Tests for read_raster function."""

    def test_read_raster_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_raster("nonexistent_file.tif")

    def test_read_raster_invalid_band(self, tmp_path):
        """Test that ValueError is raised for invalid band number."""
        # Create a simple test raster with 1 band
        filepath = tmp_path / "test.tif"
        data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.translation(10.0, 50.0) * Affine.scale(1, -1)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Try to read band 2 (which doesn't exist)
        with pytest.raises(ValueError, match="Invalid band number"):
            read_raster(filepath, band=2)

    def test_read_raster_success(self, tmp_path):
        """Test successful reading of a raster file."""
        # Create a test raster
        filepath = tmp_path / "test.tif"
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        transform = Affine.translation(10.0, 50.0) * Affine.scale(0.1, -0.1)
        nodata = -9999.0

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)

        # Read the raster
        result = read_raster(filepath)

        # Assertions
        assert result["data"].shape == (10, 10)
        assert np.array_equal(result["data"], data)
        assert result["transform"] == transform
        assert str(result["crs"]) == "EPSG:4326"
        assert result["nodata"] == nodata
        assert result["shape"] == (10, 10)
        assert result["resolution"] == (0.1, 0.1)

    def test_read_raster_with_nodata(self, tmp_path):
        """Test that nodata values are replaced with NaN."""
        # Create a test raster with nodata values
        filepath = tmp_path / "test.tif"
        data = np.array([[1, 2, -9999], [4, 5, 6], [-9999, 8, 9]], dtype=np.float32)
        transform = Affine.translation(10.0, 50.0) * Affine.scale(1, -1)
        nodata = -9999.0

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=3,
            width=3,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)

        # Read the raster
        result = read_raster(filepath)

        # Check that nodata values are NaN
        assert np.isnan(result["data"][0, 2])
        assert np.isnan(result["data"][2, 0])
        assert result["data"][0, 0] == 1
        assert result["data"][1, 1] == 5

    def test_read_raster_custom_nodata(self, tmp_path):
        """Test reading with custom nodata value."""
        # Create a test raster
        filepath = tmp_path / "test.tif"
        data = np.array([[1, 2, -999], [4, 5, 6], [-999, 8, 9]], dtype=np.float32)
        transform = Affine.translation(10.0, 50.0) * Affine.scale(1, -1)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=3,
            width=3,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Read with custom nodata value
        result = read_raster(filepath, nodata_value=-999)

        # Check that custom nodata values are NaN
        assert np.isnan(result["data"][0, 2])
        assert np.isnan(result["data"][2, 0])
        assert result["nodata"] == -999


class TestReadShapefile:
    """Tests for read_shapefile function."""

    def test_read_shapefile_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_shapefile("nonexistent_file.shp")

    def test_read_shapefile_success(self, tmp_path):
        """Test successful reading of a shapefile."""
        # Create a simple GeoDataFrame
        from shapely.geometry import Point

        gdf_original = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
            },
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )

        # Save to shapefile
        filepath = tmp_path / "test.shp"
        gdf_original.to_file(filepath)

        # Read the shapefile
        gdf_read = read_shapefile(filepath)

        # Assertions
        assert isinstance(gdf_read, gpd.GeoDataFrame)
        assert len(gdf_read) == 3
        assert list(gdf_read["id"]) == [1, 2, 3]
        assert list(gdf_read["name"]) == ["A", "B", "C"]
        assert gdf_read.crs == gdf_original.crs

    def test_read_shapefile_with_reprojection(self, tmp_path):
        """Test reading shapefile with CRS reprojection."""
        from shapely.geometry import Point

        # Create a GeoDataFrame in WGS84
        gdf_original = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(10, 45), Point(11, 46)],
            crs="EPSG:4326",
        )

        # Save to shapefile
        filepath = tmp_path / "test.shp"
        gdf_original.to_file(filepath)

        # Read and reproject to Web Mercator
        gdf_reprojected = read_shapefile(filepath, crs="EPSG:3857")

        # Check that CRS has changed
        assert gdf_reprojected.crs.to_string() == "EPSG:3857"
        assert len(gdf_reprojected) == 2

        # Check that coordinates have changed (rough check)
        original_x = gdf_original.geometry.x.values[0]
        reprojected_x = gdf_reprojected.geometry.x.values[0]
        assert abs(original_x - reprojected_x) > 100  # Should be very different
