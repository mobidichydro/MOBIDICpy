"""Tests for GIS data readers."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from loguru import logger
from shapely.geometry import LineString

from mobidic.preprocessing.gis_reader import (
    GridAlignmentError,
    check_crs,
    check_raster_alignment,
    check_vector_crs,
    grid_to_matrix,
    read_shapefile,
)


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

    def test_read_shapefile_runtime_error(self, tmp_path):
        """Test that RuntimeError is raised for invalid shapefile."""
        # Create a file with .shp extension but invalid content
        filepath = tmp_path / "invalid.shp"
        filepath.write_text("This is not a valid shapefile")

        # Try to read the invalid shapefile
        with pytest.raises(RuntimeError, match="Failed to read shapefile"):
            read_shapefile(filepath)


class TestGridToMatrix:
    """Tests for grid_to_matrix function."""

    def test_grid_to_matrix_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            grid_to_matrix("nonexistent_file.tif")

    def test_grid_to_matrix_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported file format."""
        filepath = tmp_path / "test.xyz"
        filepath.write_text("dummy data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            grid_to_matrix(filepath)

    def test_grid_to_matrix_success(self, tmp_path):
        """Test successful reading of GeoTIFF with grid_to_matrix."""
        # Create a test raster
        filepath = tmp_path / "test.tif"
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        cellsize = 10.0
        xll = 100.0  # lower-left corner
        yll = 200.0

        transform = Affine.translation(xll, yll + data.shape[0] * cellsize) @ Affine.scale(cellsize, -cellsize)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Read with grid_to_matrix
        result = grid_to_matrix(filepath)

        # Check that result is a dictionary
        assert isinstance(result, dict)
        assert "data" in result
        assert "xllcorner" in result
        assert "yllcorner" in result
        assert "cellsize" in result

        # Check that data is flipped vertically
        assert np.array_equal(result["data"], np.flipud(data))

        # Check cellsize
        assert result["cellsize"] == cellsize

        # Check corner coordinates (adjusted to cell center)
        assert result["xllcorner"] == xll + 0.5 * cellsize
        assert result["yllcorner"] == yll + 0.5 * cellsize

    def test_grid_to_matrix_with_nodata(self, tmp_path):
        """Test grid_to_matrix with nodata values."""
        # Create a test raster with nodata
        filepath = tmp_path / "test.tif"
        data = np.array([[1, 2, -9999], [4, -9999, 6], [7, 8, 9]], dtype=np.float32)
        nodata = -9999.0
        cellsize = 5.0
        xll = 0.0
        yll = 0.0

        transform = Affine.translation(xll, yll + data.shape[0] * cellsize) @ Affine.scale(cellsize, -cellsize)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)

        # Read with grid_to_matrix
        result = grid_to_matrix(filepath)

        # Check that nodata values are converted to NaN
        # After flipud: row 0 becomes row 2, row 2 becomes row 0
        flipped_data = np.flipud(data)
        flipped_data[flipped_data == nodata] = np.nan

        assert np.isnan(result["data"][2, 2])  # Original [0, 2] -> flipped [2, 2]
        assert np.isnan(result["data"][1, 1])  # Middle stays in middle

    def test_grid_to_matrix_very_small_values(self, tmp_path):
        """Test grid_to_matrix converts very small values to NaN."""
        # Create a test raster with very small values
        filepath = tmp_path / "test.tif"
        data = np.array([[1, 2, 3], [4, -1e33, 6], [7, 8, -2e33]], dtype=np.float32)
        cellsize = 1.0

        transform = Affine.translation(0, data.shape[0] * cellsize) @ Affine.scale(cellsize, -cellsize)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        # Read with grid_to_matrix
        result = grid_to_matrix(filepath)

        # Check that very small values (< -1e32) are converted to NaN
        # Note: array is flipped, so positions change
        assert np.isnan(result["data"][1, 1])  # -1e33 value
        assert np.isnan(result["data"][0, 2])  # -2e33 value
        assert result["data"][2, 0] == 1.0  # Normal value

    def test_grid_to_matrix_runtime_error(self, tmp_path):
        """Test that RuntimeError is raised for corrupted GeoTIFF."""
        # Create a corrupted file
        filepath = tmp_path / "corrupted.tif"
        filepath.write_text("This is not a valid GeoTIFF")

        with pytest.raises(RuntimeError, match="Error reading GeoTIFF file"):
            grid_to_matrix(filepath)


def _write_raster(path, shape=(4, 5), cellsize=100.0, xll=1000.0, yll=2000.0, crs="EPSG:3003"):
    """Write a GeoTIFF with the given grid and return its grid_to_matrix() result."""
    data = np.ones(shape, dtype=np.float32)
    transform = Affine.translation(xll, yll + shape[0] * cellsize) @ Affine.scale(cellsize, -cellsize)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return grid_to_matrix(path)


def _capture_warnings(func, *args, **kwargs):
    """Call func and return the loguru WARNING messages it emitted."""
    messages = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        func(*args, **kwargs)
    finally:
        logger.remove(handler_id)
    return messages


class TestCheckCrs:
    """Tests for check_crs function."""

    def test_projected_crs(self):
        """Test that a projected CRS is accepted."""
        assert check_crs("EPSG:3003", "dtm.tif") is True

    def test_geographic_crs_rejected(self):
        """Test that a geographic CRS raises an error naming the file."""
        with pytest.raises(GridAlignmentError, match=r"dtm\.tif has a geographic CRS \(EPSG:4326\)"):
            check_crs("EPSG:4326", "dtm.tif")

    def test_missing_crs_warns(self):
        """Test that a missing CRS only warns."""
        messages = _capture_warnings(check_crs, None, "dtm.tif")
        assert check_crs(None, "dtm.tif") is False
        assert any("dtm.tif has no CRS" in m for m in messages)


class TestCheckRasterAlignment:
    """Tests for check_raster_alignment function."""

    def test_aligned_raster(self, tmp_path):
        """Test that a raster on the DTM grid passes."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif")
        check_raster_alignment(ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")

    def test_shape_mismatch(self, tmp_path):
        """Test that a different shape raises an error naming the file."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif", shape=(5, 5))
        with pytest.raises(GridAlignmentError, match=r"ks\.tif does not match the DTM grid[\s\S]*shape"):
            check_raster_alignment(ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")

    def test_origin_mismatch_same_shape(self, tmp_path):
        """Test that a shifted raster with the same shape is rejected."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif", xll=1050.0)
        with pytest.raises(GridAlignmentError, match="lower-left cell center"):
            check_raster_alignment(ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")

    def test_cellsize_mismatch(self, tmp_path):
        """Test that a different cell size is rejected."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif", cellsize=50.0)
        with pytest.raises(GridAlignmentError, match="cell size"):
            check_raster_alignment(ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")

    def test_crs_mismatch_same_grid(self, tmp_path):
        """Test that the same grid in a different projected CRS is rejected."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif", crs="EPSG:32632")
        with pytest.raises(GridAlignmentError, match=r"CRS EPSG:32632 != EPSG:3003"):
            check_raster_alignment(ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")

    def test_geographic_raster_rejected(self, tmp_path):
        """Test that a raster with a geographic CRS is rejected."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif", crs="EPSG:4326")
        with pytest.raises(GridAlignmentError, match="geographic CRS"):
            check_raster_alignment(ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")

    def test_missing_crs_warns(self, tmp_path):
        """Test that a raster without CRS on the DTM grid only warns."""
        dtm = _write_raster(tmp_path / "dtm.tif")
        ks = _write_raster(tmp_path / "ks.tif", crs=None)
        messages = _capture_warnings(check_raster_alignment, ks, dtm, tmp_path / "ks.tif", tmp_path / "dtm.tif")
        assert any("has no CRS" in m for m in messages)


class TestCheckVectorCrs:
    """Tests for check_vector_crs function."""

    @staticmethod
    def _gdf(crs):
        return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 10)])], crs=crs)

    def test_matching_crs(self):
        """Test that a vector in the DTM CRS passes."""
        check_vector_crs(self._gdf("EPSG:3003"), "net.shp", "EPSG:3003")

    def test_crs_mismatch(self):
        """Test that a vector in a different CRS raises an error naming the file."""
        with pytest.raises(GridAlignmentError, match=r"net\.shp has CRS EPSG:32632, but the DTM has CRS EPSG:3003"):
            check_vector_crs(self._gdf("EPSG:32632"), "net.shp", "EPSG:3003")

    def test_geographic_crs_rejected(self):
        """Test that a vector with a geographic CRS is rejected even without a reference."""
        with pytest.raises(GridAlignmentError, match="geographic CRS"):
            check_vector_crs(self._gdf("EPSG:4326"), "net.shp")

    def test_missing_crs_warns(self):
        """Test that a vector without CRS only warns."""
        messages = _capture_warnings(check_vector_crs, self._gdf(None), "net.shp", "EPSG:3003")
        assert any("net.shp has no CRS" in m for m in messages)


def _write_sentinel_raster(path, nodata):
    """Write a 3x3 raster with one -9999 cell and the given nodata value."""
    data = np.ones((3, 3), dtype=np.float32)
    data[0, 0] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=data.dtype,
        nodata=nodata,
        crs="EPSG:3003",
        transform=Affine.translation(0, 3) @ Affine.scale(1, -1),
    ) as dst:
        dst.write(data, 1)


class TestNodataSentinels:
    """Tests for the warning on undeclared nodata sentinels."""

    def test_untagged_sentinel_warns(self, tmp_path):
        """A raster with -9999 cells and no nodata value triggers a warning."""
        path = tmp_path / "untagged.tif"
        _write_sentinel_raster(path, nodata=None)

        messages = _capture_warnings(grid_to_matrix, path)
        assert any("declares no nodata value but has 1 cells equal to -9999" in m for m in messages)
        assert grid_to_matrix(path)["data"].min() == -9999  # still read as data

    def test_tagged_nodata_no_warning(self, tmp_path):
        """A raster with a declared nodata value does not warn."""
        path = tmp_path / "tagged.tif"
        _write_sentinel_raster(path, nodata=-9999)

        assert not any("nodata" in m for m in _capture_warnings(grid_to_matrix, path))
