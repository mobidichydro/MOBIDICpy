"""Tests for MeteoRaster class."""

import numpy as np
import pytest
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from mobidic.preprocessing.meteo_raster import MeteoRaster


# Fixtures for test data
@pytest.fixture
def sample_raster_file(tmp_path):
    """Create a sample NetCDF raster file for testing."""
    # Create synthetic data
    n_times = 10
    nrows, ncols = 50, 60
    resolution = 500.0
    xllcorner = 100000.0
    yllcorner = 200000.0

    # Create coordinates
    x = xllcorner + np.arange(ncols) * resolution
    y = yllcorner + np.arange(nrows) * resolution
    time = [datetime(2023, 1, 1, i) for i in range(n_times)]

    # Create data variables (precipitation and pet in mm/h)
    precipitation = np.random.rand(n_times, nrows, ncols) * 10.0  # 0-10 mm/h
    pet = np.ones((n_times, nrows, ncols)) * 0.1  # 0.1 mm/h

    # Create dataset
    ds = xr.Dataset(
        {
            "precipitation": (["time", "y", "x"], precipitation),
            "pet": (["time", "y", "x"], pet),
        },
        coords={
            "time": time,
            "y": y,
            "x": x,
        },
    )

    # Add CRS variable
    ds["crs"] = xr.DataArray(0, attrs={"spatial_ref": "EPSG:32632"})

    # Save to file
    nc_path = tmp_path / "test_raster.nc"
    ds.to_netcdf(nc_path)

    return nc_path, {
        "shape": (nrows, ncols),
        "resolution": resolution,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "crs": "EPSG:32632",
        "n_times": n_times,
        "time": time,
    }


@pytest.fixture
def arno_raster_file():
    """Path to Arno example raster file (if it exists)."""
    arno_path = Path("examples/Arno/meteodata/Arno_meteoraster.nc")
    if arno_path.exists():
        return arno_path
    return None


# Test file loading
def test_load_valid_file(sample_raster_file):
    """Test loading a valid NetCDF raster file."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)

    assert meteo.nc_path == nc_path
    assert "precipitation" in meteo.variables
    assert "pet" in meteo.variables
    assert len(meteo.ds.time) == metadata["n_times"]
    assert meteo.grid_metadata["shape"] == metadata["shape"]
    assert np.isclose(meteo.grid_metadata["resolution"], metadata["resolution"])


def test_load_with_from_netcdf(sample_raster_file):
    """Test loading using from_netcdf class method."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster.from_netcdf(nc_path)

    assert isinstance(meteo, MeteoRaster)
    assert "precipitation" in meteo.variables


def test_load_missing_file(tmp_path):
    """Test loading a file that doesn't exist."""
    nc_path = tmp_path / "nonexistent.nc"

    with pytest.raises(FileNotFoundError, match="not found"):
        MeteoRaster(nc_path)


def test_load_invalid_structure_no_time(tmp_path):
    """Test loading a file without time dimension."""
    # Create invalid dataset (no time dimension)
    ds = xr.Dataset(
        {"var": (["y", "x"], np.random.rand(10, 10))},
        coords={"y": np.arange(10), "x": np.arange(10)},
    )

    nc_path = tmp_path / "invalid.nc"
    ds.to_netcdf(nc_path)

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        MeteoRaster(nc_path)


def test_load_invalid_structure_no_y(tmp_path):
    """Test loading a file without y dimension."""
    # Create invalid dataset (no y dimension)
    ds = xr.Dataset(
        {"var": (["time", "x"], np.random.rand(10, 10))},
        coords={"time": [datetime(2023, 1, 1, i) for i in range(10)], "x": np.arange(10)},
    )

    nc_path = tmp_path / "invalid.nc"
    ds.to_netcdf(nc_path)

    with pytest.raises(ValueError, match="must have 'y' dimension"):
        MeteoRaster(nc_path)


def test_load_invalid_structure_no_variables(tmp_path):
    """Test loading a file with no data variables."""
    # Create dataset with only crs (no meteorological variables)
    time = [datetime(2023, 1, 1, i) for i in range(10)]
    ds = xr.Dataset(
        {"crs": xr.DataArray(0)},
        coords={
            "time": time,
            "y": np.arange(10),
            "x": np.arange(10),
        },
    )

    nc_path = tmp_path / "invalid.nc"
    ds.to_netcdf(nc_path)

    with pytest.raises(ValueError, match="at least one meteorological variable"):
        MeteoRaster(nc_path)


# Test metadata extraction
def test_metadata_extraction(sample_raster_file):
    """Test that metadata is correctly extracted."""
    nc_path, expected = sample_raster_file

    meteo = MeteoRaster(nc_path)

    assert meteo.grid_metadata["shape"] == expected["shape"]
    assert np.isclose(meteo.grid_metadata["resolution"], expected["resolution"])
    assert np.isclose(meteo.grid_metadata["xllcorner"], expected["xllcorner"])
    assert np.isclose(meteo.grid_metadata["yllcorner"], expected["yllcorner"])


# Test time indexing
def test_get_timestep_exact_match(sample_raster_file):
    """Test getting data for exact time match."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    time = metadata["time"][5]  # Middle timestep

    data = meteo.get_timestep(time, "precipitation")

    assert isinstance(data, np.ndarray)
    assert data.shape == metadata["shape"]
    assert np.all(np.isfinite(data))  # No NaN values


def test_get_timestep_nearest_neighbor(sample_raster_file):
    """Test getting data with nearest neighbor time sampling."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    # Time between two timesteps
    time = datetime(2023, 1, 1, 4, 30)  # Between hour 4 and 5

    data = meteo.get_timestep(time, "precipitation")

    # Should return data (nearest neighbor)
    assert isinstance(data, np.ndarray)
    assert data.shape == metadata["shape"]


def test_get_timestep_before_range(sample_raster_file):
    """Test getting data for time before first timestep."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    time = datetime(2022, 12, 31)  # Before first timestep

    data = meteo.get_timestep(time, "precipitation")

    # Should return first timestep (nearest neighbor)
    assert isinstance(data, np.ndarray)
    assert data.shape == metadata["shape"]


def test_get_timestep_after_range(sample_raster_file):
    """Test getting data for time after last timestep."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    time = datetime(2023, 1, 2)  # After last timestep

    data = meteo.get_timestep(time, "precipitation")

    # Should return last timestep (nearest neighbor)
    assert isinstance(data, np.ndarray)
    assert data.shape == metadata["shape"]


def test_get_timestep_invalid_variable(sample_raster_file):
    """Test getting data for non-existent variable."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    time = metadata["time"][0]

    with pytest.raises(KeyError, match="not found"):
        meteo.get_timestep(time, "nonexistent_variable")


def test_get_timestep_caching(sample_raster_file):
    """Test that get_timestep caches results."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    time = metadata["time"][0]

    # First call - should load from file
    data1 = meteo.get_timestep(time, "precipitation")

    # Second call - should use cache
    data2 = meteo.get_timestep(time, "precipitation")

    # Should be the same array (not just equal, but same object)
    assert data1 is data2
    assert len(meteo._cache) == 1


def test_get_timestep_different_variables_cached_separately(sample_raster_file):
    """Test that different variables are cached separately."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    time = metadata["time"][0]

    precip = meteo.get_timestep(time, "precipitation")
    pet = meteo.get_timestep(time, "pet")

    # Should have 2 cache entries
    assert len(meteo._cache) == 2
    # Should be different arrays
    assert not np.array_equal(precip, pet)


# Test grid validation
def test_validate_grid_alignment_matching(sample_raster_file):
    """Test grid validation with matching grids."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    model_metadata = {
        "shape": metadata["shape"],
        "resolution": metadata["resolution"],
        "xllcorner": metadata["xllcorner"],
        "yllcorner": metadata["yllcorner"],
        "crs": metadata["crs"],
    }

    # Should not raise
    meteo.validate_grid_alignment(model_metadata)


def test_validate_grid_alignment_wrong_shape(sample_raster_file):
    """Test grid validation with wrong shape."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    model_metadata = {
        "shape": (100, 100),  # Wrong shape
        "resolution": metadata["resolution"],
        "xllcorner": metadata["xllcorner"],
        "yllcorner": metadata["yllcorner"],
    }

    with pytest.raises(ValueError, match="Shape mismatch"):
        meteo.validate_grid_alignment(model_metadata)


def test_validate_grid_alignment_wrong_resolution(sample_raster_file):
    """Test grid validation with wrong resolution."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    model_metadata = {
        "shape": metadata["shape"],
        "resolution": 1000.0,  # Wrong resolution
        "xllcorner": metadata["xllcorner"],
        "yllcorner": metadata["yllcorner"],
    }

    with pytest.raises(ValueError, match="Resolution mismatch"):
        meteo.validate_grid_alignment(model_metadata)


def test_validate_grid_alignment_wrong_x_origin(sample_raster_file):
    """Test grid validation with wrong x origin."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    model_metadata = {
        "shape": metadata["shape"],
        "resolution": metadata["resolution"],
        "xllcorner": metadata["xllcorner"] + 100,  # Wrong x origin
        "yllcorner": metadata["yllcorner"],
    }

    with pytest.raises(ValueError, match="X origin mismatch"):
        meteo.validate_grid_alignment(model_metadata)


def test_validate_grid_alignment_wrong_y_origin(sample_raster_file):
    """Test grid validation with wrong y origin."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    model_metadata = {
        "shape": metadata["shape"],
        "resolution": metadata["resolution"],
        "xllcorner": metadata["xllcorner"],
        "yllcorner": metadata["yllcorner"] + 100,  # Wrong y origin
    }

    with pytest.raises(ValueError, match="Y origin mismatch"):
        meteo.validate_grid_alignment(model_metadata)


def test_validate_grid_alignment_small_tolerance(sample_raster_file):
    """Test grid validation with small floating point differences."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    model_metadata = {
        "shape": metadata["shape"],
        "resolution": metadata["resolution"] + 1e-9,  # Tiny difference
        "xllcorner": metadata["xllcorner"] + 1e-6,  # Tiny difference
        "yllcorner": metadata["yllcorner"] + 1e-6,  # Tiny difference
    }

    # Should not raise (within tolerance)
    meteo.validate_grid_alignment(model_metadata)


# Test context manager
def test_context_manager(sample_raster_file):
    """Test using MeteoRaster as context manager."""
    nc_path, metadata = sample_raster_file

    with MeteoRaster(nc_path) as meteo:
        data = meteo.get_timestep(metadata["time"][0], "precipitation")
        assert data is not None

    # After exiting context, cache should be cleared
    assert len(meteo._cache) == 0


def test_close_method(sample_raster_file):
    """Test close method."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    meteo.get_timestep(metadata["time"][0], "precipitation")

    assert len(meteo._cache) > 0

    meteo.close()

    assert len(meteo._cache) == 0


# Test __repr__
def test_repr(sample_raster_file):
    """Test string representation."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)
    repr_str = repr(meteo)

    assert "MeteoRaster" in repr_str
    assert "precipitation" in repr_str
    assert "pet" in repr_str


# Test with larger synthetic dataset (replacing Arno tests)
def test_load_large_synthetic_raster(tmp_path):
    """Test loading a larger synthetic raster file (similar to real data)."""
    # Create larger synthetic data similar to Arno dimensions
    n_times = 20
    nrows, ncols = 253, 313
    resolution = 500.0
    xllcorner = 100000.0
    yllcorner = 200000.0

    # Create coordinates
    x = xllcorner + np.arange(ncols) * resolution
    y = yllcorner + np.arange(nrows) * resolution
    times = [datetime(2023, 10, 31, 0, 15) + timedelta(minutes=15 * i) for i in range(n_times)]

    # Create data with some NaN values to simulate basin boundaries
    precipitation = np.random.rand(n_times, nrows, ncols) * 5.0  # 0-5 mm/h
    pet = np.random.rand(n_times, nrows, ncols) * 0.5  # 0-0.5 mm/h

    # Add some NaN values to simulate areas outside basin
    mask = np.random.rand(nrows, ncols) < 0.3  # 30% NaN
    precipitation[:, mask] = np.nan
    pet[:, mask] = np.nan

    # Create dataset
    ds = xr.Dataset(
        {
            "precipitation": (["time", "y", "x"], precipitation),
            "pet": (["time", "y", "x"], pet),
        },
        coords={
            "time": times,
            "y": y,
            "x": x,
        },
    )
    ds["crs"] = xr.DataArray(0, attrs={"spatial_ref": "EPSG:32632"})

    # Save and load
    nc_path = tmp_path / "large_raster.nc"
    ds.to_netcdf(nc_path)
    meteo = MeteoRaster(nc_path)

    assert "precipitation" in meteo.variables
    assert "pet" in meteo.variables
    assert meteo.grid_metadata["shape"] == (253, 313)
    assert np.isclose(meteo.grid_metadata["resolution"], 500.0)


def test_synthetic_raster_timestep_with_nans(tmp_path):
    """Test extracting a timestep from synthetic raster with NaN values."""
    # Create synthetic data with NaN values
    n_times = 5
    nrows, ncols = 100, 120
    resolution = 500.0
    xllcorner = 100000.0
    yllcorner = 200000.0

    x = xllcorner + np.arange(ncols) * resolution
    y = yllcorner + np.arange(nrows) * resolution
    times = [datetime(2023, 10, 31, 0, 15) + timedelta(minutes=15 * i) for i in range(n_times)]

    precipitation = np.random.rand(n_times, nrows, ncols) * 3.0
    # Add NaN mask to simulate basin boundary
    mask = np.random.rand(nrows, ncols) < 0.25
    precipitation[:, mask] = np.nan

    ds = xr.Dataset(
        {"precipitation": (["time", "y", "x"], precipitation)},
        coords={"time": times, "y": y, "x": x},
    )
    ds["crs"] = xr.DataArray(0, attrs={"spatial_ref": "EPSG:32632"})

    nc_path = tmp_path / "masked_raster.nc"
    ds.to_netcdf(nc_path)
    meteo = MeteoRaster(nc_path)

    # Get first timestep
    time = datetime(2023, 10, 31, 0, 15)
    precip = meteo.get_timestep(time, "precipitation")

    assert precip.shape == (100, 120)
    # Precipitation should be non-negative where not NaN (basin cells)
    valid_precip = precip[~np.isnan(precip)]
    assert len(valid_precip) > 0  # Should have some valid data
    assert np.all(valid_precip >= 0)  # Valid precipitation should be non-negative


# Test start_date and end_date properties (for API consistency with MeteoData)
def test_start_date_end_date_properties(sample_raster_file):
    """Test that start_date and end_date properties work correctly."""
    nc_path, metadata = sample_raster_file

    meteo = MeteoRaster(nc_path)

    # Properties should return same values as time_range
    assert meteo.start_date == meteo.time_range[0]
    assert meteo.end_date == meteo.time_range[1]

    # Should match expected times
    assert meteo.start_date == metadata["time"][0]
    assert meteo.end_date == metadata["time"][-1]
