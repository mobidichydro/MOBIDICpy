"""Tests for hyetograph module."""

import numpy as np
import pytest
import xarray as xr
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds

from mobidic.preprocessing.hyetograph import (
    IDFParameters,
    HyetographGenerator,
    read_idf_parameters,
    read_idf_parameters_resampled,
    resample_raster_to_grid,
    idf_depth,
    _validate_raster_consistency,
)


# Fixtures for test data
@pytest.fixture
def sample_idf_rasters(tmp_path):
    """Create sample IDF parameter rasters for testing."""
    nrows, ncols = 50, 60
    cellsize = 500.0
    xllcorner = 100000.0
    yllcorner = 200000.0

    # Create synthetic IDF parameters
    # a: scale parameter (typically 20-50 mm for Italy)
    # n: shape parameter (typically 0.3-0.5)
    # k: return period factor (typically 1-3)
    a = np.random.uniform(20, 50, (nrows, ncols))
    n = np.random.uniform(0.3, 0.5, (nrows, ncols))
    k = np.random.uniform(1.0, 2.5, (nrows, ncols))

    # Add some NaN values to simulate basin mask
    mask = np.random.rand(nrows, ncols) < 0.1
    a[mask] = np.nan
    n[mask] = np.nan
    k[mask] = np.nan

    # Create transform (note: data will be flipped by grid_to_matrix)
    transform = from_bounds(
        xllcorner - cellsize / 2,
        yllcorner - cellsize / 2,
        xllcorner + (ncols - 0.5) * cellsize,
        yllcorner + (nrows - 0.5) * cellsize,
        ncols,
        nrows,
    )

    # Save rasters
    paths = {}
    for name, data in [("a", a), ("n", n), ("k", k)]:
        path = tmp_path / f"idf_{name}.tif"
        # Flip data vertically because grid_to_matrix will flip it back
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=nrows,
            width=ncols,
            count=1,
            dtype=data.dtype,
            crs="EPSG:32632",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            data_to_write = np.flipud(data.copy())
            data_to_write[np.isnan(data_to_write)] = -9999.0
            dst.write(data_to_write, 1)
        paths[name] = path

    return paths, {
        "a": a,
        "n": n,
        "k": k,
        "shape": (nrows, ncols),
        "cellsize": cellsize,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
    }


@pytest.fixture
def sample_idf_params():
    """Create sample IDFParameters for testing."""
    nrows, ncols = 50, 60
    a = np.random.uniform(20, 50, (nrows, ncols))
    n = np.random.uniform(0.3, 0.5, (nrows, ncols))
    k = np.random.uniform(1.0, 2.5, (nrows, ncols))

    return IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=100000.0,
        yllcorner=200000.0,
        cellsize=500.0,
        crs="EPSG:32632",
        shape=(nrows, ncols),
    )


# Tests for IDFParameters dataclass
def test_idf_parameters_creation():
    """Test creating IDFParameters dataclass."""
    a = np.ones((10, 10)) * 30.0
    n = np.ones((10, 10)) * 0.4
    k = np.ones((10, 10)) * 1.5

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(10, 10),
    )

    assert params.shape == (10, 10)
    assert np.allclose(params.a, 30.0)
    assert np.allclose(params.n, 0.4)
    assert np.allclose(params.k, 1.5)


# Tests for read_idf_parameters
def test_read_idf_parameters(sample_idf_rasters):
    """Test reading IDF parameters from raster files."""
    paths, expected = sample_idf_rasters

    params = read_idf_parameters(
        paths["a"],
        paths["n"],
        paths["k"],
    )

    assert params.shape == expected["shape"]
    assert np.isclose(params.cellsize, expected["cellsize"])
    # Check that non-NaN values are in expected range
    valid_a = params.a[~np.isnan(params.a)]
    assert np.all((valid_a >= 20) & (valid_a <= 50))


def test_read_idf_parameters_missing_file(tmp_path):
    """Test reading IDF parameters when a file is missing."""
    with pytest.raises(FileNotFoundError):
        read_idf_parameters(
            tmp_path / "nonexistent_a.tif",
            tmp_path / "nonexistent_n.tif",
            tmp_path / "nonexistent_k.tif",
        )


def test_validate_raster_consistency_matching(sample_idf_rasters):
    """Test raster consistency validation with matching rasters."""
    paths, _ = sample_idf_rasters

    # Read rasters as dicts (simulating grid_to_matrix output)
    from mobidic.preprocessing.gis_reader import grid_to_matrix

    a_data = grid_to_matrix(paths["a"])
    n_data = grid_to_matrix(paths["n"])
    k_data = grid_to_matrix(paths["k"])

    # Should not raise
    _validate_raster_consistency(a_data, n_data, k_data)


def test_validate_raster_consistency_shape_mismatch(tmp_path):
    """Test raster consistency validation with shape mismatch."""
    # Create rasters with different shapes
    a_data = {"data": np.ones((10, 10)), "cellsize": 100.0, "xllcorner": 0.0, "yllcorner": 0.0}
    n_data = {"data": np.ones((10, 10)), "cellsize": 100.0, "xllcorner": 0.0, "yllcorner": 0.0}
    k_data = {"data": np.ones((20, 20)), "cellsize": 100.0, "xllcorner": 0.0, "yllcorner": 0.0}  # Different shape

    with pytest.raises(ValueError, match="Shape mismatch"):
        _validate_raster_consistency(a_data, n_data, k_data)


def test_validate_raster_consistency_cellsize_mismatch(tmp_path):
    """Test raster consistency validation with cellsize mismatch."""
    a_data = {"data": np.ones((10, 10)), "cellsize": 100.0, "xllcorner": 0.0, "yllcorner": 0.0}
    n_data = {"data": np.ones((10, 10)), "cellsize": 200.0, "xllcorner": 0.0, "yllcorner": 0.0}  # Different cellsize
    k_data = {"data": np.ones((10, 10)), "cellsize": 100.0, "xllcorner": 0.0, "yllcorner": 0.0}

    with pytest.raises(ValueError, match="Cellsize mismatch"):
        _validate_raster_consistency(a_data, n_data, k_data)


# Tests for idf_depth function
def test_idf_depth_basic():
    """Test basic IDF depth calculation."""
    a = np.array([[30.0, 40.0], [35.0, 45.0]])
    n = np.array([[0.4, 0.4], [0.4, 0.4]])
    t = 1.0  # 1 hour

    depth = idf_depth(a, n, t)

    # h = a * t^n = a * 1^0.4 = a
    assert np.allclose(depth, a)


def test_idf_depth_with_duration():
    """Test IDF depth calculation with different durations."""
    a = np.array([[30.0]])
    n = np.array([[0.5]])

    # For t=4: h = 30 * 4^0.5 = 30 * 2 = 60
    depth_4h = idf_depth(a, n, 4.0)
    assert np.isclose(depth_4h[0, 0], 60.0)

    # For t=9: h = 30 * 9^0.5 = 30 * 3 = 90
    depth_9h = idf_depth(a, n, 9.0)
    assert np.isclose(depth_9h[0, 0], 90.0)


def test_idf_depth_nan_propagation():
    """Test that NaN values propagate through IDF calculation."""
    a = np.array([[30.0, np.nan], [35.0, 40.0]])
    n = np.array([[0.4, 0.4], [np.nan, 0.4]])
    t = 2.0

    depth = idf_depth(a, n, t)

    assert np.isnan(depth[0, 1])  # NaN in a
    assert np.isnan(depth[1, 0])  # NaN in n
    assert not np.isnan(depth[0, 0])
    assert not np.isnan(depth[1, 1])


# Tests for HyetographGenerator class
def test_hyetograph_generator_init(sample_idf_params):
    """Test HyetographGenerator initialization."""
    gen = HyetographGenerator(sample_idf_params, ka=0.8)

    assert gen.ka == 0.8
    assert gen.idf_params.shape == sample_idf_params.shape


def test_hyetograph_generator_from_rasters(sample_idf_rasters):
    """Test HyetographGenerator.from_rasters class method."""
    paths, _ = sample_idf_rasters

    gen = HyetographGenerator.from_rasters(
        a_raster=paths["a"],
        n_raster=paths["n"],
        k_raster=paths["k"],
        ka=0.9,
    )

    assert gen.ka == 0.9
    assert gen.idf_params is not None


def test_hyetograph_generate_chicago_decreasing(sample_idf_params):
    """Test generating Chicago decreasing hyetograph."""
    gen = HyetographGenerator(sample_idf_params, ka=0.8)

    times, precip = gen.generate(
        duration_hours=24,
        start_time=datetime(2023, 11, 1),
        method="chicago_decreasing",
        timestep_hours=1,
    )

    # Check dimensions
    assert len(times) == 24
    assert precip.shape == (24, sample_idf_params.shape[0], sample_idf_params.shape[1])

    # Check times are correct
    assert times[0] == datetime(2023, 11, 1, 0, 0)
    assert times[-1] == datetime(2023, 11, 1, 23, 0)


def test_hyetograph_generate_decreasing_intensity():
    """Test that Chicago decreasing hyetograph has monotonically decreasing intensity."""
    # Create simple uniform IDF parameters
    a = np.ones((10, 10)) * 30.0
    n = np.ones((10, 10)) * 0.4
    k = np.ones((10, 10)) * 1.5

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(10, 10),
    )

    gen = HyetographGenerator(params, ka=1.0)
    times, precip = gen.generate(
        duration_hours=10,
        start_time=datetime(2023, 1, 1),
        method="chicago_decreasing",
    )

    # For a single cell, intensity should decrease over time
    # (first hour has highest intensity in Chicago decreasing)
    cell_precip = precip[:, 5, 5]
    for i in range(1, len(cell_precip)):
        assert cell_precip[i] <= cell_precip[i - 1], f"Intensity not decreasing at step {i}"


def test_hyetograph_generate_ka_effect():
    """Test that ka (areal reduction factor) correctly scales precipitation."""
    a = np.ones((10, 10)) * 30.0
    n = np.ones((10, 10)) * 0.5
    k = np.ones((10, 10)) * 1.0

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(10, 10),
    )

    gen_full = HyetographGenerator(params, ka=1.0)
    gen_reduced = HyetographGenerator(params, ka=0.5)

    _, precip_full = gen_full.generate(5, datetime(2023, 1, 1))
    _, precip_reduced = gen_reduced.generate(5, datetime(2023, 1, 1))

    # With ka=0.5, precipitation should be half
    assert np.allclose(precip_reduced, precip_full * 0.5, rtol=1e-10)


def test_hyetograph_generate_unsupported_method(sample_idf_params):
    """Test that unsupported method raises error."""
    gen = HyetographGenerator(sample_idf_params)

    with pytest.raises(ValueError, match="Unsupported hyetograph method"):
        gen.generate(24, datetime(2023, 1, 1), method="unsupported_method")


def test_hyetograph_generate_total_depth():
    """Test that total depth matches IDF curve value."""
    # With uniform parameters, we can verify the total depth
    a = np.ones((5, 5)) * 30.0
    n = np.ones((5, 5)) * 0.5
    k = np.ones((5, 5)) * 2.0

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(5, 5),
    )

    gen = HyetographGenerator(params, ka=0.8)
    duration = 4  # hours

    _, precip = gen.generate(duration, datetime(2023, 1, 1))

    # Total depth = ka * k * a * t^n = 0.8 * 2.0 * 30.0 * 4^0.5 = 0.8 * 2.0 * 30.0 * 2 = 96 mm
    # precip is in mm/h, so total = sum(precip) * 1 hour = sum(precip)
    expected_total = 0.8 * 2.0 * 30.0 * (duration**0.5)
    actual_total = np.sum(precip[:, 2, 2])  # Pick a cell

    assert np.isclose(actual_total, expected_total, rtol=1e-10)


# Tests for to_netcdf
def test_hyetograph_to_netcdf(sample_idf_params, tmp_path):
    """Test exporting hyetograph to NetCDF."""
    gen = HyetographGenerator(sample_idf_params, ka=0.8)

    times, precip = gen.generate(
        duration_hours=12,
        start_time=datetime(2023, 11, 1),
    )

    output_path = tmp_path / "test_hyetograph.nc"
    gen.to_netcdf(
        output_path,
        times=times,
        precipitation=precip,
        add_metadata={"event": "test_storm"},
    )

    # Verify file exists and is valid
    assert output_path.exists()

    # Read back and verify
    ds = xr.open_dataset(output_path)

    assert "precipitation" in ds.data_vars
    assert "time" in ds.coords
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert ds.precipitation.shape == precip.shape
    assert ds.attrs["event"] == "test_storm"
    assert ds.precipitation.attrs["units"] == "mm h-1"

    ds.close()


def test_hyetograph_netcdf_compatible_with_meteoraster(sample_idf_params, tmp_path):
    """Test that exported NetCDF is compatible with MeteoRaster."""
    from mobidic.preprocessing.meteo_raster import MeteoRaster

    gen = HyetographGenerator(sample_idf_params, ka=0.8)
    times, precip = gen.generate(24, datetime(2023, 11, 1))

    output_path = tmp_path / "hyetograph_for_simulation.nc"
    gen.to_netcdf(output_path, times=times, precipitation=precip)

    # Should be loadable by MeteoRaster
    meteo = MeteoRaster(output_path)

    assert "precipitation" in meteo.variables
    assert meteo.grid_metadata["shape"] == sample_idf_params.shape
    assert len(meteo.ds.time) == 24

    meteo.close()


def test_hyetograph_netcdf_cf_compliance(sample_idf_params, tmp_path):
    """Test that exported NetCDF follows CF conventions."""
    gen = HyetographGenerator(sample_idf_params, ka=0.8)
    times, precip = gen.generate(6, datetime(2023, 11, 1))

    output_path = tmp_path / "cf_hyetograph.nc"
    gen.to_netcdf(output_path, times=times, precipitation=precip)

    ds = xr.open_dataset(output_path)

    # Check CF global attributes
    assert "Conventions" in ds.attrs
    assert "CF-1.12" in ds.attrs["Conventions"]

    # Check variable attributes
    assert ds.precipitation.attrs.get("long_name") is not None
    assert ds.precipitation.attrs.get("units") is not None
    assert ds.precipitation.attrs.get("grid_mapping") == "crs"

    # Check coordinate attributes
    assert ds.x.attrs.get("standard_name") == "projection_x_coordinate"
    assert ds.y.attrs.get("standard_name") == "projection_y_coordinate"

    # Check CRS variable exists
    assert "crs" in ds

    ds.close()


def test_hyetograph_with_different_timestep():
    """Test hyetograph generation with different timestep."""
    a = np.ones((10, 10)) * 30.0
    n = np.ones((10, 10)) * 0.4
    k = np.ones((10, 10)) * 1.5

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(10, 10),
    )

    gen = HyetographGenerator(params, ka=1.0)

    # Generate with 2-hour timestep
    times, precip = gen.generate(
        duration_hours=12,
        start_time=datetime(2023, 1, 1),
        timestep_hours=2,
    )

    # Should have 6 timesteps (12/2 = 6)
    assert len(times) == 6
    assert precip.shape[0] == 6

    # Time intervals should be 2 hours
    assert (times[1] - times[0]).total_seconds() == 7200


def test_hyetograph_mass_conservation():
    """Test that total mass is conserved regardless of timestep."""
    a = np.ones((5, 5)) * 30.0
    n = np.ones((5, 5)) * 0.5
    k = np.ones((5, 5)) * 2.0

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(5, 5),
    )

    gen = HyetographGenerator(params, ka=0.8)
    duration = 6  # hours

    # Generate with 1-hour timestep
    _, precip_1h = gen.generate(duration, datetime(2023, 1, 1), timestep_hours=1)

    # Generate with 2-hour timestep
    _, precip_2h = gen.generate(duration, datetime(2023, 1, 1), timestep_hours=2)

    # Total depth should be the same (sum of precip * timestep)
    total_1h = np.sum(precip_1h[:, 2, 2]) * 1  # 1-hour timestep
    total_2h = np.sum(precip_2h[:, 2, 2]) * 2  # 2-hour timestep

    assert np.isclose(total_1h, total_2h, rtol=1e-10)


# Tests for edge cases
def test_hyetograph_with_nan_mask(sample_idf_params):
    """Test hyetograph generation with NaN values in IDF parameters."""
    # Add some NaN values
    sample_idf_params.a[5:10, 5:10] = np.nan
    sample_idf_params.n[15:20, 15:20] = np.nan

    gen = HyetographGenerator(sample_idf_params, ka=0.8)
    times, precip = gen.generate(6, datetime(2023, 1, 1))

    # NaN values should propagate to output
    assert np.any(np.isnan(precip))
    # Cells with NaN parameters should have NaN precipitation
    assert np.all(np.isnan(precip[:, 5:10, 5:10]))


def test_hyetograph_single_hour_duration():
    """Test hyetograph with single hour duration."""
    a = np.ones((5, 5)) * 30.0
    n = np.ones((5, 5)) * 0.5
    k = np.ones((5, 5)) * 1.0

    params = IDFParameters(
        a=a,
        n=n,
        k=k,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=100.0,
        crs=None,
        shape=(5, 5),
    )

    gen = HyetographGenerator(params, ka=1.0)
    times, precip = gen.generate(1, datetime(2023, 1, 1))

    # Should have 1 timestep
    assert len(times) == 1
    assert precip.shape[0] == 1

    # For t=1: h = a * t^n = 30 * 1^0.5 = 30 mm
    # Intensity = 30 mm / 1 hour = 30 mm/h
    expected_intensity = 1.0 * 1.0 * 30.0 * (1**0.5)  # ka * k * a * t^n
    assert np.isclose(precip[0, 2, 2], expected_intensity)


# ============================================================================
# Tests for grid resampling functionality
# ============================================================================


@pytest.fixture
def sample_reference_raster(tmp_path):
    """Create a sample reference raster (DEM) for testing resampling."""
    nrows, ncols = 100, 120
    cellsize = 250.0  # Different resolution from IDF rasters
    xllcorner = 100000.0
    yllcorner = 200000.0

    # Create synthetic DEM data
    dem = np.random.uniform(100, 500, (nrows, ncols))

    # Add some NaN values to simulate basin mask
    mask = np.random.rand(nrows, ncols) < 0.15
    dem[mask] = np.nan

    # Create transform
    transform = from_bounds(
        xllcorner - cellsize / 2,
        yllcorner - cellsize / 2,
        xllcorner + (ncols - 0.5) * cellsize,
        yllcorner + (nrows - 0.5) * cellsize,
        ncols,
        nrows,
    )

    # Save raster
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=dem.dtype,
        crs="EPSG:32632",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        data_to_write = np.flipud(dem.copy())
        data_to_write[np.isnan(data_to_write)] = -9999.0
        dst.write(data_to_write, 1)

    return path, {
        "shape": (nrows, ncols),
        "cellsize": cellsize,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "dem": dem,
    }


@pytest.fixture
def sample_idf_rasters_different_resolution(tmp_path):
    """Create IDF rasters with different resolution than reference grid."""
    # Coarser resolution than reference DEM
    nrows, ncols = 25, 30
    cellsize = 1000.0  # 1km resolution (coarser than 250m DEM)
    xllcorner = 100000.0
    yllcorner = 200000.0

    # Create uniform IDF parameters for easier verification
    a = np.ones((nrows, ncols)) * 35.0
    n = np.ones((nrows, ncols)) * 0.42
    k = np.ones((nrows, ncols)) * 1.8

    # Create transform
    transform = from_bounds(
        xllcorner - cellsize / 2,
        yllcorner - cellsize / 2,
        xllcorner + (ncols - 0.5) * cellsize,
        yllcorner + (nrows - 0.5) * cellsize,
        ncols,
        nrows,
    )

    # Save rasters
    paths = {}
    for name, data in [("a", a), ("n", n), ("k", k)]:
        path = tmp_path / f"idf_{name}_coarse.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=nrows,
            width=ncols,
            count=1,
            dtype=data.dtype,
            crs="EPSG:32632",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(np.flipud(data), 1)
        paths[name] = path

    return paths, {
        "a_value": 35.0,
        "n_value": 0.42,
        "k_value": 1.8,
        "shape": (nrows, ncols),
        "cellsize": cellsize,
    }


# Tests for resample_raster_to_grid
def test_resample_raster_to_grid_basic(sample_idf_rasters_different_resolution, sample_reference_raster):
    """Test basic raster resampling to reference grid."""
    idf_paths, idf_meta = sample_idf_rasters_different_resolution
    ref_path, ref_meta = sample_reference_raster

    # Get reference raster properties
    with rasterio.open(ref_path) as src:
        ref_shape = (src.height, src.width)
        ref_transform = src.transform
        ref_crs = src.crs

    # Resample 'a' parameter
    a_resampled = resample_raster_to_grid(
        idf_paths["a"],
        ref_shape,
        ref_transform,
        ref_crs,
    )

    # Check output shape matches reference
    assert a_resampled.shape == ref_shape

    # Since input was uniform, output should be approximately uniform
    valid_values = a_resampled[~np.isnan(a_resampled)]
    assert len(valid_values) > 0
    assert np.allclose(valid_values, idf_meta["a_value"], rtol=0.01)


def test_resample_raster_to_grid_with_mask(sample_idf_rasters_different_resolution, sample_reference_raster):
    """Test raster resampling with mask applied."""
    idf_paths, idf_meta = sample_idf_rasters_different_resolution
    ref_path, ref_meta = sample_reference_raster

    # Create a simple mask (center region valid)
    ref_shape = ref_meta["shape"]
    mask = np.zeros(ref_shape, dtype=bool)
    mask[20:80, 30:90] = True

    with rasterio.open(ref_path) as src:
        ref_transform = src.transform
        ref_crs = src.crs

    a_resampled = resample_raster_to_grid(
        idf_paths["a"],
        ref_shape,
        ref_transform,
        ref_crs,
        ref_mask=mask,
    )

    # Outside mask should be NaN
    assert np.all(np.isnan(a_resampled[~mask]))

    # Inside mask should have values
    valid_in_mask = a_resampled[mask]
    assert not np.all(np.isnan(valid_in_mask))


def test_resample_raster_to_grid_missing_file(tmp_path, sample_reference_raster):
    """Test that missing input file raises error."""
    ref_path, ref_meta = sample_reference_raster

    with rasterio.open(ref_path) as src:
        ref_shape = (src.height, src.width)
        ref_transform = src.transform
        ref_crs = src.crs

    with pytest.raises(FileNotFoundError):
        resample_raster_to_grid(
            tmp_path / "nonexistent.tif",
            ref_shape,
            ref_transform,
            ref_crs,
        )


# Tests for read_idf_parameters_resampled
def test_read_idf_parameters_resampled(sample_idf_rasters_different_resolution, sample_reference_raster):
    """Test reading and resampling IDF parameters to reference grid."""
    idf_paths, idf_meta = sample_idf_rasters_different_resolution
    ref_path, ref_meta = sample_reference_raster

    params = read_idf_parameters_resampled(
        idf_paths["a"],
        idf_paths["n"],
        idf_paths["k"],
        ref_path,
    )

    # Shape should match reference grid
    assert params.shape == ref_meta["shape"]

    # Cellsize should match reference
    assert np.isclose(params.cellsize, ref_meta["cellsize"])

    # Values should be approximately uniform (since input was uniform)
    valid_a = params.a[~np.isnan(params.a)]
    valid_n = params.n[~np.isnan(params.n)]
    valid_k = params.k[~np.isnan(params.k)]

    assert np.allclose(valid_a, idf_meta["a_value"], rtol=0.01)
    assert np.allclose(valid_n, idf_meta["n_value"], rtol=0.01)
    assert np.allclose(valid_k, idf_meta["k_value"], rtol=0.01)


def test_read_idf_parameters_resampled_missing_ref(sample_idf_rasters_different_resolution, tmp_path):
    """Test that missing reference raster raises error."""
    idf_paths, _ = sample_idf_rasters_different_resolution

    with pytest.raises(FileNotFoundError, match="Reference raster not found"):
        read_idf_parameters_resampled(
            idf_paths["a"],
            idf_paths["n"],
            idf_paths["k"],
            tmp_path / "nonexistent_dem.tif",
        )


# Tests for HyetographGenerator.from_rasters with resampling
def test_hyetograph_generator_from_rasters_with_resampling(
    sample_idf_rasters_different_resolution, sample_reference_raster
):
    """Test creating HyetographGenerator with IDF resampling."""
    idf_paths, idf_meta = sample_idf_rasters_different_resolution
    ref_path, ref_meta = sample_reference_raster

    generator = HyetographGenerator.from_rasters(
        a_raster=idf_paths["a"],
        n_raster=idf_paths["n"],
        k_raster=idf_paths["k"],
        ka=0.8,
        ref_raster=ref_path,
    )

    # Generator should use resampled parameters with reference grid shape
    assert generator.idf_params.shape == ref_meta["shape"]
    assert np.isclose(generator.idf_params.cellsize, ref_meta["cellsize"])
    assert generator.ka == 0.8


def test_hyetograph_generator_from_rasters_without_resampling(sample_idf_rasters):
    """Test creating HyetographGenerator without resampling (original behavior)."""
    paths, expected = sample_idf_rasters

    generator = HyetographGenerator.from_rasters(
        a_raster=paths["a"],
        n_raster=paths["n"],
        k_raster=paths["k"],
        ka=0.9,
    )

    # Should use original IDF raster shape
    assert generator.idf_params.shape == expected["shape"]


def test_hyetograph_full_workflow_with_resampling(
    sample_idf_rasters_different_resolution, sample_reference_raster, tmp_path
):
    """Test complete hyetograph workflow with resampling."""
    idf_paths, idf_meta = sample_idf_rasters_different_resolution
    ref_path, ref_meta = sample_reference_raster

    # Create generator with resampling
    generator = HyetographGenerator.from_rasters(
        a_raster=idf_paths["a"],
        n_raster=idf_paths["n"],
        k_raster=idf_paths["k"],
        ka=0.8,
        ref_raster=ref_path,
    )

    # Generate hyetograph
    times, precip = generator.generate(
        duration_hours=24,
        start_time=datetime(2023, 11, 1),
    )

    # Check output dimensions
    assert len(times) == 24
    assert precip.shape[0] == 24
    assert precip.shape[1:] == ref_meta["shape"]

    # Export to NetCDF
    output_path = tmp_path / "hyetograph_resampled.nc"
    generator.to_netcdf(output_path, times=times, precipitation=precip)

    # Verify file
    assert output_path.exists()

    # Should be compatible with MeteoRaster
    from mobidic.preprocessing.meteo_raster import MeteoRaster

    meteo = MeteoRaster(output_path)
    assert meteo.grid_metadata["shape"] == ref_meta["shape"]
    meteo.close()
