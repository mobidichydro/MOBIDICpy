"""Tests for I/O module."""

import numpy as np
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString
from unittest.mock import Mock

from mobidic.preprocessing.io import (
    save_gisdata,
    save_network,
    load_gisdata,
    load_network,
)
from mobidic.preprocessing.preprocessor import GISData


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock()
    config.basin.id = "TestBasin"
    config.basin.paramset_id = "TestParams"
    config.basin.baricenter.lon = 11.0
    config.basin.baricenter.lat = 44.0
    config.simulation.resample = 1
    return config


@pytest.fixture
def sample_grids():
    """Create sample grid data."""
    return {
        "dtm": np.random.rand(10, 10) * 100,
        "flow_dir": np.random.randint(1, 9, size=(10, 10)).astype(float),
        "flow_acc": np.random.randint(1, 100, size=(10, 10)).astype(float),
        "Wc0": np.random.rand(10, 10) * 50,
        "Wg0": np.random.rand(10, 10) * 100,
        "ks": np.random.rand(10, 10) * 10,
        "kf": np.full((10, 10), 1e-7),
        "CH": np.full((10, 10), 1e-3),
        "Alb": np.full((10, 10), 0.2),
    }


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "shape": (10, 10),
        "resolution": (10.0, 10.0),
        "crs": "EPSG:32632",
        "nodata": np.nan,
        "xllcorner": 5.0,
        "yllcorner": 5.0,
        "cellsize": 10.0,
    }


@pytest.fixture
def sample_network():
    """Create sample river network."""
    network = gpd.GeoDataFrame(
        {
            "mobidic_id": [0, 1, 2],
            "upstream_1": [np.nan, np.nan, 0],
            "upstream_2": [np.nan, np.nan, 1],
            "downstream": [2, 2, -1],
            "strahler_order": [1, 1, 2],
            "length_m": [1000.0, 1200.0, 1500.0],
            "width_m": [1.0, 1.0, 2.83],
            "lag_time_s": [200.0, 240.0, 300.0],
            "storage_coeff": [0.45, 0.44, 0.43],
            "n_manning": [0.03, 0.03, 0.03],
            "calc_order": [1, 1, 2],
        },
        geometry=[
            LineString([(0, 0), (10, 10)]),
            LineString([(20, 0), (30, 10)]),
            LineString([(10, 10), (30, 10), (40, 20)]),
        ],
        crs="EPSG:32632",
    )
    return network


@pytest.fixture
def sample_hillslope_map():
    """Create sample hillslope-reach mapping."""
    return np.random.randint(-9999, 3, size=(10, 10)).astype(float)


@pytest.fixture
def sample_gisdata(sample_grids, sample_metadata, sample_network, sample_hillslope_map, mock_config):
    """Create sample GISData object."""
    return GISData(
        grids=sample_grids,
        metadata=sample_metadata,
        network=sample_network,
        hillslope_reach_map=sample_hillslope_map,
        config=mock_config,
    )


class TestSaveGISData:
    """Tests for save_gisdata function."""

    def test_save_gisdata_creates_file(self, sample_gisdata, tmp_path):
        """Test that save_gisdata creates a NetCDF file."""
        output_path = tmp_path / "test_gisdata.nc"

        save_gisdata(sample_gisdata, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_gisdata_structure(self, sample_gisdata, tmp_path):
        """Test that saved NetCDF has correct structure."""
        output_path = tmp_path / "test_gisdata.nc"

        save_gisdata(sample_gisdata, output_path)

        # Load and verify structure
        ds = xr.open_dataset(output_path)

        # Check dimensions
        assert "x" in ds.dims
        assert "y" in ds.dims
        assert ds.dims["x"] == 10
        assert ds.dims["y"] == 10

        # Check variables
        assert "dtm" in ds
        assert "flow_dir" in ds
        assert "Wc0" in ds
        assert "hillslope_reach_map" in ds

        # Check attributes
        assert "basin_id" in ds.attrs
        assert ds.attrs["basin_id"] == "TestBasin"
        assert "crs" in ds.attrs

        ds.close()

    def test_save_gisdata_preserves_data(self, sample_gisdata, tmp_path):
        """Test that saved data matches original."""
        output_path = tmp_path / "test_gisdata.nc"

        save_gisdata(sample_gisdata, output_path)

        # Load and verify data
        ds = xr.open_dataset(output_path)

        np.testing.assert_array_almost_equal(ds["dtm"].values, sample_gisdata.grids["dtm"])
        np.testing.assert_array_almost_equal(ds["flow_dir"].values, sample_gisdata.grids["flow_dir"])
        np.testing.assert_array_almost_equal(ds["hillslope_reach_map"].values, sample_gisdata.hillslope_reach_map)

        ds.close()

    def test_save_gisdata_creates_parent_directory(self, sample_gisdata, tmp_path):
        """Test that save_gisdata creates parent directories."""
        output_path = tmp_path / "subdir" / "test_gisdata.nc"

        save_gisdata(sample_gisdata, output_path)

        assert output_path.exists()


class TestSaveNetwork:
    """Tests for save_network function."""

    def test_save_network_creates_file(self, sample_network, tmp_path):
        """Test that save_network creates a GeoParquet file."""
        output_path = tmp_path / "test_network.parquet"

        save_network(sample_network, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_network_preserves_data(self, sample_network, tmp_path):
        """Test that saved network matches original."""
        output_path = tmp_path / "test_network.parquet"

        save_network(sample_network, output_path)

        # Load and verify
        loaded = gpd.read_parquet(output_path)

        assert len(loaded) == len(sample_network)
        assert list(loaded.columns) == list(sample_network.columns)
        assert loaded.crs == sample_network.crs

        # Check some values
        np.testing.assert_array_equal(loaded["mobidic_id"].values, sample_network["mobidic_id"].values)
        np.testing.assert_array_equal(loaded["strahler_order"].values, sample_network["strahler_order"].values)


class TestLoadGISData:
    """Tests for load_gisdata function."""

    def test_load_gisdata_reads_file(self, sample_gisdata, tmp_path):
        """Test that load_gisdata reads saved file."""
        gisdata_path = tmp_path / "test_gisdata.nc"
        network_path = tmp_path / "test_network.parquet"

        # Save first
        save_gisdata(sample_gisdata, gisdata_path)
        save_network(sample_gisdata.network, network_path)

        # Load
        loaded = load_gisdata(gisdata_path, network_path)

        assert isinstance(loaded, GISData)
        assert "dtm" in loaded.grids
        assert "flow_dir" in loaded.grids
        assert isinstance(loaded.network, gpd.GeoDataFrame)
        assert isinstance(loaded.hillslope_reach_map, np.ndarray)

    def test_load_gisdata_preserves_grids(self, sample_gisdata, tmp_path):
        """Test that loaded grids match original."""
        gisdata_path = tmp_path / "test_gisdata.nc"
        network_path = tmp_path / "test_network.parquet"

        # Save first
        save_gisdata(sample_gisdata, gisdata_path)
        save_network(sample_gisdata.network, network_path)

        # Load
        loaded = load_gisdata(gisdata_path, network_path)

        # Check grids
        np.testing.assert_array_almost_equal(loaded.grids["dtm"], sample_gisdata.grids["dtm"])
        np.testing.assert_array_almost_equal(loaded.grids["flow_dir"], sample_gisdata.grids["flow_dir"])
        np.testing.assert_array_almost_equal(loaded.grids["Wc0"], sample_gisdata.grids["Wc0"])

    def test_load_gisdata_preserves_network(self, sample_gisdata, tmp_path):
        """Test that loaded network matches original."""
        gisdata_path = tmp_path / "test_gisdata.nc"
        network_path = tmp_path / "test_network.parquet"

        # Save first
        save_gisdata(sample_gisdata, gisdata_path)
        save_network(sample_gisdata.network, network_path)

        # Load
        loaded = load_gisdata(gisdata_path, network_path)

        # Check network
        assert len(loaded.network) == len(sample_gisdata.network)
        np.testing.assert_array_equal(loaded.network["mobidic_id"].values, sample_gisdata.network["mobidic_id"].values)

    def test_load_gisdata_file_not_found(self, tmp_path):
        """Test that load_gisdata raises error if file not found."""
        gisdata_path = tmp_path / "nonexistent.nc"
        network_path = tmp_path / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError):
            load_gisdata(gisdata_path, network_path)


class TestLoadNetwork:
    """Tests for load_network function."""

    def test_load_network_reads_file(self, sample_network, tmp_path):
        """Test that load_network reads saved file."""
        output_path = tmp_path / "test_network.parquet"

        # Save first
        save_network(sample_network, output_path)

        # Load
        loaded = load_network(output_path)

        assert isinstance(loaded, gpd.GeoDataFrame)
        assert len(loaded) == len(sample_network)

    def test_load_network_file_not_found(self, tmp_path):
        """Test that load_network raises error if file not found."""
        network_path = tmp_path / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError):
            load_network(network_path)


class TestRoundTrip:
    """Test complete save/load round trip."""

    def test_complete_round_trip(self, sample_gisdata, tmp_path):
        """Test saving and loading preserves all data."""
        gisdata_path = tmp_path / "roundtrip_gisdata.nc"
        network_path = tmp_path / "roundtrip_network.parquet"

        # Save
        sample_gisdata.save(gisdata_path, network_path)

        # Load
        loaded = GISData.load(gisdata_path, network_path)

        # Verify grids
        for key in sample_gisdata.grids:
            np.testing.assert_array_almost_equal(loaded.grids[key], sample_gisdata.grids[key])

        # Verify hillslope map
        np.testing.assert_array_almost_equal(loaded.hillslope_reach_map, sample_gisdata.hillslope_reach_map)

        # Verify network
        assert len(loaded.network) == len(sample_gisdata.network)
        np.testing.assert_array_equal(loaded.network["mobidic_id"].values, sample_gisdata.network["mobidic_id"].values)

        # Verify metadata
        assert loaded.metadata["shape"] == sample_gisdata.metadata["shape"]
        assert loaded.metadata["resolution"] == sample_gisdata.metadata["resolution"]
