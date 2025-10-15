"""Tests for preprocessing orchestrator module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import geopandas as gpd
from shapely.geometry import LineString

from mobidic.preprocessing.preprocessor import (
    GISData,
    run_preprocessing,
    _get_default_parameter_value,
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()
    config.basin.id = "TestBasin"
    config.basin.paramset_id = "TestParams"
    config.basin.baricenter.lon = 11.0
    config.basin.baricenter.lat = 44.0

    config.raster_files.dtm = "dtm.tif"
    config.raster_files.flow_dir = "flowdir.tif"
    config.raster_files.flow_acc = "flowacc.tif"
    config.raster_files.Wc0 = "wc0.tif"
    config.raster_files.Wg0 = "wg0.tif"
    config.raster_files.ks = "ks.tif"
    config.raster_files.kf = None
    config.raster_files.CH = None
    config.raster_files.Alb = None
    config.raster_files.Ma = None
    config.raster_files.Mf = None
    config.raster_files.gamma = None
    config.raster_files.kappa = None
    config.raster_files.beta = None
    config.raster_files.alpha = None

    config.raster_settings.flow_dir_type = "Grass"

    config.vector_files.river_network.shp = "network.shp"

    config.parameters.routing.wcel = 5.0
    config.parameters.routing.Br0 = 1.0
    config.parameters.routing.NBr = 1.5
    config.parameters.routing.n_Man = 0.03

    config.parameters.soil.kf = 1e-7
    config.parameters.soil.gamma = 2.69e-7
    config.parameters.soil.kappa = 1.10e-7
    config.parameters.soil.beta = 7.62e-6
    config.parameters.soil.alpha = 2.50e-5
    config.parameters.soil.ks_min = None
    config.parameters.soil.ks_max = None

    config.parameters.energy.CH = 1e-3
    config.parameters.energy.Alb = 0.2

    config.simulation.resample = 1

    return config


@pytest.fixture
def mock_grids():
    """Create mock grid data."""
    return {
        "dtm": np.random.rand(10, 10) * 100,
        "flow_dir": np.random.randint(1, 9, size=(10, 10)),
        "flow_acc": np.random.randint(1, 100, size=(10, 10)),
        "Wc0": np.random.rand(10, 10) * 50,
        "Wg0": np.random.rand(10, 10) * 100,
        "ks": np.random.rand(10, 10) * 10,
    }


@pytest.fixture
def mock_metadata():
    """Create mock metadata."""
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
def mock_network():
    """Create mock river network."""
    network = gpd.GeoDataFrame(
        {
            "mobidic_id": [0, 1, 2],
            "upstream_1": [np.nan, np.nan, 0],
            "upstream_2": [np.nan, np.nan, 1],
            "downstream": [2, 2, -1],
            "strahler_order": [1, 1, 2],
            "length_m": [1000, 1200, 1500],
            "width_m": [1.0, 1.0, 2.83],
            "hillslope_cells": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
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
def mock_hillslope_reach_map():
    """Create mock hillslope-reach mapping."""
    return np.random.randint(-1, 3, size=(10, 10))


class TestGISData:
    """Tests for GISData class."""

    def test_gisdata_initialization(
        self, mock_grids, mock_metadata, mock_network, mock_hillslope_reach_map, mock_config
    ):
        """Test GISData initialization."""
        gisdata = GISData(
            grids=mock_grids,
            metadata=mock_metadata,
            network=mock_network,
            hillslope_reach_map=mock_hillslope_reach_map,
            config=mock_config,
        )

        assert gisdata.grids == mock_grids
        assert gisdata.metadata == mock_metadata
        assert isinstance(gisdata.network, gpd.GeoDataFrame)
        assert isinstance(gisdata.hillslope_reach_map, np.ndarray)
        assert gisdata.config == mock_config

    def test_gisdata_save(
        self, mock_grids, mock_metadata, mock_network, mock_hillslope_reach_map, mock_config, tmp_path
    ):
        """Test GISData save method."""
        gisdata = GISData(
            grids=mock_grids,
            metadata=mock_metadata,
            network=mock_network,
            hillslope_reach_map=mock_hillslope_reach_map,
            config=mock_config,
        )

        gisdata_path = tmp_path / "gisdata.nc"
        network_path = tmp_path / "network.parquet"

        with (
            patch("mobidic.preprocessing.io.save_gisdata") as mock_save_gisdata,
            patch("mobidic.preprocessing.io.save_network") as mock_save_network,
        ):
            gisdata.save(gisdata_path, network_path)

            mock_save_gisdata.assert_called_once()
            mock_save_network.assert_called_once()

    def test_gisdata_load(self, tmp_path):
        """Test GISData load method."""
        gisdata_path = tmp_path / "gisdata.nc"
        network_path = tmp_path / "network.parquet"

        # Create dummy files
        gisdata_path.touch()
        network_path.touch()

        with patch("mobidic.preprocessing.io.load_gisdata") as mock_load:
            mock_load.return_value = MagicMock(spec=GISData)

            result = GISData.load(gisdata_path, network_path)

            mock_load.assert_called_once_with(gisdata_path, network_path)
            assert result is not None


class TestPreprocessor:
    """Tests for preprocessing orchestrator."""

    def test_get_default_parameter_value(self, mock_config):
        """Test default parameter value retrieval."""
        assert _get_default_parameter_value("kf", mock_config) == 1e-7
        assert _get_default_parameter_value("CH", mock_config) == 1e-3
        assert _get_default_parameter_value("Alb", mock_config) == 0.2
        assert _get_default_parameter_value("gamma", mock_config) == 2.69e-7
        assert _get_default_parameter_value("Ma", mock_config) == 0.0
        assert _get_default_parameter_value("Mf", mock_config) == 1.0

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    @patch("mobidic.preprocessing.preprocessor.convert_to_mobidic_notation")
    @patch("mobidic.preprocessing.preprocessor.process_river_network")
    @patch("mobidic.preprocessing.preprocessor.compute_hillslope_cells")
    @patch("mobidic.preprocessing.preprocessor.map_hillslope_to_reach")
    def test_run_preprocessing_no_degradation(
        self,
        mock_map_hillslope,
        mock_compute_hillslope,
        mock_process_network,
        mock_convert_notation,
        mock_grid_to_matrix,
        mock_config,
        mock_network,
    ):
        """Test run_preprocessing with no grid degradation."""
        # Setup mocks
        mock_grid_to_matrix.return_value = {
            "data": np.random.rand(10, 10),
            "xllcorner": 5.0,
            "yllcorner": 5.0,
            "cellsize": 10.0,
            "crs": "EPSG:32632",
        }
        mock_convert_notation.return_value = np.random.randint(1, 9, size=(10, 10))
        mock_process_network.return_value = mock_network
        mock_compute_hillslope.return_value = mock_network
        mock_map_hillslope.return_value = np.random.randint(-1, 3, size=(10, 10))

        # Run preprocessing
        gisdata = run_preprocessing(mock_config)

        # Verify
        assert isinstance(gisdata, GISData)
        assert "dtm" in gisdata.grids
        assert "flow_dir" in gisdata.grids
        assert "Wc0" in gisdata.grids
        assert isinstance(gisdata.network, gpd.GeoDataFrame)
        assert isinstance(gisdata.hillslope_reach_map, np.ndarray)

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    @patch("mobidic.preprocessing.preprocessor.degrade_raster")
    @patch("mobidic.preprocessing.preprocessor.degrade_flow_direction")
    @patch("mobidic.preprocessing.preprocessor.convert_to_mobidic_notation")
    @patch("mobidic.preprocessing.preprocessor.process_river_network")
    @patch("mobidic.preprocessing.preprocessor.compute_hillslope_cells")
    @patch("mobidic.preprocessing.preprocessor.map_hillslope_to_reach")
    def test_run_preprocessing_with_degradation(
        self,
        mock_map_hillslope,
        mock_compute_hillslope,
        mock_process_network,
        mock_convert_notation,
        mock_degrade_flowdir,
        mock_degrade_raster,
        mock_grid_to_matrix,
        mock_config,
        mock_network,
    ):
        """Test run_preprocessing with grid degradation."""
        # Setup config with degradation
        mock_config.simulation.resample = 2

        # Setup mocks
        mock_grid_to_matrix.return_value = {
            "data": np.random.rand(20, 20),
            "xllcorner": 5.0,
            "yllcorner": 5.0,
            "cellsize": 10.0,
            "crs": "EPSG:32632",
        }
        mock_degrade_raster.return_value = np.random.rand(10, 10)
        mock_degrade_flowdir.return_value = (
            np.random.randint(1, 9, size=(10, 10)),
            np.random.randint(1, 100, size=(10, 10)),
        )
        mock_convert_notation.return_value = np.random.randint(1, 9, size=(10, 10))
        mock_process_network.return_value = mock_network
        mock_compute_hillslope.return_value = mock_network
        mock_map_hillslope.return_value = np.random.randint(-1, 3, size=(10, 10))

        # Run preprocessing
        gisdata = run_preprocessing(mock_config)

        # Verify degradation was called
        assert mock_degrade_raster.call_count > 0
        mock_degrade_flowdir.assert_called_once()

        # Verify result
        assert isinstance(gisdata, GISData)
        assert gisdata.metadata["shape"] == (10, 10)
        assert gisdata.metadata["resolution"] == (20.0, 20.0)
