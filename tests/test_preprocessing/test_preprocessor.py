"""Tests for preprocessing orchestrator module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import geopandas as gpd
from shapely.geometry import LineString

from mobidic.preprocessing.gis_reader import GridAlignmentError
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

    config.simulation.decimation = 1

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

            mock_load.assert_called_once_with(gisdata_path, network_path, reservoirs_path=None)
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
    @patch("mobidic.preprocessing.preprocessor.decimate_raster")
    @patch("mobidic.preprocessing.preprocessor.decimate_flow_direction")
    @patch("mobidic.preprocessing.preprocessor.convert_to_mobidic_notation")
    @patch("mobidic.preprocessing.preprocessor.process_river_network")
    @patch("mobidic.preprocessing.preprocessor.compute_hillslope_cells")
    @patch("mobidic.preprocessing.preprocessor.map_hillslope_to_reach")
    def test_run_preprocessing_with_decimation(
        self,
        mock_map_hillslope,
        mock_compute_hillslope,
        mock_process_network,
        mock_convert_notation,
        mock_decimate_flowdir,
        mock_decimate_raster,
        mock_grid_to_matrix,
        mock_config,
        mock_network,
    ):
        """Test run_preprocessing with grid decimation."""
        # Setup config with decimation
        mock_config.simulation.decimation = 2

        # Setup mocks
        mock_grid_to_matrix.return_value = {
            "data": np.random.rand(20, 20),
            "xllcorner": 5.0,
            "yllcorner": 5.0,
            "cellsize": 10.0,
            "crs": "EPSG:32632",
        }
        mock_decimate_raster.return_value = np.random.rand(10, 10)
        mock_decimate_flowdir.return_value = (
            np.random.randint(1, 9, size=(10, 10)),
            np.random.randint(1, 100, size=(10, 10)),
        )
        mock_convert_notation.return_value = np.random.randint(1, 9, size=(10, 10))
        mock_process_network.return_value = mock_network
        mock_compute_hillslope.return_value = mock_network
        mock_map_hillslope.return_value = np.random.randint(-1, 3, size=(10, 10))

        # Run preprocessing
        gisdata = run_preprocessing(mock_config)

        # Verify decimation was called
        assert mock_decimate_raster.call_count > 0
        mock_decimate_flowdir.assert_called_once()

        # Verify result
        assert isinstance(gisdata, GISData)
        assert gisdata.metadata["shape"] == (10, 10)
        assert gisdata.metadata["resolution"] == (20.0, 20.0)


def _grid(shape=(10, 10), xll=5.0, crs="EPSG:32632"):
    """Mock grid_to_matrix() result."""
    return {"data": np.random.rand(*shape), "xllcorner": xll, "yllcorner": 5.0, "cellsize": 10.0, "crs": crs}


class TestPreprocessorInputChecks:
    """Tests for CRS and grid alignment checks in run_preprocessing."""

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    def test_raster_mismatch_names_file(self, mock_grid_to_matrix, mock_config):
        """Test that a raster off the DTM grid stops preprocessing with its path in the error."""
        mock_grid_to_matrix.side_effect = [_grid(), _grid(shape=(12, 10))]

        with pytest.raises(GridAlignmentError, match=r"flowdir\.tif does not match the DTM grid \(dtm\.tif\)"):
            run_preprocessing(mock_config)

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    def test_raster_crs_mismatch(self, mock_grid_to_matrix, mock_config):
        """Test that a raster on the same grid but in another CRS is rejected."""
        mock_grid_to_matrix.side_effect = [_grid(), _grid(crs="EPSG:3003")]

        with pytest.raises(GridAlignmentError, match="CRS EPSG:3003 != EPSG:32632"):
            run_preprocessing(mock_config)

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    def test_geographic_dtm_rejected(self, mock_grid_to_matrix, mock_config):
        """Test that a DTM with a geographic CRS is rejected."""
        mock_grid_to_matrix.return_value = _grid(crs="EPSG:4326")

        with pytest.raises(GridAlignmentError, match=r"dtm\.tif has a geographic CRS"):
            run_preprocessing(mock_config)

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    @patch("mobidic.preprocessing.preprocessor.convert_to_mobidic_notation")
    @patch("mobidic.preprocessing.preprocessor.process_river_network")
    @patch("mobidic.preprocessing.preprocessor.compute_hillslope_cells")
    @patch("mobidic.preprocessing.preprocessor.map_hillslope_to_reach")
    @patch("mobidic.preprocessing.preprocessor.process_reservoirs")
    def test_dtm_crs_passed_to_vectors_and_reservoir_error_raised(
        self,
        mock_process_reservoirs,
        mock_map_hillslope,
        mock_compute_hillslope,
        mock_process_network,
        mock_convert_notation,
        mock_grid_to_matrix,
        mock_config,
        mock_network,
    ):
        """Test that the DTM CRS is the reference for the shapefiles and a reservoir CRS error is not swallowed."""
        mock_grid_to_matrix.return_value = _grid()
        mock_convert_notation.return_value = np.random.randint(1, 9, size=(10, 10))
        mock_process_network.return_value = mock_network
        mock_compute_hillslope.return_value = mock_network
        mock_map_hillslope.return_value = np.random.randint(-1, 3, size=(10, 10))
        mock_config.parameters.reservoirs.res_shape = "reservoirs.shp"
        mock_process_reservoirs.side_effect = GridAlignmentError("reservoirs.shp has CRS EPSG:3003")

        with pytest.raises(GridAlignmentError, match="reservoirs.shp"):
            run_preprocessing(mock_config)

        assert mock_process_network.call_args.kwargs["reference_crs"] == "EPSG:32632"
        assert mock_process_reservoirs.call_args.kwargs["reference_crs"] == "EPSG:32632"


def _capture_warnings(func, *args, **kwargs):
    """Call func and return its result and the loguru WARNING messages it emitted."""
    from loguru import logger

    messages = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        result = func(*args, **kwargs)
    finally:
        logger.remove(handler_id)
    return result, messages


class TestNanInDomain:
    """Tests for NaN handling inside the model domain."""

    def test_fill_nan_in_domain(self):
        """NaN cells inside the domain are replaced, the ones outside are kept."""
        from mobidic.preprocessing.preprocessor import _fill_nan_in_domain

        data = np.array([[1.0, np.nan], [np.nan, 4.0]])
        domain = np.array([[True, True], [False, True]])

        filled, messages = _capture_warnings(_fill_nan_in_domain, data, domain, 9.0, "ks.tif", "parameters.soil.ks")

        np.testing.assert_array_equal(filled, [[1.0, 9.0], [np.nan, 4.0]])
        assert np.isnan(data[0, 1])  # input not modified
        assert any(
            "ks.tif: 1 NaN cells" in m and "replaced with the default parameters.soil.ks = 9.0" in m for m in messages
        )

    def test_fill_nan_in_domain_no_nan(self):
        """No warning when there is nothing to replace."""
        from mobidic.preprocessing.preprocessor import _fill_nan_in_domain

        data = np.ones((2, 2))
        filled, messages = _capture_warnings(_fill_nan_in_domain, data, np.ones((2, 2), bool), 9.0, "ks.tif", "x")
        assert filled is data
        assert messages == []

    def test_restrict_domain(self):
        """Cells where flow_dir or flow_acc is NaN are excluded from the domain, with a warning."""
        from mobidic.preprocessing.preprocessor import _restrict_domain

        grids = {
            "dtm": np.array([[1.0, 2.0, 3.0, np.nan]]),
            "flow_dir": np.array([[1.0, np.nan, 3.0, 4.0]]),
            "flow_acc": np.array([[1.0, 2.0, np.nan, 4.0]]),
        }
        domain, messages = _capture_warnings(_restrict_domain, grids, {"flow_dir": "fd.tif", "flow_acc": "fa.tif"})

        np.testing.assert_array_equal(domain, [[True, False, False, False]])
        for name in ("dtm", "flow_dir", "flow_acc"):
            assert np.isnan(grids[name][0, 1:]).all()
        assert any("fd.tif: 1 NaN cells inside the DTM domain" in m for m in messages)
        assert any("fa.tif: 1 NaN cells inside the DTM domain" in m for m in messages)

    @patch("mobidic.preprocessing.preprocessor.grid_to_matrix")
    @patch("mobidic.preprocessing.preprocessor.convert_to_mobidic_notation")
    @patch("mobidic.preprocessing.preprocessor.process_river_network")
    @patch("mobidic.preprocessing.preprocessor.compute_hillslope_cells")
    @patch("mobidic.preprocessing.preprocessor.map_hillslope_to_reach")
    def test_run_preprocessing_fills_parameters(
        self,
        mock_map_hillslope,
        mock_compute_hillslope,
        mock_process_network,
        mock_convert_notation,
        mock_grid_to_matrix,
        mock_config,
        mock_network,
    ):
        """NaN parameter cells inside the domain get the YAML default, converted to model units."""
        mock_config.parameters.soil.Wc0 = 100.0  # mm
        mock_config.parameters.soil.Wg0 = 200.0  # mm
        mock_config.parameters.soil.ks = 3.6  # mm/h
        mock_config.raster_files.gamma = "gamma.tif"
        mock_config.raster_files.CLC = None

        def grid(nan_cell=None):
            g = _grid()
            g["data"] = np.full((10, 10), 5.0)
            if nan_cell is not None:
                g["data"][nan_cell] = np.nan
            return g

        # Read order: dtm, flow_dir, flow_acc, Wc0, Wg0, ks, gamma
        mock_grid_to_matrix.side_effect = [
            grid(),
            grid(),
            grid(),
            grid((1, 1)),
            grid((2, 2)),
            grid((3, 3)),
            grid((4, 4)),
        ]
        mock_convert_notation.side_effect = lambda fd, from_notation: fd
        mock_process_network.return_value = mock_network
        mock_compute_hillslope.return_value = mock_network
        mock_map_hillslope.return_value = np.zeros((10, 10))

        gisdata, messages = _capture_warnings(run_preprocessing, mock_config)

        assert gisdata.grids["Wc0"][1, 1] == pytest.approx(0.1)  # 100 mm -> m
        assert gisdata.grids["Wg0"][2, 2] == pytest.approx(0.2)  # 200 mm -> m
        assert gisdata.grids["ks"][3, 3] == pytest.approx(1e-6)  # 3.6 mm/h -> m/s
        assert gisdata.grids["gamma"][4, 4] == pytest.approx(mock_config.parameters.soil.gamma)
        assert sum("NaN values are replaced with the default" in m for m in messages) == 4
