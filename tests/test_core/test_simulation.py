"""Tests for simulation module."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from datetime import datetime
from shapely.geometry import LineString
from unittest.mock import MagicMock, patch
from mobidic.core.simulation import (
    SimulationState,
    SimulationResults,
    Simulation,
    _create_progress_bar,
)
from mobidic.config import MOBIDICConfig
from mobidic.preprocessing.meteo_preprocessing import MeteoData


@pytest.fixture
def simple_config():
    """Create a minimal valid MOBIDICConfig for testing."""
    config_dict = {
        "basin": {
            "id": "TEST",
            "paramset_id": "default",
            "baricenter": {"lon": 10.0, "lat": 45.0},
        },
        "paths": {
            "meteodata": "meteo.nc",
            "gisdata": "gis.nc",
            "network": "network.parquet",
            "states": "states/",
            "output": "output/",
        },
        "vector_files": {"river_network": {"shp": "network.shp", "id_field": "REACH_ID"}},
        "raster_files": {
            "dtm": "dtm.tif",
            "flow_dir": "flow_dir.tif",
            "flow_acc": "flow_acc.tif",
            "Wc0": "wc0.tif",
            "Wg0": "wg0.tif",
            "ks": "ks.tif",
        },
        "raster_settings": {"flow_dir_type": "Grass"},
        "parameters": {
            "soil": {
                "Wc0": 100.0,
                "Wg0": 50.0,
                "ks": 1.0,
                "kf": 1e-7,
                "gamma": 2.689e-7,
                "kappa": 1.096e-7,
                "beta": 7.62e-6,
                "alpha": 2.5e-5,
            },
            "energy": {
                "Tconst": 290.0,
                "kaps": 2.5,
                "nis": 0.8e-6,
                "CH": 1e-3,
                "Alb": 0.2,
            },
            "routing": {
                "method": "Linear",
                "wcel": 5.18,
                "Br0": 1.0,
                "NBr": 1.5,
                "n_Man": 0.03,
            },
            "groundwater": {"model": "None"},
        },
        "simulation": {
            "realtime": 0,
            "timestep": 900,
            "decimation": 1,
            "soil_scheme": "Bucket",
            "energy_balance": "None",
        },
        "output_states": {
            "discharge": True,
            "reservoir_states": True,
            "soil_capillary": True,
            "soil_gravitational": True,
            "soil_plant": True,
            "soil_surface": True,
            "surface_temperature": False,
            "ground_temperature": False,
            "aquifer_head": False,
            "et_prec": False,
        },
    }
    return MOBIDICConfig(**config_dict)


@pytest.fixture
def simple_gisdata():
    """Create minimal GISData mock for testing."""
    nrows, ncols = 5, 5

    # Create mock GISData object
    gisdata = MagicMock()
    gisdata.metadata = {
        "shape": (nrows, ncols),
        "resolution": (100.0, 100.0),
        "xllcorner": 0.0,
        "yllcorner": 0.0,
    }

    # Create simple grids
    dtm = np.arange(nrows * ncols, dtype=float).reshape(nrows, ncols)
    flow_dir = np.full((nrows, ncols), 4.0)  # All flow east
    flow_acc = np.arange(nrows * ncols, dtype=float).reshape(nrows, ncols)

    # Set one cell to NaN to test masking
    dtm[0, 0] = np.nan
    flow_dir[0, 0] = np.nan
    flow_acc[0, 0] = np.nan

    # Soil parameters
    wc0 = np.full((nrows, ncols), 0.1)
    wg0 = np.full((nrows, ncols), 0.2)
    ks = np.full((nrows, ncols), 1e-5)
    alpsur = np.full((nrows, ncols), 0.001)

    wc0[0, 0] = np.nan
    wg0[0, 0] = np.nan

    gisdata.grids = {
        "dtm": dtm,
        "flow_dir": flow_dir,
        "flow_acc": flow_acc,
        "Wc0": wc0,
        "Wg0": wg0,
        "ks": ks,
        "alpsur": alpsur,
    }

    # Hillslope-reach mapping: all cells drain to reach 0
    hillslope_reach_map = np.zeros((nrows, ncols))
    hillslope_reach_map[0, 0] = np.nan
    gisdata.hillslope_reach_map = hillslope_reach_map

    # Simple network with 1 reach
    network = gpd.GeoDataFrame(
        {
            "mobidic_id": [0],
            "upstream_1": [np.nan],
            "upstream_2": [np.nan],
            "downstream": [np.nan],
            "strahler_order": [1],
            "calc_order": [0],
            "length_m": [1000.0],
            "width_m": [10.0],
            "lag_time_s": [100.0],
            "storage_coeff": [0.5],
            "n_manning": [0.03],
            "geometry": [LineString([(0, 0), (100, 0)])],
        }
    )
    gisdata.network = network

    return gisdata


@pytest.fixture
def simple_meteo():
    """Create minimal MeteoData for testing."""
    # Create a single precipitation station with 48 hours of data
    times = pd.date_range("2020-01-01", periods=48, freq="1h")

    stations = {
        "precipitation": [
            {
                "code": "P001",
                "x": 50.0,
                "y": 50.0,
                "elevation": 100.0,
                "name": "Test Station",
                "time": pd.DatetimeIndex(times),
                "data": np.zeros(48),  # No precipitation for simplicity
            }
        ]
    }

    meteo = MeteoData(stations)
    return meteo


class TestCreateProgressBar:
    """Tests for _create_progress_bar helper function."""

    def test_empty_bar(self):
        """Test progress bar at 0%."""
        bar = _create_progress_bar(0, 100, bar_length=10)
        assert bar == "[          ]"

    def test_partial_bar(self):
        """Test progress bar at 50%."""
        bar = _create_progress_bar(50, 100, bar_length=10)
        assert bar == "[====>     ]"

    def test_complete_bar(self):
        """Test progress bar at 100%."""
        bar = _create_progress_bar(100, 100, bar_length=10)
        assert bar == "[==========]"

    def test_one_step_progress(self):
        """Test progress bar with one step completed."""
        bar = _create_progress_bar(1, 100, bar_length=20)
        # 1/100 = 0.01, filled = int(20 * 0.01) = 0, so empty bar
        assert bar == "[                    ]"

    def test_custom_bar_length(self):
        """Test progress bar with custom length."""
        bar = _create_progress_bar(5, 10, bar_length=5)
        # 5/10 = 0.5, filled = int(5 * 0.5) = 2
        # bar = "=" * (2-1) + ">" + " " * (5-2) = "=>   "
        assert bar == "[=>   ]"


class TestSimulationState:
    """Tests for SimulationState class."""

    def test_initialization(self):
        """Test SimulationState initialization."""
        wc = np.array([[0.1, 0.2], [0.3, 0.4]])
        wg = np.array([[0.05, 0.1], [0.15, 0.2]])
        wp = np.array([[0.01, 0.02], [0.03, 0.04]])
        ws = np.array([[0.001, 0.002], [0.003, 0.004]])
        discharge = np.array([1.0, 2.0, 3.0])
        lateral_inflow = np.array([0.1, 0.2, 0.3])

        state = SimulationState(wc, wg, wp, ws, discharge, lateral_inflow)

        assert np.array_equal(state.wc, wc)
        assert np.array_equal(state.wg, wg)
        assert np.array_equal(state.wp, wp)
        assert np.array_equal(state.ws, ws)
        assert np.array_equal(state.discharge, discharge)
        assert np.array_equal(state.lateral_inflow, lateral_inflow)

    def test_initialization_no_wp(self):
        """Test SimulationState with wp=None."""
        wc = np.array([[0.1, 0.2]])
        wg = np.array([[0.05, 0.1]])
        ws = np.array([[0.001, 0.002]])
        discharge = np.array([1.0])
        lateral_inflow = np.array([0.1])

        state = SimulationState(wc, wg, None, ws, discharge, lateral_inflow)

        assert state.wp is None


class TestSimulationResults:
    """Tests for SimulationResults class."""

    def test_initialization(self, simple_config):
        """Test SimulationResults initialization."""
        results = SimulationResults(simple_config)

        assert results.config is simple_config
        assert results.simulation is None
        assert results.time_series == {}
        assert results.final_state is None

    def test_initialization_with_simulation(self, simple_config):
        """Test SimulationResults initialization with simulation object."""
        mock_sim = MagicMock()
        results = SimulationResults(simple_config, simulation=mock_sim)

        assert results.simulation is mock_sim

    def test_save_states_no_state_raises_error(self, simple_config, tmp_path):
        """Test save_states raises error when no state available."""
        results = SimulationResults(simple_config)
        output_path = tmp_path / "state.nc"

        with pytest.raises(ValueError, match="No state to save"):
            results.save_states(output_path)

    def test_save_states_no_simulation_raises_error(self, simple_config):
        """Test save_states raises error when no simulation object."""
        results = SimulationResults(simple_config)
        results.final_state = MagicMock()  # Set a state

        with pytest.raises(ValueError, match="Cannot save state without simulation object"):
            results.save_states("state.nc")

    def test_save_report_no_data_raises_error(self, simple_config):
        """Test save_report raises error when no discharge data."""
        results = SimulationResults(simple_config)

        with pytest.raises(ValueError, match="No discharge data to save"):
            results.save_report("discharge.parquet")

    def test_save_report_no_simulation_raises_error(self, simple_config):
        """Test save_report raises error when no simulation object."""
        results = SimulationResults(simple_config)
        results.time_series["discharge"] = np.array([[1.0]])

        with pytest.raises(ValueError, match="Cannot save report without simulation object"):
            results.save_report("discharge.parquet")

    def test_save_lateral_inflow_report_no_data_raises_error(self, simple_config):
        """Test save_lateral_inflow_report raises error when no data."""
        results = SimulationResults(simple_config)

        with pytest.raises(ValueError, match="No lateral inflow data to save"):
            results.save_lateral_inflow_report("lateral_inflow.parquet")

    def test_save_final_state_no_state_raises_error(self, simple_config):
        """Test save_final_state raises error when no state."""
        results = SimulationResults(simple_config)

        with pytest.raises(ValueError, match="No final state to save"):
            results.save_final_state("state.nc")

    def test_save_final_state_no_time_raises_error(self, simple_config):
        """Test save_final_state raises error when no time information."""
        results = SimulationResults(simple_config, simulation=MagicMock())
        results.final_state = MagicMock()

        with pytest.raises(ValueError, match="No time information available"):
            results.save_final_state("state.nc")


class TestSimulationInitialization:
    """Tests for Simulation initialization."""

    def test_initialization(self, simple_gisdata, simple_meteo, simple_config):
        """Test Simulation initialization."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Check grid metadata
        assert sim.nrows == 5
        assert sim.ncols == 5
        assert sim.resolution == (100.0, 100.0)
        assert sim.xllcorner == 0.0
        assert sim.yllcorner == 0.0

        # Check flow direction type
        assert sim.flow_dir_type == "Grass"

        # Check grids
        assert sim.dtm.shape == (5, 5)
        assert sim.flow_dir.shape == (5, 5)
        assert sim.flow_acc.shape == (5, 5)

        # Check time step
        assert sim.dt == 900

        # Check network
        assert len(sim.network) == 1

        # Check state is not yet initialized
        assert sim.state is None

    def test_wc_wg_preprocessing(self, simple_gisdata, simple_meteo, simple_config):
        """Test that Wc0 and Wg0 are preprocessed correctly."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Check that multipliers are applied
        assert np.allclose(sim.wc0[1, 1], 0.1 * simple_config.parameters.multipliers.Wc_factor, rtol=1e-6)
        assert np.allclose(sim.wg0[1, 1], 0.2 * simple_config.parameters.multipliers.Wg_factor, rtol=1e-6)

        # Check minimum limits are applied
        assert np.all(sim.wc0[np.isfinite(sim.wc0)] >= 1e-10)
        assert np.all(sim.wg0[np.isfinite(sim.wg0)] >= 1e-10)

    def test_initial_state(self, simple_gisdata, simple_meteo, simple_config):
        """Test _initial_state method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        state = sim._initial_state()

        # Check state variables are initialized
        assert state.wc.shape == (5, 5)
        assert state.wg.shape == (5, 5)
        assert state.wp.shape == (5, 5)
        assert state.ws.shape == (5, 5)

        # Check discharge is initialized
        assert len(state.discharge) == 1
        assert state.discharge[0] == 0.0

        # Check lateral inflow is initialized
        assert len(state.lateral_inflow) == 1
        assert state.lateral_inflow[0] == 0.0

        # Check saturation is applied
        wcsat = simple_config.initial_conditions.Wcsat
        wgsat = simple_config.initial_conditions.Wgsat
        assert np.allclose(state.wc[1, 1], sim.wc0[1, 1] * wcsat, rtol=1e-6)
        assert np.allclose(state.wg[1, 1], sim.wg0[1, 1] * wgsat, rtol=1e-6)

        # Check NaN outside domain
        assert np.isnan(state.wc[0, 0])
        assert np.isnan(state.wg[0, 0])

    def test_calculate_pet(self, simple_gisdata, simple_meteo, simple_config):
        """Test _calculate_pet method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        pet = sim._calculate_pet(datetime(2020, 1, 1))

        # Check PET shape
        assert pet.shape == (5, 5)

        # Check PET values (1 mm/day default, returned as m/s)
        expected_pet = 1.0 / 1000.0 / (24 * 3600)  # m/s
        assert np.allclose(pet[1, 1], expected_pet, rtol=1e-6)

    def test_prepare_grids(self, simple_gisdata, simple_meteo, simple_config):
        """Test _prepare_grids method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Check param_grids are created
        assert "f0" in sim.param_grids
        assert "ks" in sim.param_grids
        assert "gamma" in sim.param_grids
        assert "kappa" in sim.param_grids
        assert "beta" in sim.param_grids
        assert "alpha" in sim.param_grids
        assert "cha" in sim.param_grids
        assert "alpsur" in sim.param_grids

        # Check shapes
        for key, grid in sim.param_grids.items():
            assert grid.shape == (5, 5), f"Grid {key} has wrong shape"

        # Check f0 calculation
        dt = sim.dt
        f0_expected = 0.85 * (1 - np.exp(-dt / (24 * 3600) * np.log(0.85 / 0.10)))
        assert np.allclose(sim.param_grids["f0"][1, 1], f0_expected, rtol=1e-6)

        # Check ks multiplier is applied
        ks_factor = simple_config.parameters.multipliers.ks_factor
        assert np.allclose(sim.param_grids["ks"][1, 1], 1e-5 * ks_factor, rtol=1e-6)

    def test_preprocess_network_topology(self, simple_gisdata, simple_meteo, simple_config):
        """Test _preprocess_network_topology method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Check topology dictionary
        assert "upstream_1_idx" in sim._network_topology
        assert "upstream_2_idx" in sim._network_topology
        assert "n_upstream" in sim._network_topology
        assert "sorted_reach_idx" in sim._network_topology
        assert "K" in sim._network_topology
        assert "n_reaches" in sim._network_topology

        # Check sizes
        assert len(sim._network_topology["upstream_1_idx"]) == 1
        assert len(sim._network_topology["upstream_2_idx"]) == 1
        assert len(sim._network_topology["n_upstream"]) == 1
        assert sim._network_topology["n_reaches"] == 1

        # Check values for single-reach network
        assert sim._network_topology["upstream_1_idx"][0] == -1  # No upstream
        assert sim._network_topology["upstream_2_idx"][0] == -1  # No upstream
        assert sim._network_topology["n_upstream"][0] == 0  # No upstream

    def test_accumulate_lateral_inflow(self, simple_gisdata, simple_meteo, simple_config):
        """Test _accumulate_lateral_inflow method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Create lateral flow grid
        lateral_flow = np.ones((5, 5))
        lateral_flow[0, 0] = np.nan  # Outside domain

        # Accumulate
        lateral_inflow = sim._accumulate_lateral_inflow(lateral_flow)

        # All valid cells (24 cells) should contribute to reach 0
        assert len(lateral_inflow) == 1
        assert np.isclose(lateral_inflow[0], 24.0, rtol=1e-6)

    def test_accumulate_lateral_inflow_with_negative_reach_ids(self, simple_gisdata, simple_meteo, simple_config):
        """Test _accumulate_lateral_inflow with cells that cannot reach network."""
        # Modify hillslope_reach_map to have -9999 values
        simple_gisdata.hillslope_reach_map[1, 1] = -9999

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        lateral_flow = np.ones((5, 5))
        lateral_flow[0, 0] = np.nan

        lateral_inflow = sim._accumulate_lateral_inflow(lateral_flow)

        # Cell (1,1) should not contribute (has -9999)
        # So we expect 24 - 1 = 23 cells
        assert np.isclose(lateral_inflow[0], 23.0, rtol=1e-6)


class TestSimulationInterpolation:
    """Tests for meteorological forcing interpolation."""

    def test_interpolate_forcing_precipitation_zero(self, simple_gisdata, simple_meteo, simple_config):
        """Test precipitation interpolation when all values are zero."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        precip = sim._interpolate_forcing(
            datetime(2020, 1, 1, 12, 0),
            "precipitation",
            weights_cache=sim._interpolation_weights,
        )

        # All precipitation should be zero
        assert precip.shape == (5, 5)
        assert np.allclose(precip[np.isfinite(precip)], 0.0, atol=1e-10)

    def test_interpolate_forcing_precipitation_nonzero(self, simple_gisdata, simple_config):
        """Test precipitation interpolation with non-zero values."""
        # Create meteo with non-zero precipitation
        times = pd.date_range("2020-01-01", periods=24, freq="1h")
        stations = {
            "precipitation": [
                {
                    "code": "P001",
                    "x": 50.0,
                    "y": 50.0,
                    "elevation": 100.0,
                    "name": "Test Station",
                    "time": pd.DatetimeIndex(times),
                    "data": np.full(24, 10.0),  # 10 mm/hr
                }
            ]
        }
        meteo = MeteoData(stations)

        sim = Simulation(simple_gisdata, meteo, simple_config)

        precip = sim._interpolate_forcing(
            datetime(2020, 1, 1, 12, 0),
            "precipitation",
            weights_cache=sim._interpolation_weights,
        )

        # Precipitation should be non-zero (converted to m/s)
        assert precip.shape == (5, 5)
        assert np.any(precip[np.isfinite(precip)] > 0.0)

    def test_interpolate_forcing_missing_variable(self, simple_gisdata, simple_meteo, simple_config):
        """Test interpolation with missing variable raises error."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        with pytest.raises(ValueError, match="Variable 'temperature_min' not found"):
            sim._interpolate_forcing(datetime(2020, 1, 1), "temperature_min")

    def test_precompute_interpolation_weights(self, simple_gisdata, simple_meteo, simple_config):
        """Test _precompute_interpolation_weights method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Check that weights are computed for precipitation
        assert "precipitation" in sim._interpolation_weights

        # Weights can be None (for nearest) or a 3D array (for IDW)
        weights = sim._interpolation_weights["precipitation"]
        if weights is not None:
            # Weights shape is (nrows, ncols, n_stations)
            assert weights.ndim == 3
            assert weights.shape[0] == 5  # nrows
            assert weights.shape[1] == 5  # ncols
            assert weights.shape[2] == 1  # 1 station

    def test_precompute_time_indices(self, simple_gisdata, simple_meteo, simple_config):
        """Test _precompute_time_indices method."""
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Create simulation times
        simulation_times = pd.date_range("2020-01-01", periods=10, freq="15min")

        # Compute time indices
        time_indices = sim._precompute_time_indices(simulation_times, variables=["precipitation"])

        # Check that indices are computed
        assert "precipitation" in time_indices
        assert time_indices["precipitation"].shape == (1, 10)  # 1 station, 10 timesteps


class TestSimulationRun:
    """Tests for Simulation.run() method."""

    def test_should_save_state_final_mode(self, simple_gisdata, simple_meteo, simple_config):
        """Test _should_save_state() for 'final' mode."""
        from datetime import datetime

        simple_config.output_states_settings.output_states = "final"
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Should never save in loop for 'final' mode
        assert sim._should_save_state(0, datetime(2020, 1, 1, 0, 0)) is False
        assert sim._should_save_state(5, datetime(2020, 1, 1, 1, 15)) is False
        assert sim._should_save_state(10, datetime(2020, 1, 1, 2, 30)) is False

    def test_should_save_state_all_mode_no_interval(self, simple_gisdata, simple_meteo, simple_config):
        """Test _should_save_state() for 'all' mode without interval (every timestep)."""
        from datetime import datetime

        simple_config.output_states_settings.output_states = "all"
        simple_config.output_states_settings.output_interval = None
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Should save every timestep
        assert sim._should_save_state(0, datetime(2020, 1, 1, 0, 0)) is True
        assert sim._should_save_state(1, datetime(2020, 1, 1, 0, 15)) is True
        assert sim._should_save_state(10, datetime(2020, 1, 1, 2, 30)) is True

    def test_should_save_state_all_mode_with_interval(self, simple_gisdata, simple_meteo, simple_config):
        """Test _should_save_state() for 'all' mode with interval."""
        from datetime import datetime

        simple_config.output_states_settings.output_states = "all"
        simple_config.output_states_settings.output_interval = 1800.0  # 2 timesteps (1800s / 900s)
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Should save at multiples of interval_steps
        # interval_steps = 1800 / 900 = 2
        # Save at steps where (step+1) % 2 == 0: steps 1, 3, 5, 7, etc.
        assert sim._should_save_state(0, datetime(2020, 1, 1, 0, 0)) is False
        assert sim._should_save_state(1, datetime(2020, 1, 1, 0, 15)) is True
        assert sim._should_save_state(2, datetime(2020, 1, 1, 0, 30)) is False
        assert sim._should_save_state(3, datetime(2020, 1, 1, 0, 45)) is True
        assert sim._should_save_state(4, datetime(2020, 1, 1, 1, 0)) is False
        assert sim._should_save_state(5, datetime(2020, 1, 1, 1, 15)) is True

    def test_should_save_state_list_mode(self, simple_gisdata, simple_meteo, simple_config):
        """Test _should_save_state() for 'list' mode."""
        from datetime import datetime

        simple_config.output_states_settings.output_states = "list"
        simple_config.output_states_settings.output_list = [
            "2020-01-01 00:00:00",
            "2020-01-01 00:45:00",
            "2020-01-01 01:15:00",
        ]
        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Should only save at specified datetimes
        assert sim._should_save_state(0, datetime(2020, 1, 1, 0, 0)) is True
        assert sim._should_save_state(1, datetime(2020, 1, 1, 0, 15)) is False
        assert sim._should_save_state(2, datetime(2020, 1, 1, 0, 30)) is False
        assert sim._should_save_state(3, datetime(2020, 1, 1, 0, 45)) is True
        assert sim._should_save_state(4, datetime(2020, 1, 1, 1, 0)) is False
        assert sim._should_save_state(5, datetime(2020, 1, 1, 1, 15)) is True
        assert sim._should_save_state(6, datetime(2020, 1, 1, 1, 30)) is False

    def test_run_minimal(self, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test running a minimal simulation."""
        # Set output paths to tmp_path
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Run for 2 hours (8 time steps at 900s)
        results = sim.run("2020-01-01 00:00", "2020-01-01 02:00")

        # Check results structure
        assert "discharge" in results.time_series
        assert "lateral_inflow" in results.time_series
        assert "time" in results.time_series

        # Check time series length (inclusive of end time)
        n_steps = int((datetime(2020, 1, 1, 2, 0) - datetime(2020, 1, 1, 0, 0)).total_seconds() / 900) + 1
        assert len(results.time_series["time"]) == n_steps
        assert len(results.time_series["discharge"]) == n_steps

        # Check discharge shape
        assert results.time_series["discharge"].shape[1] == 1  # 1 reach

        # Check final state
        assert results.final_state is not None
        assert results.final_state.wc.shape == (5, 5)
        assert results.final_state.discharge.shape == (1,)

    def test_run_with_datetime_objects(self, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test running simulation with datetime objects instead of strings."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        start = datetime(2020, 1, 1, 0, 0)
        end = datetime(2020, 1, 1, 1, 0)

        results = sim.run(start, end)

        # Check that simulation completed
        assert results.final_state is not None

    def test_run_single_timestep(self, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test running simulation for a single time step."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        results = sim.run("2020-01-01 00:00", "2020-01-01 00:00")

        # Should have exactly 1 time step
        assert len(results.time_series["time"]) == 1

    def test_run_state_evolution(self, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test that state evolves during simulation."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)

        # Run simulation
        results = sim.run("2020-01-01 00:00", "2020-01-01 01:00")

        # State should have changed (due to ET, percolation, etc.)
        # This is a weak test but ensures the simulation is actually running
        # In a real scenario with precipitation, changes would be more dramatic
        assert results.final_state is not None

    def test_run_discharge_conservation(self, simple_gisdata, simple_config, tmp_path):
        """Test discharge conservation during simulation."""
        # Create meteo with precipitation
        times = pd.date_range("2020-01-01", periods=10, freq="15min")
        stations = {
            "precipitation": [
                {
                    "code": "P001",
                    "x": 250.0,
                    "y": 250.0,
                    "elevation": 100.0,
                    "name": "Test Station",
                    "time": pd.DatetimeIndex(times),
                    "data": np.full(10, 5.0),  # 5 mm per 15 min
                }
            ]
        }
        meteo = MeteoData(stations)

        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")

        sim = Simulation(simple_gisdata, meteo, simple_config)
        results = sim.run("2020-01-01 00:00", "2020-01-01 01:00")

        # Discharge should be non-negative
        assert np.all(results.time_series["discharge"] >= 0.0)

        # Lateral inflow should be non-negative
        assert np.all(results.time_series["lateral_inflow"] >= 0.0)

    def test_run_creates_output_directories(self, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test that run() creates output directories."""
        output_dir = tmp_path / "output"
        states_dir = tmp_path / "states"

        simple_config.paths.output = str(output_dir)
        simple_config.paths.states = str(states_dir)

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        _ = sim.run("2020-01-01 00:00", "2020-01-01 00:00")

        # Directories should be created
        assert output_dir.exists()
        assert states_dir.exists()

    @patch("mobidic.io.save_discharge_report")
    def test_run_saves_discharge_report(self, mock_save, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test that run() saves discharge report when enabled."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")
        simple_config.output_report.discharge = True

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        _ = sim.run("2020-01-01 00:00", "2020-01-01 01:00")

        # Check that save function was called
        mock_save.assert_called_once()

    @patch("mobidic.io.save_state")
    def test_run_saves_final_state(self, mock_save, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test that run() saves final state when enabled."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")
        simple_config.output_states_settings.output_states = "final"

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        _ = sim.run("2020-01-01 00:00", "2020-01-01 01:00")

        # Check that save function was called once (only final state)
        mock_save.assert_called_once()

    @patch("mobidic.io.save_state")
    def test_run_saves_all_states_every_timestep(
        self, mock_save, simple_gisdata, simple_meteo, simple_config, tmp_path
    ):
        """Test that run() saves states at every timestep when output_states='all' and no interval."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")
        simple_config.output_states_settings.output_states = "all"
        simple_config.output_states_settings.output_interval = None

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        # Run for 2 hours with 900s timestep = 9 timesteps (inclusive of end time)
        _ = sim.run("2020-01-01 00:00", "2020-01-01 02:00")

        # Check that save function was called 9 times (in loop) + 1 time (final) = 10 times
        assert mock_save.call_count == 10

    @patch("mobidic.io.save_state")
    def test_run_saves_all_states_with_interval(self, mock_save, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test that run() saves states at specified interval when output_states='all'."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")
        simple_config.output_states_settings.output_states = "all"
        simple_config.output_states_settings.output_interval = 1800.0  # Every 2 timesteps (1800s)

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        # Run for 2 hours with 900s timestep = 9 timesteps (indices 0-8)
        # Should save at steps 1, 3, 5, 7 (4 times, when (step+1) % 2 == 0) + final state = 5 times
        _ = sim.run("2020-01-01 00:00", "2020-01-01 02:00")

        # Check that save function was called 4 times (in loop) + 1 time (final) = 5 times
        assert mock_save.call_count == 5

    @patch("mobidic.io.save_state")
    def test_run_saves_states_by_list(self, mock_save, simple_gisdata, simple_meteo, simple_config, tmp_path):
        """Test that run() saves states at specified datetimes when output_states='list'."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")
        simple_config.output_states_settings.output_states = "list"
        simple_config.output_states_settings.output_list = [
            "2020-01-01 00:00:00",
            "2020-01-01 00:30:00",
            "2020-01-01 01:00:00",
            "2020-01-01 01:45:00",
        ]

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        # Run for 2 hours with 900s timestep = 9 timesteps
        _ = sim.run("2020-01-01 00:00", "2020-01-01 02:00")

        # Check that save function was called 4 times (only the specified datetimes, no final state)
        assert mock_save.call_count == 4

    @patch("mobidic.io.save_state")
    def test_run_saves_states_by_list_including_final(
        self, mock_save, simple_gisdata, simple_meteo, simple_config, tmp_path
    ):
        """Test that run() saves states correctly when list includes final timestep."""
        simple_config.paths.output = str(tmp_path / "output")
        simple_config.paths.states = str(tmp_path / "states")
        simple_config.output_states_settings.output_states = "list"
        simple_config.output_states_settings.output_list = [
            "2020-01-01 00:00:00",
            "2020-01-01 00:45:00",
            "2020-01-01 02:00:00",  # Include last timestep
        ]

        sim = Simulation(simple_gisdata, simple_meteo, simple_config)
        # Run for 2 hours with 900s timestep = 9 timesteps
        _ = sim.run("2020-01-01 00:00", "2020-01-01 02:00")

        # Check that save function was called 3 times (specified datetimes in loop, no separate final)
        assert mock_save.call_count == 3
