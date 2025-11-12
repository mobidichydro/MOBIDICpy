"""Tests for mobidic.io.state module."""

from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from mobidic.io.state import save_state, load_state
from mobidic.core.simulation import SimulationState
from mobidic.config.schema import OutputStates


@pytest.fixture
def sample_grid_metadata():
    """Create sample grid metadata."""
    return {
        "shape": (10, 15),  # 10 rows, 15 columns
        "resolution": (100.0, 100.0),  # 100m resolution
        "xllcorner": 1000000.0,
        "yllcorner": 2000000.0,
        "crs": "EPSG:32632",  # UTM Zone 32N
    }


@pytest.fixture
def sample_state(sample_grid_metadata):
    """Create sample simulation state."""
    nrows, ncols = sample_grid_metadata["shape"]
    n_reaches = 5

    # Create state variables with realistic values
    wc = np.random.rand(nrows, ncols) * 0.2  # Capillary water [m]
    wg = np.random.rand(nrows, ncols) * 0.1  # Gravitational water [m]
    wp = np.random.rand(nrows, ncols) * 0.005  # Plant water [m]
    ws = np.random.rand(nrows, ncols) * 0.01  # Surface water [m]
    discharge = np.random.rand(n_reaches) * 10.0  # Discharge [m³/s]
    lateral_inflow = np.random.rand(n_reaches) * 2.0  # Lateral inflow [m³/s]

    return SimulationState(wc, wg, wp, ws, discharge, lateral_inflow)


@pytest.fixture
def all_states_enabled():
    """OutputStates configuration with all states enabled."""
    return OutputStates(
        discharge=True,
        reservoir_states=False,
        soil_capillary=True,
        soil_gravitational=True,
        soil_plant=True,
        soil_surface=True,
        surface_temperature=False,
        ground_temperature=False,
        aquifer_head=False,
        et_prec=False,
    )


@pytest.fixture
def minimal_states():
    """OutputStates configuration with minimal states."""
    return OutputStates(
        discharge=False,
        reservoir_states=False,
        soil_capillary=True,
        soil_gravitational=True,
        soil_plant=False,
        soil_surface=False,
        surface_temperature=False,
        ground_temperature=False,
        aquifer_head=False,
        et_prec=False,
    )


class TestSaveState:
    """Tests for save_state function."""

    def test_save_all_states(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test saving state with all variables enabled."""
        output_path = tmp_path / "state_all.nc"
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify
        ds = xr.open_dataset(output_path)

        # Check all variables are present
        assert "Wc" in ds
        assert "Wg" in ds
        assert "Wp" in ds
        assert "Ws" in ds
        assert "discharge" in ds
        assert "lateral_inflow" in ds
        assert "crs" in ds

        # Check dimensions
        assert ds.Wc.shape == (10, 15)
        assert ds.discharge.shape == (5,)

        # Check coordinates
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert "reach" in ds.coords
        assert "time" in ds.coords

        # Check data values
        np.testing.assert_array_almost_equal(ds.Wc.values, sample_state.wc)
        np.testing.assert_array_almost_equal(ds.discharge.values, sample_state.discharge)

        ds.close()

    def test_save_minimal_states(self, tmp_path, sample_state, sample_grid_metadata, minimal_states):
        """Test saving with only required states."""
        output_path = tmp_path / "state_minimal.nc"
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=minimal_states,
        )

        # Load and verify
        ds = xr.open_dataset(output_path)

        # Check only minimal variables are present
        assert "Wc" in ds
        assert "Wg" in ds
        assert "Wp" not in ds
        assert "Ws" not in ds
        assert "discharge" not in ds
        assert "lateral_inflow" not in ds

        # Check reach coordinate not present when discharge disabled
        assert "reach" not in ds.coords

        ds.close()

    def test_save_without_plant_water(self, tmp_path, sample_grid_metadata, all_states_enabled):
        """Test saving state when plant water is None."""
        nrows, ncols = sample_grid_metadata["shape"]
        n_reaches = 5

        # Create state without plant water
        state = SimulationState(
            wc=np.random.rand(nrows, ncols) * 0.2,
            wg=np.random.rand(nrows, ncols) * 0.1,
            wp=None,  # No plant water
            ws=np.random.rand(nrows, ncols) * 0.01,
            discharge=np.random.rand(n_reaches) * 10.0,
            lateral_inflow=np.random.rand(n_reaches) * 2.0,
        )

        output_path = tmp_path / "state_no_wp.nc"
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=n_reaches,
            output_states=all_states_enabled,
        )

        # Load and verify
        ds = xr.open_dataset(output_path)

        # Wp should not be present even though soil_plant is True
        assert "Wp" not in ds
        assert "Wc" in ds
        assert "Wg" in ds

        ds.close()

    def test_save_with_metadata(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test saving with additional metadata."""
        output_path = tmp_path / "state_metadata.nc"
        time = datetime(2020, 6, 15, 12, 0)

        metadata = {
            "basin": "Test Basin",
            "model_version": "0.0.1",
            "notes": "Test simulation",
        }

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
            add_metadata=metadata,
        )

        # Load and check global attributes
        ds = xr.open_dataset(output_path)

        assert ds.attrs["basin"] == "Test Basin"
        assert ds.attrs["model_version"] == "0.0.1"
        assert ds.attrs["notes"] == "Test simulation"

        # Check standard attributes
        assert ds.attrs["Conventions"] == "CF-1.12"
        assert ds.attrs["title"] == "MOBIDIC simulation state"
        assert ds.attrs["simulation_time"] == time.isoformat()

        ds.close()

    def test_save_creates_parent_directory(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test that save_state creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dirs" / "state.nc"
        time = datetime(2020, 6, 15, 12, 0)

        assert not output_path.parent.exists()

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_save_string_path(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test that function accepts string paths."""
        output_path = str(tmp_path / "state.nc")
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        assert Path(output_path).exists()

    def test_cf_compliant_metadata(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test that output is CF-1.12 compliant."""
        output_path = tmp_path / "state_cf.nc"
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        ds = xr.open_dataset(output_path)

        # Check CF conventions
        assert ds.attrs["Conventions"] == "CF-1.12"

        # Check coordinate attributes
        assert ds.x.attrs["standard_name"] == "projection_x_coordinate"
        assert ds.x.attrs["axis"] == "X"
        assert ds.y.attrs["standard_name"] == "projection_y_coordinate"
        assert ds.y.attrs["axis"] == "Y"

        # Check variable attributes include grid_mapping
        assert ds.Wc.attrs["grid_mapping"] == "crs"
        assert ds.Wc.attrs["units"] == "m"

        # Check CRS variable exists
        assert "crs" in ds
        assert "crs_wkt" in ds.crs.attrs

        ds.close()

    def test_compression_enabled(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test that NetCDF compression is enabled."""
        output_path = tmp_path / "state_compressed.nc"
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Open and check encoding
        ds = xr.open_dataset(output_path)

        # NetCDF4 with compression should be used
        assert ds is not None

        # File should exist and be reasonably sized
        file_size = output_path.stat().st_size
        assert file_size > 0
        assert file_size < 1000000  # Should be less than 1MB with compression

        ds.close()

    def test_coordinate_arrays(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test that coordinate arrays are correctly computed."""
        output_path = tmp_path / "state_coords.nc"
        time = datetime(2020, 6, 15, 12, 0)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        ds = xr.open_dataset(output_path)

        # Check x coordinates
        expected_x = sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0
        np.testing.assert_array_almost_equal(ds.x.values, expected_x)

        # Check y coordinates
        expected_y = sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0
        np.testing.assert_array_almost_equal(ds.y.values, expected_y)

        # Check reach coordinates
        np.testing.assert_array_equal(ds.reach.values, np.arange(5))

        ds.close()


class TestLoadState:
    """Tests for load_state function."""

    def test_load_basic(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test basic loading of state."""
        output_path = tmp_path / "state.nc"
        time = datetime(2020, 6, 15, 12, 0)

        # Save first
        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load
        loaded_state, loaded_time, loaded_metadata = load_state(output_path, network_size=5)

        # Check state variables
        assert isinstance(loaded_state, SimulationState)
        np.testing.assert_array_almost_equal(loaded_state.wc, sample_state.wc)
        np.testing.assert_array_almost_equal(loaded_state.wg, sample_state.wg)
        np.testing.assert_array_almost_equal(loaded_state.wp, sample_state.wp)
        np.testing.assert_array_almost_equal(loaded_state.ws, sample_state.ws)
        np.testing.assert_array_almost_equal(loaded_state.discharge, sample_state.discharge)
        np.testing.assert_array_almost_equal(loaded_state.lateral_inflow, sample_state.lateral_inflow)

        # Check time
        assert loaded_time == time

        # Check metadata
        assert loaded_metadata["shape"] == sample_grid_metadata["shape"]
        assert loaded_metadata["resolution"] == sample_grid_metadata["resolution"]

    def test_load_file_not_found(self, tmp_path):
        """Test error handling when file doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent.nc"

        with pytest.raises(FileNotFoundError, match="State file not found"):
            load_state(nonexistent_path, network_size=5)

    def test_load_string_path(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test loading with string path."""
        output_path = tmp_path / "state.nc"
        time = datetime(2020, 6, 15, 12, 0)

        # Save
        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load using string path
        loaded_state, loaded_time, loaded_metadata = load_state(str(output_path), network_size=5)

        assert isinstance(loaded_state, SimulationState)
        assert loaded_time == time

    def test_load_missing_wc_variable(self, tmp_path, sample_grid_metadata):
        """Test loading when Wc is missing (should initialize with NaN)."""
        # Create a NetCDF file without Wc
        ds = xr.Dataset(
            {
                "Wg": (["y", "x"], np.random.rand(10, 15)),
                "crs": ([], 0),
            },
            coords={
                "x": sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0,
                "y": sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0,
                "time": datetime(2020, 6, 15),
            },
        )

        output_path = tmp_path / "state_no_wc.nc"
        ds.to_netcdf(output_path)
        ds.close()

        # Load - should handle missing Wc gracefully
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # Wc should be all NaN
        assert np.all(np.isnan(loaded_state.wc))

        # Wg should be loaded correctly
        assert not np.all(np.isnan(loaded_state.wg))

    def test_load_missing_wg_variable(self, tmp_path, sample_grid_metadata):
        """Test loading when Wg is missing (should initialize with NaN)."""
        # Create a NetCDF file without Wg
        ds = xr.Dataset(
            {
                "Wc": (["y", "x"], np.random.rand(10, 15)),
                "crs": ([], 0),
            },
            coords={
                "x": sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0,
                "y": sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0,
                "time": datetime(2020, 6, 15),
            },
        )

        output_path = tmp_path / "state_no_wg.nc"
        ds.to_netcdf(output_path)
        ds.close()

        # Load - should handle missing Wg gracefully
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # Wg should be all NaN
        assert np.all(np.isnan(loaded_state.wg))

        # Wc should be loaded correctly
        assert not np.all(np.isnan(loaded_state.wc))

    def test_load_missing_ws_variable(self, tmp_path, sample_grid_metadata):
        """Test loading when Ws is missing (should initialize with NaN)."""
        # Create a NetCDF file without Ws
        ds = xr.Dataset(
            {
                "Wc": (["y", "x"], np.random.rand(10, 15)),
                "Wg": (["y", "x"], np.random.rand(10, 15)),
                "crs": ([], 0),
            },
            coords={
                "x": sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0,
                "y": sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0,
                "time": datetime(2020, 6, 15),
            },
        )

        output_path = tmp_path / "state_no_ws.nc"
        ds.to_netcdf(output_path)
        ds.close()

        # Load - should handle missing Ws gracefully
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # Ws should be all NaN
        assert np.all(np.isnan(loaded_state.ws))

    def test_load_missing_discharge(self, tmp_path, sample_grid_metadata):
        """Test loading when discharge is missing (should initialize with zeros)."""
        # Create a NetCDF file without discharge
        ds = xr.Dataset(
            {
                "Wc": (["y", "x"], np.random.rand(10, 15)),
                "Wg": (["y", "x"], np.random.rand(10, 15)),
                "crs": ([], 0),
            },
            coords={
                "x": sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0,
                "y": sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0,
                "time": datetime(2020, 6, 15),
            },
        )

        output_path = tmp_path / "state_no_discharge.nc"
        ds.to_netcdf(output_path)
        ds.close()

        # Load
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # Discharge should be all zeros
        np.testing.assert_array_equal(loaded_state.discharge, np.zeros(5))
        np.testing.assert_array_equal(loaded_state.lateral_inflow, np.zeros(5))

    def test_load_missing_lateral_inflow(self, tmp_path, sample_grid_metadata):
        """Test loading when lateral_inflow is missing (should initialize with zeros)."""
        # Create a NetCDF file with discharge but no lateral_inflow
        ds = xr.Dataset(
            {
                "Wc": (["y", "x"], np.random.rand(10, 15)),
                "Wg": (["y", "x"], np.random.rand(10, 15)),
                "discharge": (["reach"], np.random.rand(5) * 10.0),
                "crs": ([], 0),
            },
            coords={
                "x": sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0,
                "y": sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0,
                "reach": np.arange(5),
                "time": datetime(2020, 6, 15),
            },
        )

        output_path = tmp_path / "state_no_lateral.nc"
        ds.to_netcdf(output_path)
        ds.close()

        # Load
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # lateral_inflow should be all zeros
        np.testing.assert_array_equal(loaded_state.lateral_inflow, np.zeros(5))

        # discharge should be loaded correctly
        assert not np.all(loaded_state.discharge == 0.0)

    def test_load_missing_wp(self, tmp_path, sample_grid_metadata):
        """Test loading when Wp is missing (should be None)."""
        # Create a NetCDF file without Wp
        ds = xr.Dataset(
            {
                "Wc": (["y", "x"], np.random.rand(10, 15)),
                "Wg": (["y", "x"], np.random.rand(10, 15)),
                "crs": ([], 0),
            },
            coords={
                "x": sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0,
                "y": sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0,
                "time": datetime(2020, 6, 15),
            },
        )

        output_path = tmp_path / "state_no_wp.nc"
        ds.to_netcdf(output_path)
        ds.close()

        # Load
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # Wp should be None
        assert loaded_state.wp is None

    def test_load_network_size_mismatch(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test warning when network size doesn't match."""
        output_path = tmp_path / "state.nc"
        time = datetime(2020, 6, 15, 12, 0)

        # Save with 5 reaches
        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load with different network size (should work but warn)
        loaded_state, _, _ = load_state(output_path, network_size=10)

        # Should still load, discharge array will have wrong size
        assert len(loaded_state.discharge) == 5  # Still the saved size

    def test_round_trip_preserves_data(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test that save/load round trip preserves data exactly."""
        output_path = tmp_path / "state_roundtrip.nc"
        time = datetime(2020, 6, 15, 12, 30, 45)

        # Save
        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load
        loaded_state, loaded_time, loaded_metadata = load_state(output_path, network_size=5)

        # Verify exact data match (within floating point precision)
        np.testing.assert_array_almost_equal(loaded_state.wc, sample_state.wc, decimal=10)
        np.testing.assert_array_almost_equal(loaded_state.wg, sample_state.wg, decimal=10)
        np.testing.assert_array_almost_equal(loaded_state.wp, sample_state.wp, decimal=10)
        np.testing.assert_array_almost_equal(loaded_state.ws, sample_state.ws, decimal=10)
        np.testing.assert_array_almost_equal(loaded_state.discharge, sample_state.discharge, decimal=10)
        np.testing.assert_array_almost_equal(loaded_state.lateral_inflow, sample_state.lateral_inflow, decimal=10)

        # Verify time (exact match)
        assert loaded_time == time

        # Verify metadata
        assert loaded_metadata["shape"] == sample_grid_metadata["shape"]
        np.testing.assert_almost_equal(loaded_metadata["resolution"][0], sample_grid_metadata["resolution"][0])
        np.testing.assert_almost_equal(loaded_metadata["resolution"][1], sample_grid_metadata["resolution"][1])

    def test_load_minimal_states(self, tmp_path, sample_state, sample_grid_metadata, minimal_states):
        """Test loading file with minimal states."""
        output_path = tmp_path / "state_minimal.nc"
        time = datetime(2020, 6, 15, 12, 0)

        # Save with minimal states
        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=minimal_states,
        )

        # Load
        loaded_state, _, _ = load_state(output_path, network_size=5)

        # Should have Wc and Wg
        assert not np.all(np.isnan(loaded_state.wc))
        assert not np.all(np.isnan(loaded_state.wg))

        # Should initialize missing variables
        assert loaded_state.wp is None
        assert np.all(np.isnan(loaded_state.ws))
        np.testing.assert_array_equal(loaded_state.discharge, np.zeros(5))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_cell_grid(self, tmp_path, all_states_enabled):
        """Test with a 1x1 grid."""
        grid_metadata = {
            "shape": (1, 1),
            "resolution": (100.0, 100.0),
            "xllcorner": 0.0,
            "yllcorner": 0.0,
            "crs": "EPSG:32632",
        }

        state = SimulationState(
            wc=np.array([[0.1]]),
            wg=np.array([[0.05]]),
            wp=np.array([[0.002]]),
            ws=np.array([[0.001]]),
            discharge=np.array([1.5]),
            lateral_inflow=np.array([0.3]),
        )

        output_path = tmp_path / "state_single_cell.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=grid_metadata,
            network_size=1,
            output_states=all_states_enabled,
        )

        # Load and verify
        loaded_state, _, loaded_metadata = load_state(output_path, network_size=1)

        assert loaded_metadata["shape"] == (1, 1)
        assert loaded_state.wc.shape == (1, 1)
        np.testing.assert_almost_equal(loaded_state.wc[0, 0], 0.1)

    def test_large_grid(self, tmp_path, all_states_enabled):
        """Test with a larger grid."""
        grid_metadata = {
            "shape": (100, 200),
            "resolution": (50.0, 50.0),
            "xllcorner": 500000.0,
            "yllcorner": 4000000.0,
            "crs": "EPSG:32632",
        }

        state = SimulationState(
            wc=np.random.rand(100, 200) * 0.2,
            wg=np.random.rand(100, 200) * 0.1,
            wp=np.random.rand(100, 200) * 0.005,
            ws=np.random.rand(100, 200) * 0.01,
            discharge=np.random.rand(50) * 10.0,
            lateral_inflow=np.random.rand(50) * 2.0,
        )

        output_path = tmp_path / "state_large.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=grid_metadata,
            network_size=50,
            output_states=all_states_enabled,
        )

        # Check file exists and has reasonable size
        assert output_path.exists()
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        assert file_size_mb < 10.0  # Should be compressed

        # Load and verify shape
        loaded_state, _, _ = load_state(output_path, network_size=50)
        assert loaded_state.wc.shape == (100, 200)
        assert len(loaded_state.discharge) == 50

    def test_zero_values(self, tmp_path, sample_grid_metadata, all_states_enabled):
        """Test with all zero state values."""
        nrows, ncols = sample_grid_metadata["shape"]

        state = SimulationState(
            wc=np.zeros((nrows, ncols)),
            wg=np.zeros((nrows, ncols)),
            wp=np.zeros((nrows, ncols)),
            ws=np.zeros((nrows, ncols)),
            discharge=np.zeros(5),
            lateral_inflow=np.zeros(5),
        )

        output_path = tmp_path / "state_zeros.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load and verify
        loaded_state, _, _ = load_state(output_path, network_size=5)

        assert np.all(loaded_state.wc == 0.0)
        assert np.all(loaded_state.discharge == 0.0)

    def test_very_small_values(self, tmp_path, sample_grid_metadata, all_states_enabled):
        """Test with very small values (numerical precision)."""
        nrows, ncols = sample_grid_metadata["shape"]

        state = SimulationState(
            wc=np.ones((nrows, ncols)) * 1e-12,
            wg=np.ones((nrows, ncols)) * 1e-12,
            wp=np.ones((nrows, ncols)) * 1e-12,
            ws=np.ones((nrows, ncols)) * 1e-12,
            discharge=np.ones(5) * 1e-12,
            lateral_inflow=np.ones(5) * 1e-12,
        )

        output_path = tmp_path / "state_tiny.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load and verify precision
        loaded_state, _, _ = load_state(output_path, network_size=5)

        np.testing.assert_allclose(loaded_state.wc, state.wc, rtol=1e-10)
        np.testing.assert_allclose(loaded_state.discharge, state.discharge, rtol=1e-10)

    def test_no_network_variables(self, tmp_path, sample_grid_metadata):
        """Test when discharge is disabled (no network variables)."""
        nrows, ncols = sample_grid_metadata["shape"]

        state = SimulationState(
            wc=np.random.rand(nrows, ncols) * 0.2,
            wg=np.random.rand(nrows, ncols) * 0.1,
            wp=None,
            ws=np.random.rand(nrows, ncols) * 0.01,
            discharge=np.random.rand(5) * 10.0,
            lateral_inflow=np.random.rand(5) * 2.0,
        )

        output_states = OutputStates(
            discharge=False,  # Disabled
            reservoir_states=False,
            soil_capillary=True,
            soil_gravitational=True,
            soil_plant=False,
            soil_surface=True,
            surface_temperature=False,
            ground_temperature=False,
            aquifer_head=False,
            et_prec=False,
        )

        output_path = tmp_path / "state_no_network.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=output_states,
        )

        # Load and verify
        ds = xr.open_dataset(output_path)

        # Should not have discharge or reach coordinate
        assert "discharge" not in ds
        assert "reach" not in ds.coords

        # Should have soil variables
        assert "Wc" in ds
        assert "Wg" in ds

        ds.close()

    def test_different_resolutions_x_y(self, tmp_path, all_states_enabled):
        """Test with different x and y resolutions."""
        grid_metadata = {
            "shape": (10, 15),
            "resolution": (100.0, 200.0),  # Different x and y resolution
            "xllcorner": 1000000.0,
            "yllcorner": 2000000.0,
            "crs": "EPSG:32632",
        }

        state = SimulationState(
            wc=np.random.rand(10, 15) * 0.2,
            wg=np.random.rand(10, 15) * 0.1,
            wp=None,
            ws=np.random.rand(10, 15) * 0.01,
            discharge=np.random.rand(5) * 10.0,
            lateral_inflow=np.random.rand(5) * 2.0,
        )

        output_path = tmp_path / "state_diff_res.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load and verify resolution
        loaded_state, _, loaded_metadata = load_state(output_path, network_size=5)

        assert loaded_metadata["resolution"] == (100.0, 200.0)

    def test_crs_metadata_preservation(self, tmp_path, sample_state, all_states_enabled):
        """Test that CRS information is preserved."""
        grid_metadata = {
            "shape": (10, 15),
            "resolution": (100.0, 100.0),
            "xllcorner": 1000000.0,
            "yllcorner": 2000000.0,
            "crs": "PROJCS['WGS_1984_UTM_Zone_32N',GEOGCS['GCS_WGS_1984',...]]",
        }

        output_path = tmp_path / "state_crs.nc"
        time = datetime(2020, 1, 1)

        save_state(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load and verify CRS
        ds = xr.open_dataset(output_path)

        # CRS should be in attributes
        assert "crs_wkt" in ds.crs.attrs
        assert "WGS_1984_UTM_Zone_32N" in ds.crs.attrs["crs_wkt"]

        ds.close()
