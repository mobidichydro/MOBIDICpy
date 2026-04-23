"""Tests for mobidic.io.state module."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import xarray as xr
import pytest
from mobidic.io.state import load_state, StateWriter
from mobidic.core.reservoir import ReservoirState
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


def save_state_helper(state, output_path, time, grid_metadata, network_size, output_states, add_metadata=None):
    """Helper function to save a single state using StateWriter (for tests)."""
    writer = StateWriter(
        output_path=output_path,
        grid_metadata=grid_metadata,
        network_size=network_size,
        output_states=output_states,
        flushing=-1,
        add_metadata=add_metadata,
    )
    writer.append_state(state, time)
    writer.close()


class TestLoadState:
    """Tests for load_state function."""

    def test_load_basic(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test basic loading of state."""
        output_path = tmp_path / "state.nc"
        time = datetime(2020, 6, 15, 12, 0)

        # Save first
        save_state_helper(
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
        save_state_helper(
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
        """Test loading when Wc is missing (should initialize with zeros)."""
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

        # Wc should be all zeros
        np.testing.assert_array_equal(loaded_state.wc, np.zeros((10, 15)))

        # Wg should be loaded correctly
        assert not np.all(np.isnan(loaded_state.wg))

    def test_load_missing_wg_variable(self, tmp_path, sample_grid_metadata):
        """Test loading when Wg is missing (should initialize with zeros)."""
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

        # Wg should be all zeros
        np.testing.assert_array_equal(loaded_state.wg, np.zeros((10, 15)))

        # Wc should be loaded correctly
        assert not np.all(np.isnan(loaded_state.wc))

    def test_load_missing_ws_variable(self, tmp_path, sample_grid_metadata):
        """Test loading when Ws is missing (should initialize with zeros)."""
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

        # Ws should be all zeros
        np.testing.assert_array_equal(loaded_state.ws, np.zeros((10, 15)))

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
        """Test loading when Wp is missing (should be zeros)."""
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

        # Wp should be zeros
        np.testing.assert_array_equal(loaded_state.wp, np.zeros((10, 15)))

    def test_load_network_size_mismatch(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """Test warning when network size doesn't match."""
        output_path = tmp_path / "state.nc"
        time = datetime(2020, 6, 15, 12, 0)

        # Save with 5 reaches
        save_state_helper(
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
        save_state_helper(
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
        save_state_helper(
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

        # Should initialize missing variables with zeros
        np.testing.assert_array_equal(loaded_state.wp, np.zeros((10, 15)))
        np.testing.assert_array_equal(loaded_state.ws, np.zeros((10, 15)))
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

        save_state_helper(
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

        save_state_helper(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=grid_metadata,
            network_size=50,
            output_states=all_states_enabled,
        )

        # Check file exists (may be chunked)
        chunk_path = output_path.parent / f"{output_path.stem}_001{output_path.suffix}"
        assert chunk_path.exists() or output_path.exists()
        actual_path = chunk_path if chunk_path.exists() else output_path
        file_size_mb = actual_path.stat().st_size / (1024 * 1024)
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

        save_state_helper(
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

        save_state_helper(
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

        save_state_helper(
            state=state,
            output_path=output_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=output_states,
        )

        # Load and verify (use chunked filename if exists)
        chunk_path = output_path.parent / f"{output_path.stem}_001{output_path.suffix}"
        actual_path = chunk_path if chunk_path.exists() else output_path
        ds = xr.open_dataset(actual_path)

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

        save_state_helper(
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

        save_state_helper(
            state=sample_state,
            output_path=output_path,
            time=time,
            grid_metadata=grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )

        # Load and verify CRS (use chunked filename if exists)
        chunk_path = output_path.parent / f"{output_path.stem}_001{output_path.suffix}"
        actual_path = chunk_path if chunk_path.exists() else output_path
        ds = xr.open_dataset(actual_path)

        # CRS should be in attributes
        assert "crs_wkt" in ds.crs.attrs
        assert "WGS_1984_UTM_Zone_32N" in ds.crs.attrs["crs_wkt"]

        ds.close()


class TestChunking:
    """Test file chunking functionality."""

    def test_chunking_creates_multiple_files(self, tmp_path, all_states_enabled):
        """Test that chunking creates multiple files when max_file_size is exceeded."""
        # Create a large grid to force chunking with small max_file_size
        grid_metadata = {
            "shape": (200, 300),  # Large grid
            "resolution": (50.0, 50.0),
            "xllcorner": 500000.0,
            "yllcorner": 4000000.0,
            "crs": "EPSG:32632",
        }

        output_path = tmp_path / "chunked_states.nc"
        base_time = datetime(2020, 1, 1)

        # Use small max_file_size to force chunking (0.1 MB)
        with StateWriter(
            output_path=output_path,
            grid_metadata=grid_metadata,
            network_size=10,
            output_states=all_states_enabled,
            flushing=1,  # Flush after each state
            max_file_size=0.1,  # Very small to force chunking
        ) as writer:
            # Write multiple states
            for i in range(5):
                state = SimulationState(
                    wc=np.random.rand(200, 300) * 0.2,
                    wg=np.random.rand(200, 300) * 0.1,
                    wp=np.random.rand(200, 300) * 0.005,
                    ws=np.random.rand(200, 300) * 0.01,
                    discharge=np.random.rand(10) * 10.0,
                    lateral_inflow=np.random.rand(10) * 2.0,
                )
                current_time = base_time + timedelta(hours=i)
                writer.append_state(state, current_time)

        # Check that multiple chunk files were created
        chunk_001 = tmp_path / "chunked_states_001.nc"
        chunk_002 = tmp_path / "chunked_states_002.nc"

        assert chunk_001.exists(), "First chunk file should exist"
        assert chunk_002.exists(), "Second chunk file should exist (file size exceeded limit)"

        # Verify we can load from the first chunk
        state, time, metadata = load_state(output_path, network_size=10, time_index=0)
        assert state.wc.shape == (200, 300)
        assert len(state.discharge) == 10

    def test_no_chunking_below_size_limit(self, tmp_path, all_states_enabled):
        """Test that no chunking occurs when files stay below max_file_size."""
        grid_metadata = {
            "shape": (10, 15),  # Small grid
            "resolution": (100.0, 100.0),
            "xllcorner": 1000000.0,
            "yllcorner": 2000000.0,
            "crs": "EPSG:32632",
        }

        output_path = tmp_path / "single_chunk.nc"
        base_time = datetime(2020, 1, 1)

        # Use large max_file_size - file will stay below this
        with StateWriter(
            output_path=output_path,
            grid_metadata=grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
            flushing=1,
            max_file_size=500.0,  # Large limit
        ) as writer:
            # Write a few states
            for i in range(3):
                state = SimulationState(
                    wc=np.random.rand(10, 15) * 0.2,
                    wg=np.random.rand(10, 15) * 0.1,
                    wp=np.random.rand(10, 15) * 0.005,
                    ws=np.random.rand(10, 15) * 0.01,
                    discharge=np.random.rand(5) * 10.0,
                    lateral_inflow=np.random.rand(5) * 2.0,
                )
                current_time = base_time + timedelta(hours=i)
                writer.append_state(state, current_time)

        # Check that only one chunk file was created
        chunk_001 = tmp_path / "single_chunk_001.nc"
        chunk_002 = tmp_path / "single_chunk_002.nc"

        assert chunk_001.exists(), "First chunk file should exist"
        assert not chunk_002.exists(), "Second chunk file should not exist (size limit not exceeded)"


# ---------------------------------------------------------------------------
# Helpers for config-driven loading and advanced StateWriter tests
# ---------------------------------------------------------------------------


def _make_mock_config(*, gw_model="None", energy_balance="None", wg_wc_tr=-1.0):
    """Build a minimal config-like object that exposes only what load_state needs."""
    return SimpleNamespace(
        parameters=SimpleNamespace(
            multipliers=SimpleNamespace(
                Wc_factor=1.0,
                Wg_factor=1.0,
                Wg_Wc_tr=wg_wc_tr,
            ),
            groundwater=SimpleNamespace(model=gw_model),
            energy=SimpleNamespace(Tconst=290.0),
        ),
        initial_conditions=SimpleNamespace(
            Wcsat=0.3,
            Wgsat=0.2,
            Ws=0.0,
            groundwater_head=5.0,
        ),
        simulation=SimpleNamespace(energy_balance=energy_balance),
    )


def _make_mock_gisdata(nrows, ncols, *, with_nan_border=True):
    """Build a minimal GISData-like object with the grids that load_state touches."""
    flow_acc = np.ones((nrows, ncols), dtype=float)
    if with_nan_border:
        flow_acc[0, :] = np.nan
        flow_acc[-1, :] = np.nan
    return SimpleNamespace(
        grids={
            "Wc0": np.full((nrows, ncols), 0.2),
            "Wg0": np.full((nrows, ncols), 0.1),
            "flow_acc": flow_acc,
        }
    )


def _write_minimal_state_nc(path, *, extra_vars=None, nrows=10, ncols=15, network_size=5):
    """Write a tiny state NetCDF with configurable included variables."""
    coords = {
        "x": 1e6 + np.arange(ncols) * 100.0,
        "y": 2e6 + np.arange(nrows) * 100.0,
        "time": datetime(2020, 6, 15),
    }
    data_vars = {"crs": ([], 0)}
    if extra_vars:
        for name, arr in extra_vars.items():
            if arr.ndim == 1:
                data_vars[name] = (["reach"], arr)
                coords["reach"] = np.arange(len(arr))
            else:
                data_vars[name] = (["y", "x"], arr)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(path)
    ds.close()


class TestLoadStateConfigDrivenInit:
    """load_state must use config + gisdata to initialize missing state variables."""

    def test_missing_wc_initialized_from_config(self, tmp_path, sample_grid_metadata):
        """Missing Wc is rebuilt from Wc0 * Wc_factor * Wcsat with NaN outside the domain."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_wc.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={"Wg": np.random.rand(nrows, ncols) * 0.1},
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config()
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)

        assert state.wc.shape == (nrows, ncols)
        # Interior cells: Wc0 (0.2) * Wcsat (0.3) = 0.06
        assert state.wc[nrows // 2, ncols // 2] == pytest.approx(0.06)
        # Border rows have NaN flow_acc => NaN wc
        assert np.all(np.isnan(state.wc[0, :]))
        assert np.all(np.isnan(state.wc[-1, :]))

    def test_missing_wc_with_wg_wc_transition(self, tmp_path, sample_grid_metadata):
        """Wg_Wc_tr >= 0 triggers the transition branch in the Wc initialization path."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_wc_tr.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={"Wg": np.random.rand(nrows, ncols) * 0.1},
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config(wg_wc_tr=0.5)
        gisdata = _make_mock_gisdata(nrows, ncols, with_nan_border=False)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)

        # wg0' = min(0.5 * 0.1, 0.3) = 0.05; wc0' = 0.3 - 0.05 = 0.25; wc = 0.25 * 0.3
        np.testing.assert_allclose(state.wc, 0.25 * 0.3)

    def test_missing_wg_initialized_from_config(self, tmp_path, sample_grid_metadata):
        """Missing Wg is rebuilt from Wg0 * Wg_factor * Wgsat with NaN outside the domain."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_wg.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={"Wc": np.random.rand(nrows, ncols) * 0.2},
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config()
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)

        assert state.wg[nrows // 2, ncols // 2] == pytest.approx(0.1 * 0.2)
        assert np.all(np.isnan(state.wg[0, :]))

    def test_missing_wg_with_wg_wc_transition(self, tmp_path, sample_grid_metadata):
        """Wg_Wc_tr >= 0 triggers the transition branch in the Wg initialization path."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_wg_tr.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={"Wc": np.random.rand(nrows, ncols) * 0.2},
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config(wg_wc_tr=0.5)
        gisdata = _make_mock_gisdata(nrows, ncols, with_nan_border=False)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)

        np.testing.assert_allclose(state.wg, 0.05 * 0.2)

    def test_missing_ws_initialized_from_config(self, tmp_path, sample_grid_metadata):
        """Missing Ws defaults to config.initial_conditions.Ws inside the domain, NaN outside."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_ws.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={
                "Wc": np.random.rand(nrows, ncols) * 0.2,
                "Wg": np.random.rand(nrows, ncols) * 0.1,
            },
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config()
        config.initial_conditions.Ws = 0.01
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)
        assert state.ws[nrows // 2, ncols // 2] == pytest.approx(0.01)
        assert np.all(np.isnan(state.ws[0, :]))

    def test_missing_wp_initialized_from_config(self, tmp_path, sample_grid_metadata):
        """Missing Wp, with config+gisdata, is zeros inside domain and NaN outside."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_wp.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={
                "Wc": np.random.rand(nrows, ncols) * 0.2,
                "Wg": np.random.rand(nrows, ncols) * 0.1,
            },
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config()
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)
        assert state.wp[nrows // 2, ncols // 2] == 0.0
        assert np.all(np.isnan(state.wp[0, :]))

    def test_groundwater_head_initialized_when_model_linear(self, tmp_path, sample_grid_metadata):
        """When groundwater is Linear and h is missing, h is filled from config."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_h.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={
                "Wc": np.random.rand(nrows, ncols) * 0.2,
                "Wg": np.random.rand(nrows, ncols) * 0.1,
            },
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config(gw_model="Linear")
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)
        assert state.h is not None
        assert state.h[nrows // 2, ncols // 2] == pytest.approx(5.0)
        assert np.all(np.isnan(state.h[0, :]))

    def test_groundwater_head_stays_none_when_model_is_none(self, tmp_path, sample_grid_metadata):
        """When groundwater model is 'None', missing h remains None."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "no_h.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={"Wc": np.zeros((nrows, ncols)), "Wg": np.zeros((nrows, ncols))},
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config(gw_model="None")
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)
        assert state.h is None

    def test_energy_balance_ts_td_initialized(self, tmp_path, sample_grid_metadata):
        """When energy balance is active and Ts/Td are missing, both are filled with Tconst."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "missing_ts_td.nc"
        _write_minimal_state_nc(
            path,
            extra_vars={"Wc": np.zeros((nrows, ncols)), "Wg": np.zeros((nrows, ncols))},
            nrows=nrows,
            ncols=ncols,
        )
        config = _make_mock_config(energy_balance="1L")
        gisdata = _make_mock_gisdata(nrows, ncols)

        state, _, _ = load_state(path, network_size=5, config=config, gisdata=gisdata)
        assert state.ts is not None
        assert state.td is not None
        assert state.ts[nrows // 2, ncols // 2] == pytest.approx(290.0)
        assert state.td[nrows // 2, ncols // 2] == pytest.approx(290.0)
        assert np.all(np.isnan(state.ts[0, :]))
        assert np.all(np.isnan(state.td[0, :]))

    def test_existing_h_ts_td_loaded_from_file(self, tmp_path, sample_grid_metadata):
        """If h, Ts, Td exist in the file, they are loaded verbatim (regardless of config)."""
        nrows, ncols = sample_grid_metadata["shape"]
        path = tmp_path / "with_h_ts_td.nc"
        h_in = np.full((nrows, ncols), 2.5)
        ts_in = np.full((nrows, ncols), 295.0)
        td_in = np.full((nrows, ncols), 294.0)
        _write_minimal_state_nc(
            path,
            extra_vars={
                "Wc": np.zeros((nrows, ncols)),
                "Wg": np.zeros((nrows, ncols)),
                "h": h_in,
                "Ts": ts_in,
                "Td": td_in,
            },
            nrows=nrows,
            ncols=ncols,
        )

        state, _, _ = load_state(path, network_size=5)
        np.testing.assert_array_equal(state.h, h_in)
        np.testing.assert_array_equal(state.ts, ts_in)
        np.testing.assert_array_equal(state.td, td_in)


class TestLoadStateChunkFallback:
    """load_state should transparently fall back to chunk files when the base path is absent."""

    def test_auto_falls_back_to_first_chunk(self, tmp_path, sample_state, sample_grid_metadata, all_states_enabled):
        """A base path that doesn't exist but has a matching _001 chunk is loaded."""
        base_path = tmp_path / "run_states.nc"
        time = datetime(2021, 1, 1)
        save_state_helper(
            state=sample_state,
            output_path=base_path,
            time=time,
            grid_metadata=sample_grid_metadata,
            network_size=5,
            output_states=all_states_enabled,
        )
        # StateWriter writes to run_states_001.nc, not run_states.nc
        assert not base_path.exists()
        assert (tmp_path / "run_states_001.nc").exists()

        # Loading via the base path must succeed by falling back to the chunk.
        loaded_state, _, _ = load_state(base_path, network_size=5)
        assert loaded_state.wc.shape == sample_grid_metadata["shape"]

    def test_error_when_neither_base_nor_chunk_exists(self, tmp_path):
        """Missing base file AND missing _001 chunk raises FileNotFoundError."""
        missing = tmp_path / "nothing_here.nc"
        with pytest.raises(FileNotFoundError):
            load_state(missing, network_size=3)


class TestStateWriterAdvanced:
    """Additional StateWriter coverage for optional variables and edge cases."""

    def _grid_metadata(self, nrows=6, ncols=8):
        return {
            "shape": (nrows, ncols),
            "resolution": (100.0, 100.0),
            "xllcorner": 0.0,
            "yllcorner": 0.0,
            "crs": "EPSG:32632",
        }

    def _output_states(self, **overrides):
        defaults = dict(
            discharge=True,
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
        defaults.update(overrides)
        return OutputStates(**defaults)

    def test_empty_buffer_flush_is_noop(self, tmp_path):
        """Calling flush() with no buffered states is a no-op and does not create files."""
        md = self._grid_metadata()
        out = tmp_path / "empty_flush.nc"
        writer = StateWriter(
            output_path=out,
            grid_metadata=md,
            network_size=2,
            output_states=self._output_states(),
            flushing=-1,
        )
        writer.flush()  # no-op, should not raise
        writer.close()
        # With no states ever appended, no chunk file should have been created.
        assert not (tmp_path / "empty_flush_001.nc").exists()

    def test_existing_chunks_are_removed_on_init(self, tmp_path):
        """StateWriter removes pre-existing chunk files matching its pattern on init."""
        md = self._grid_metadata()
        stale = tmp_path / "run_001.nc"
        stale.write_bytes(b"stale")
        assert stale.exists()

        _ = StateWriter(
            output_path=tmp_path / "run.nc",
            grid_metadata=md,
            network_size=1,
            output_states=self._output_states(),
            flushing=-1,
        )
        assert not stale.exists()

    def test_writes_all_optional_grids_h_ts_td_et(self, tmp_path):
        """Enabling aquifer_head/surface_temperature/ground_temperature/evapotranspiration persists them."""
        md = self._grid_metadata()
        out = tmp_path / "optional_grids.nc"
        nrows, ncols = md["shape"]
        state = SimulationState(
            wc=np.full((nrows, ncols), 0.1),
            wg=np.full((nrows, ncols), 0.05),
            wp=None,
            ws=np.full((nrows, ncols), 0.01),
            discharge=np.array([1.0, 2.0]),
            lateral_inflow=np.array([0.1, 0.2]),
            h=np.full((nrows, ncols), 2.0),
            ts=np.full((nrows, ncols), 293.0),
            td=np.full((nrows, ncols), 292.0),
            et=np.full((nrows, ncols), 1e-7),
        )
        with StateWriter(
            output_path=out,
            grid_metadata=md,
            network_size=2,
            output_states=self._output_states(
                aquifer_head=True,
                surface_temperature=True,
                ground_temperature=True,
                evapotranspiration=True,
            ),
            flushing=-1,
        ) as writer:
            writer.append_state(state, datetime(2020, 1, 1))

        chunk = tmp_path / "optional_grids_001.nc"
        assert chunk.exists()
        with xr.open_dataset(chunk) as ds:
            for name, attr_marker in [
                ("h", "Groundwater Head"),
                ("Ts", "Surface Temperature"),
                ("Td", "Deep Soil Temperature"),
                ("ET", "Actual Evapotranspiration Rate"),
            ]:
                assert name in ds.data_vars
                assert ds[name].attrs["long_name"] == attr_marker

    def test_writes_reservoir_states(self, tmp_path):
        """Reservoir states are serialized into per-reservoir time series arrays."""
        md = self._grid_metadata()
        out = tmp_path / "with_reservoirs.nc"
        nrows, ncols = md["shape"]
        reservoirs = [
            ReservoirState(volume=1e6, stage=10.0, inflow=5.0, outflow=4.5),
            ReservoirState(volume=5e5, stage=7.5, inflow=2.0, outflow=1.8),
        ]
        state = SimulationState(
            wc=np.zeros((nrows, ncols)),
            wg=np.zeros((nrows, ncols)),
            wp=None,
            ws=np.zeros((nrows, ncols)),
            discharge=np.zeros(2),
            lateral_inflow=np.zeros(2),
            reservoir_states=reservoirs,
        )
        with StateWriter(
            output_path=out,
            grid_metadata=md,
            network_size=2,
            output_states=self._output_states(reservoir_states=True),
            flushing=-1,
            reservoir_size=len(reservoirs),
        ) as writer:
            writer.append_state(state, datetime(2020, 1, 1))
            writer.append_state(state, datetime(2020, 1, 1, 1))

        chunk = tmp_path / "with_reservoirs_001.nc"
        with xr.open_dataset(chunk) as ds:
            assert "reservoir" in ds.coords
            assert ds.sizes["reservoir"] == 2
            for name in (
                "reservoir_volume",
                "reservoir_stage",
                "reservoir_inflow",
                "reservoir_outflow",
            ):
                assert name in ds.data_vars
            np.testing.assert_allclose(ds["reservoir_volume"].values[0], [1e6, 5e5])
            np.testing.assert_allclose(ds["reservoir_outflow"].values[0], [4.5, 1.8])

    def test_multi_flush_appends_time(self, tmp_path):
        """Two separate flushes append to the same file, growing the time dimension."""
        md = self._grid_metadata()
        out = tmp_path / "multi_flush.nc"
        nrows, ncols = md["shape"]
        with StateWriter(
            output_path=out,
            grid_metadata=md,
            network_size=2,
            output_states=self._output_states(),
            flushing=1,  # flush after every append
        ) as writer:
            for i in range(3):
                state = SimulationState(
                    wc=np.full((nrows, ncols), 0.1 * (i + 1)),
                    wg=np.full((nrows, ncols), 0.05),
                    wp=None,
                    ws=np.full((nrows, ncols), 0.01),
                    discharge=np.array([float(i), float(i)]),
                    lateral_inflow=np.zeros(2),
                )
                writer.append_state(state, datetime(2020, 1, 1) + timedelta(hours=i))

        chunk = tmp_path / "multi_flush_001.nc"
        with xr.open_dataset(chunk) as ds:
            assert ds.sizes["time"] == 3
            # First timestep Wc equals 0.1, last equals 0.3
            np.testing.assert_allclose(ds["Wc"].isel(time=0).values, 0.1)
            np.testing.assert_allclose(ds["Wc"].isel(time=-1).values, 0.3)
