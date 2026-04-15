"""Tests for mobidic.io.report module."""

from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import pytest
from mobidic.io.report import (
    save_discharge_report,
    load_discharge_report,
    save_lateral_inflow_report,
)


@pytest.fixture
def sample_network():
    """Create a sample river network GeoDataFrame for testing."""
    # Create 5 reaches with a simple topology
    # Reaches 0, 1, 2 are headwater reaches
    # Reach 3 is a junction (receives from 0 and 1)
    # Reach 4 is outlet (receives from 3 and 2)
    geometries = [
        LineString([(0, 0), (1, 0)]),  # reach 0
        LineString([(0, 1), (1, 1)]),  # reach 1
        LineString([(0, 2), (1, 2)]),  # reach 2
        LineString([(1, 0.5), (2, 0.5)]),  # reach 3 (junction)
        LineString([(2, 1), (3, 1)]),  # reach 4 (outlet)
    ]

    network = gpd.GeoDataFrame(
        {
            "mobidic_id": [0, 1, 2, 3, 4],
            "upstream_1": [np.nan, np.nan, np.nan, 0.0, 3.0],
            "upstream_2": [np.nan, np.nan, np.nan, 1.0, 2.0],
            "downstream": [3.0, 3.0, 4.0, 4.0, np.nan],
            "strahler_order": [1, 1, 1, 2, 3],
            "length_m": [100.0, 100.0, 100.0, 150.0, 200.0],
            "width_m": [5.0, 5.0, 5.0, 8.0, 12.0],
            "geometry": geometries,
        }
    )

    return network


@pytest.fixture
def sample_timeseries():
    """Create sample discharge time series data."""
    # 10 time steps, 5 reaches
    n_timesteps = 10
    n_reaches = 5

    # Create time stamps (hourly)
    start_time = datetime(2020, 1, 1, 0, 0)
    time_stamps = [start_time + timedelta(hours=i) for i in range(n_timesteps)]

    # Create discharge values with some variation
    discharge = np.zeros((n_timesteps, n_reaches))
    for i in range(n_reaches):
        # Increase discharge downstream
        base_flow = (i + 1) * 2.0
        discharge[:, i] = base_flow + np.sin(np.arange(n_timesteps) * 0.5)

    return discharge, time_stamps


class TestSaveDischargeReport:
    """Tests for save_discharge_report function."""

    def test_save_all_reaches(self, tmp_path, sample_network, sample_timeseries):
        """Test saving discharge report with all reaches."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge_all.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify structure
        df = pd.read_parquet(output_path)
        assert len(df) == len(time_stamps)
        assert len(df.columns) == len(sample_network)
        assert df.index.name == "time"

        # Check column names
        expected_columns = [f"reach_{i:04d}" for i in range(5)]
        assert list(df.columns) == expected_columns

        # Check data values
        np.testing.assert_array_almost_equal(df.values, discharge)

    def test_save_from_file(self, tmp_path, sample_network, sample_timeseries):
        """Test saving discharge report with reaches loaded from JSON file."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge_from_file.parquet"

        # Create a JSON file with reach IDs
        reach_file = tmp_path / "reach_ids.json"
        selected_reaches = [1, 3, 4]
        with open(reach_file, "w") as f:
            json.dump(selected_reaches, f)

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="file",
            reach_file=reach_file,
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify
        df = pd.read_parquet(output_path)
        assert len(df) == len(time_stamps)

        # Check correct reaches were selected
        assert len(df.columns) == len(selected_reaches)
        expected_columns = [f"reach_{i:04d}" for i in selected_reaches]
        assert list(df.columns) == expected_columns

        # Check values match
        expected_data = discharge[:, selected_reaches]
        np.testing.assert_array_almost_equal(df.values, expected_data)

    def test_save_selected_reaches_list(self, tmp_path, sample_network, sample_timeseries):
        """Test saving discharge report with selected reaches."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge_selected.parquet"

        selected_reaches = [0, 2, 4]

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="list",
            selected_reaches=selected_reaches,
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify
        df = pd.read_parquet(output_path)
        assert len(df) == len(time_stamps)
        assert len(df.columns) == len(selected_reaches)

        # Check column names
        expected_columns = [f"reach_{i:04d}" for i in selected_reaches]
        assert list(df.columns) == expected_columns

        # Check data values
        expected_data = discharge[:, selected_reaches]
        np.testing.assert_array_almost_equal(df.values, expected_data)

    def test_save_with_metadata(self, tmp_path, sample_network, sample_timeseries):
        """Test saving discharge report with additional metadata."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge_metadata.parquet"

        metadata = {
            "basin": "Test Basin",
            "model_version": "0.1.0",
            "notes": "Test run",
        }

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
            add_metadata=metadata,
        )

        # Check parquet file exists
        assert output_path.exists()

        # Check metadata JSON file exists
        metadata_path = output_path.with_suffix(".json")
        assert metadata_path.exists()

        # Load and verify metadata
        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        # Check standard metadata fields
        assert saved_metadata["reach_selection"] == "all"
        assert saved_metadata["n_reaches"] == 5
        assert saved_metadata["n_timesteps"] == 10
        assert saved_metadata["start_time"] == time_stamps[0].isoformat()
        assert saved_metadata["end_time"] == time_stamps[-1].isoformat()
        assert saved_metadata["reach_ids"] == [0, 1, 2, 3, 4]

        # Check custom metadata
        assert saved_metadata["basin"] == "Test Basin"
        assert saved_metadata["model_version"] == "0.1.0"
        assert saved_metadata["notes"] == "Test run"

    def test_save_creates_parent_directory(self, tmp_path, sample_network, sample_timeseries):
        """Test that save_discharge_report creates parent directories if needed."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "subdir" / "nested" / "discharge.parquet"

        # Parent directories don't exist yet
        assert not output_path.parent.exists()

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Check directory was created
        assert output_path.parent.exists()
        assert output_path.exists()

    def test_invalid_reach_selection_mode(self, tmp_path, sample_network, sample_timeseries):
        """Test error handling for invalid reach selection mode."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        with pytest.raises(ValueError, match="Invalid reach_selection"):
            save_discharge_report(
                discharge_timeseries=discharge,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="invalid_mode",
            )

    def test_missing_selected_reaches_parameter(self, tmp_path, sample_network, sample_timeseries):
        """Test error when reach_selection='list' but selected_reaches not provided."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        with pytest.raises(ValueError, match="selected_reaches must be provided"):
            save_discharge_report(
                discharge_timeseries=discharge,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="list",
                selected_reaches=None,
            )

    def test_invalid_reach_ids(self, tmp_path, sample_network, sample_timeseries):
        """Test error handling for invalid reach IDs."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        # Try to select reach IDs that don't exist (network has 5 reaches: 0-4)
        invalid_reaches = [0, 2, 10, 15]

        with pytest.raises(ValueError, match="Invalid reach IDs"):
            save_discharge_report(
                discharge_timeseries=discharge,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="list",
                selected_reaches=invalid_reaches,
            )

    def test_string_path_input(self, tmp_path, sample_network, sample_timeseries):
        """Test that function accepts string paths in addition to Path objects."""
        discharge, time_stamps = sample_timeseries
        output_path = str(tmp_path / "discharge.parquet")

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        assert Path(output_path).exists()

    def test_parquet_compression(self, tmp_path, sample_network, sample_timeseries):
        """Test that Parquet file uses compression."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Check that file was written with pyarrow engine
        df = pd.read_parquet(output_path, engine="pyarrow")
        assert df is not None

        # File should be relatively small due to compression
        file_size = output_path.stat().st_size
        assert file_size > 0
        assert file_size < 100000  # Should be much smaller than uncompressed

    def test_save_csv_format(self, tmp_path, sample_network, sample_timeseries):
        """Test saving discharge report in CSV format."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.csv"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
            output_format="csv",
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify structure
        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert len(df) == len(time_stamps)
        assert len(df.columns) == len(sample_network)

        # Check data values
        np.testing.assert_array_almost_equal(df.values, discharge)

    def test_save_parquet_format_explicit(self, tmp_path, sample_network, sample_timeseries):
        """Test saving discharge report with explicit Parquet format."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
            output_format="Parquet",
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify
        df = pd.read_parquet(output_path)
        assert len(df) == len(time_stamps)
        np.testing.assert_array_almost_equal(df.values, discharge)

    def test_invalid_output_format(self, tmp_path, sample_network, sample_timeseries):
        """Test error handling for invalid output format."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.txt"

        with pytest.raises(ValueError, match="Invalid output_format"):
            save_discharge_report(
                discharge_timeseries=discharge,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="all",
                output_format="txt",
            )


class TestLoadDischargeReport:
    """Tests for load_discharge_report function."""

    def test_load_basic(self, tmp_path, sample_network, sample_timeseries):
        """Test basic loading of discharge report."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        # Save first
        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Load
        df = load_discharge_report(output_path)

        # Verify structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(time_stamps)
        assert len(df.columns) == len(sample_network)
        assert df.index.name == "time"

        # Verify data
        np.testing.assert_array_almost_equal(df.values, discharge)

    def test_load_file_not_found(self, tmp_path):
        """Test error handling when file doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError, match="Report file not found"):
            load_discharge_report(nonexistent_path)

    def test_load_string_path(self, tmp_path, sample_network, sample_timeseries):
        """Test loading with string path."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        # Save
        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Load using string path
        df = load_discharge_report(str(output_path))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(time_stamps)

    def test_round_trip_preserves_data(self, tmp_path, sample_network, sample_timeseries):
        """Test that save/load round trip preserves data exactly."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        # Save
        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Load
        df = load_discharge_report(output_path)

        # Verify exact data match
        np.testing.assert_array_almost_equal(df.values, discharge, decimal=10)

        # Verify time stamps
        loaded_times = df.index.to_pydatetime()
        for original, loaded in zip(time_stamps, loaded_times):
            assert original == loaded

    def test_load_selected_reaches(self, tmp_path, sample_network, sample_timeseries):
        """Test loading report with only selected reaches."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"

        selected_reaches = [1, 3]

        # Save with selection
        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="list",
            selected_reaches=selected_reaches,
        )

        # Load
        df = load_discharge_report(output_path)

        # Verify only selected reaches are present
        assert len(df.columns) == len(selected_reaches)
        expected_columns = [f"reach_{i:04d}" for i in selected_reaches]
        assert list(df.columns) == expected_columns

        # Verify data
        expected_data = discharge[:, selected_reaches]
        np.testing.assert_array_almost_equal(df.values, expected_data)


class TestSaveLateralInflowReport:
    """Tests for save_lateral_inflow_report function."""

    def test_save_lateral_inflow_all_reaches(self, tmp_path, sample_network, sample_timeseries):
        """Test saving lateral inflow report with all reaches."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow.parquet"

        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify structure
        df = pd.read_parquet(output_path)
        assert len(df) == len(time_stamps)
        assert len(df.columns) == len(sample_network)
        assert df.index.name == "time"

        # Check data values
        np.testing.assert_array_almost_equal(df.values, lateral_inflow)

    def test_save_lateral_inflow_from_file(self, tmp_path, sample_network, sample_timeseries):
        """Test saving lateral inflow report with reaches loaded from JSON file."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow_from_file.parquet"

        # Create a JSON file with reach IDs
        reach_file = tmp_path / "reach_ids.json"
        selected_reaches = [0, 2]
        with open(reach_file, "w") as f:
            json.dump(selected_reaches, f)

        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="file",
            reach_file=reach_file,
        )

        # Load and verify
        df = pd.read_parquet(output_path)

        # Check correct reaches were selected
        assert len(df.columns) == len(selected_reaches)
        expected_data = lateral_inflow[:, selected_reaches]
        np.testing.assert_array_almost_equal(df.values, expected_data)

    def test_save_lateral_inflow_selected_list(self, tmp_path, sample_network, sample_timeseries):
        """Test saving lateral inflow report with selected reaches."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow_selected.parquet"

        selected_reaches = [0, 3]

        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="list",
            selected_reaches=selected_reaches,
        )

        # Load and verify
        df = pd.read_parquet(output_path)
        assert len(df.columns) == len(selected_reaches)

        expected_data = lateral_inflow[:, selected_reaches]
        np.testing.assert_array_almost_equal(df.values, expected_data)

    def test_save_lateral_inflow_creates_directory(self, tmp_path, sample_network, sample_timeseries):
        """Test that function creates parent directories."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "nested" / "dir" / "lateral_inflow.parquet"

        assert not output_path.parent.exists()

        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_save_lateral_inflow_invalid_selection(self, tmp_path, sample_network, sample_timeseries):
        """Test error handling for invalid reach selection."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow.parquet"

        with pytest.raises(ValueError, match="Invalid reach_selection"):
            save_lateral_inflow_report(
                lateral_inflow_timeseries=lateral_inflow,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="invalid",
            )

    def test_save_lateral_inflow_missing_selected_reaches(self, tmp_path, sample_network, sample_timeseries):
        """Test error when list mode but no selected_reaches provided."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow.parquet"

        with pytest.raises(ValueError, match="selected_reaches must be provided"):
            save_lateral_inflow_report(
                lateral_inflow_timeseries=lateral_inflow,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="list",
                selected_reaches=None,
            )

    def test_lateral_inflow_round_trip(self, tmp_path, sample_network, sample_timeseries):
        """Test that lateral inflow data survives save/load round trip."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow.parquet"

        # Save
        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Load using the discharge loading function (they're compatible)
        df = load_discharge_report(output_path)

        # Verify data match
        np.testing.assert_array_almost_equal(df.values, lateral_inflow, decimal=10)

    def test_lateral_inflow_string_path(self, tmp_path, sample_network, sample_timeseries):
        """Test that function accepts string paths."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = str(tmp_path / "lateral_inflow.parquet")

        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        assert Path(output_path).exists()

    def test_save_lateral_inflow_csv_format(self, tmp_path, sample_network, sample_timeseries):
        """Test saving lateral inflow report in CSV format."""
        lateral_inflow, time_stamps = sample_timeseries
        output_path = tmp_path / "lateral_inflow.csv"

        save_lateral_inflow_report(
            lateral_inflow_timeseries=lateral_inflow,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
            output_format="csv",
        )

        # Check file exists
        assert output_path.exists()

        # Load and verify
        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert len(df) == len(time_stamps)
        assert len(df.columns) == len(sample_network)
        np.testing.assert_array_almost_equal(df.values, lateral_inflow)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_reach_network(self, tmp_path):
        """Test with a network containing only one reach."""
        # Create single-reach network
        geometry = LineString([(0, 0), (1, 1)])
        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "upstream_1": [np.nan],
                "upstream_2": [np.nan],
                "downstream": [np.nan],
                "geometry": [geometry],
            }
        )

        # Create time series
        discharge = np.array([[1.5], [2.0], [2.5]])
        time_stamps = [datetime(2020, 1, 1, i) for i in range(3)]

        output_path = tmp_path / "single_reach.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=network,
            output_path=output_path,
            reach_selection="all",
        )

        # Verify
        df = load_discharge_report(output_path)
        assert len(df.columns) == 1
        assert df.columns[0] == "reach_0000"
        np.testing.assert_array_almost_equal(df.values.flatten(), discharge.flatten())

    def test_large_timeseries(self, tmp_path, sample_network):
        """Test with a large time series (performance check)."""
        # Create large time series (1 year, hourly data)
        n_timesteps = 8760  # 365 days * 24 hours
        n_reaches = len(sample_network)

        discharge = np.random.rand(n_timesteps, n_reaches) * 10
        start_time = datetime(2020, 1, 1)
        time_stamps = [start_time + timedelta(hours=i) for i in range(n_timesteps)]

        output_path = tmp_path / "large_discharge.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        # Verify file exists and can be loaded
        assert output_path.exists()
        df = load_discharge_report(output_path)
        assert len(df) == n_timesteps

        # Compression should keep file size reasonable
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        assert file_size_mb < 5.0  # Should be less than 5 MB with compression

    def test_zero_discharge_values(self, tmp_path, sample_network):
        """Test with all zero discharge values."""
        n_timesteps = 5
        n_reaches = len(sample_network)

        discharge = np.zeros((n_timesteps, n_reaches))
        time_stamps = [datetime(2020, 1, 1, i) for i in range(n_timesteps)]

        output_path = tmp_path / "zero_discharge.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        df = load_discharge_report(output_path)
        assert np.all(df.values == 0.0)

    def test_very_small_discharge_values(self, tmp_path, sample_network):
        """Test with very small discharge values (numerical precision)."""
        n_timesteps = 5
        n_reaches = len(sample_network)

        discharge = np.ones((n_timesteps, n_reaches)) * 1e-10
        time_stamps = [datetime(2020, 1, 1, i) for i in range(n_timesteps)]

        output_path = tmp_path / "tiny_discharge.parquet"

        save_discharge_report(
            discharge_timeseries=discharge,
            time_stamps=time_stamps,
            network=sample_network,
            output_path=output_path,
            reach_selection="all",
        )

        df = load_discharge_report(output_path)
        np.testing.assert_allclose(df.values, discharge, rtol=1e-15)

    def test_file_selection_missing_file(self, tmp_path, sample_network, sample_timeseries):
        """Test error handling when reach_file doesn't exist."""
        discharge, time_stamps = sample_timeseries
        output_path = tmp_path / "discharge.parquet"
        reach_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Reach file not found"):
            save_discharge_report(
                discharge_timeseries=discharge,
                time_stamps=time_stamps,
                network=sample_network,
                output_path=output_path,
                reach_selection="file",
                reach_file=reach_file,
            )
