"""Tests for meteorological data preprocessing module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path
from mobidic.preprocessing.meteo_preprocessing import (
    MeteoData,
    MATMeteoReader,
    NetCDFMeteoReader,
    convert_mat_to_netcdf,
)


@pytest.fixture(scope="module")
def real_mat_file_path():
    """Path to real MATLAB meteo data file (for integration tests)."""
    return Path("examples/Arno/meteodata/meteodata.mat")


@pytest.fixture(scope="module")
def mock_mat_file(tmp_path_factory):
    """Create a small mock .mat file for fast unit testing.

    Contains minimal data: 2 stations, 10 timesteps, 2 variables.
    """
    try:
        import scipy.io
    except ImportError:
        pytest.skip("scipy not available")

    # Create temporary directory for this test session
    tmp_dir = tmp_path_factory.mktemp("meteo_test_data")
    mock_path = tmp_dir / "mock_meteo.mat"

    # Create minimal mock data structure matching MATLAB format
    # MATLAB datenum for 2023-11-01 00:00:00 is approximately 739184
    base_datenum = 739184.0
    time_steps = 10
    n_stations = 2

    # Create time array (MATLAB datenum format, column vector)
    # Use linspace to get exactly time_steps points
    time_array = np.linspace(base_datenum, base_datenum + (time_steps - 1) * 0.0104, time_steps).reshape(
        -1, 1
    )  # 15-min intervals

    # Define MATLAB struct dtype (matching the real .mat file structure)
    station_dtype = np.dtype(
        [
            ("code", "O"),
            ("est", "O"),
            ("nord", "O"),
            ("quota", "O"),
            ("name", "O"),
            ("time", "O"),
            ("dati", "O"),
        ]
    )

    # Create precipitation stations array
    # Use same station codes/locations for all variables (matching real MOBIDIC data)
    sp_stations = np.zeros((1, n_stations), dtype=station_dtype)
    for i in range(n_stations):
        sp_stations[0, i]["code"] = np.array([[1000 + i]])
        sp_stations[0, i]["est"] = np.array([[1600000.0 + i * 1000]])
        sp_stations[0, i]["nord"] = np.array([[4800000.0 + i * 1000]])
        sp_stations[0, i]["quota"] = np.array([[500.0 + i * 100]])
        sp_stations[0, i]["name"] = np.array([[f"Station_{i}"]], dtype="U20")
        sp_stations[0, i]["time"] = time_array.copy()
        sp_stations[0, i]["dati"] = np.random.rand(time_steps, 1) * 10  # Random precipitation 0-10 mm

    # Create temperature stations array (same stations as precipitation)
    s_ta_min_stations = np.zeros((1, n_stations), dtype=station_dtype)
    for i in range(n_stations):
        s_ta_min_stations[0, i]["code"] = np.array([[1000 + i]])  # Same codes as precipitation
        s_ta_min_stations[0, i]["est"] = np.array([[1600000.0 + i * 1000]])
        s_ta_min_stations[0, i]["nord"] = np.array([[4800000.0 + i * 1000]])
        s_ta_min_stations[0, i]["quota"] = np.array([[500.0 + i * 100]])
        s_ta_min_stations[0, i]["name"] = np.array([[f"Station_{i}"]], dtype="U20")
        s_ta_min_stations[0, i]["time"] = time_array.copy()
        s_ta_min_stations[0, i]["dati"] = np.random.rand(time_steps, 1) * 15 + 5  # Temperature 5-20°C

    # Create .mat file with mock data (use struct_as_record=True for compatibility)
    mat_data = {
        "sp": sp_stations,
        "s_ta_min": s_ta_min_stations,
    }

    scipy.io.savemat(str(mock_path), mat_data, oned_as="column")

    return mock_path


class TestMATMeteoReader:
    """Tests for MATLAB .mat meteo data reader (using mock data for speed)."""

    def test_reader_initialization(self, mock_mat_file):
        """Test MATMeteoReader initialization."""
        reader = MATMeteoReader(mock_mat_file)
        assert reader.file_path == mock_mat_file

    def test_reader_initialization_nonexistent_file(self):
        """Test that reader raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            MATMeteoReader("nonexistent.mat")

    def test_read_mat_file(self, mock_mat_file):
        """Test reading MATLAB meteo data file."""
        reader = MATMeteoReader(mock_mat_file)
        meteo_data = reader.read()

        assert isinstance(meteo_data, MeteoData)
        assert len(meteo_data.variables) > 0

        # Check expected variables from mock file
        assert "precipitation" in meteo_data.variables
        assert "temperature_min" in meteo_data.variables

    def test_station_structure(self, mock_mat_file):
        """Test that station data has correct structure."""
        meteo_data = MeteoData.from_mat(mock_mat_file)

        # Check each variable has stations
        for var_name, stations in meteo_data.stations.items():
            assert len(stations) > 0, f"No stations for {var_name}"

            # Check first station structure
            station = stations[0]
            assert "code" in station
            assert "x" in station
            assert "y" in station
            assert "elevation" in station
            assert "name" in station
            assert "time" in station
            assert "data" in station

            # Check data types
            assert isinstance(station["code"], (int, np.integer))
            assert isinstance(station["x"], (float, np.floating))
            assert isinstance(station["y"], (float, np.floating))
            assert isinstance(station["elevation"], (float, np.floating))
            assert isinstance(station["name"], str)
            assert isinstance(station["time"], pd.DatetimeIndex)
            assert isinstance(station["data"], np.ndarray)

    def test_precipitation_stations_count(self, mock_mat_file):
        """Test expected number of precipitation stations in mock file."""
        meteo_data = MeteoData.from_mat(mock_mat_file)
        # Mock file has 2 precipitation stations
        assert len(meteo_data.stations["precipitation"]) == 2

    def test_datetime_conversion(self, mock_mat_file):
        """Test that MATLAB datenum is correctly converted to pandas datetime."""
        meteo_data = MeteoData.from_mat(mock_mat_file)

        # Check that we have valid timestamps
        first_station = meteo_data.stations["precipitation"][0]
        assert len(first_station["time"]) == 10

        # Check that timestamps are in 2023
        assert all(first_station["time"].year == 2023)

        # Check that timestamps are monotonically increasing
        assert (first_station["time"][1:] > first_station["time"][:-1]).all()


class TestMATMeteoReaderIntegration:
    """Integration tests with real .mat file (fewer tests, validates correctness)."""

    def test_real_mat_file_exists(self, real_mat_file_path):
        """Test that example meteo data file exists."""
        assert real_mat_file_path.exists(), f"Test file not found: {real_mat_file_path}"

    def test_read_real_mat_file(self, real_mat_file_path):
        """Integration test: Read real MATLAB file and verify structure."""
        if not real_mat_file_path.exists():
            pytest.skip("Real test data file not found")

        meteo_data = MeteoData.from_mat(real_mat_file_path)

        # Check expected variables are present
        expected_vars = {
            "precipitation",
            "temperature_min",
            "temperature_max",
            "humidity",
            "wind_speed",
            "radiation",
        }
        assert set(meteo_data.variables) == expected_vars

        # Check we have the expected number of stations (223 for Arno basin)
        assert len(meteo_data.stations["precipitation"]) == 223

        # Check date range is correct for Arno example
        assert meteo_data.start_date.year == 2023
        assert meteo_data.start_date.month == 11
        assert meteo_data.end_date.year == 2023


class TestMeteoData:
    """Tests for MeteoData container class (using mock data)."""

    @pytest.fixture
    def meteo_data(self, mock_mat_file):
        """Load mock meteo data (fast)."""
        return MeteoData.from_mat(mock_mat_file)

    def test_meteo_data_repr(self, meteo_data):
        """Test MeteoData string representation."""
        repr_str = repr(meteo_data)
        assert "MeteoData" in repr_str
        assert "variables=" in repr_str

    def test_meteo_data_date_range(self, meteo_data):
        """Test that date range is extracted correctly."""
        assert meteo_data.start_date is not None
        assert meteo_data.end_date is not None
        assert isinstance(meteo_data.start_date, pd.Timestamp)
        assert isinstance(meteo_data.end_date, pd.Timestamp)
        assert meteo_data.start_date <= meteo_data.end_date

    def test_to_netcdf(self, meteo_data, tmp_path):
        """Test conversion to NetCDF format."""
        output_path = tmp_path / "test_meteo.nc"

        # Save to NetCDF
        meteo_data.to_netcdf(output_path)

        # Check file was created
        assert output_path.exists()

        # Load and check structure
        ds = xr.open_dataset(output_path)

        # Check dimensions
        assert "time" in ds.dims
        assert "station" in ds.dims

        # Check coordinates
        assert "station_code" in ds.coords
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert "elevation" in ds.coords
        assert "station_name" in ds.coords

        # Check data variables (mock file has precipitation and temperature_min)
        assert "precipitation" in ds.data_vars
        assert "temperature_min" in ds.data_vars
        assert ds["precipitation"].dims == ("time", "station")

        # Check global attributes
        assert "title" in ds.attrs
        assert "Conventions" in ds.attrs
        assert ds.attrs["Conventions"] == "CF-1.12"

        ds.close()

    def test_to_netcdf_with_metadata(self, meteo_data, tmp_path):
        """Test NetCDF export with additional metadata."""
        output_path = tmp_path / "test_meteo_metadata.nc"
        metadata = {"basin": "Test", "basin_id": "1234"}

        meteo_data.to_netcdf(output_path, add_metadata=metadata)

        ds = xr.open_dataset(output_path)
        assert ds.attrs["basin"] == "Test"
        assert ds.attrs["basin_id"] == "1234"
        ds.close()

    def test_to_netcdf_compression(self, meteo_data, tmp_path):
        """Test NetCDF compression settings."""
        output_path = tmp_path / "test_meteo_compressed.nc"

        # Save with high compression
        meteo_data.to_netcdf(output_path, compression_level=9)

        assert output_path.exists()
        ds = xr.open_dataset(output_path)
        ds.close()


class TestConvertMatToNetCDF:
    """Tests for convenience conversion function (using mock data)."""

    def test_convert_mat_to_netcdf(self, mock_mat_file, tmp_path):
        """Test direct conversion from MAT to NetCDF."""
        output_path = tmp_path / "converted_meteo.nc"

        convert_mat_to_netcdf(mock_mat_file, output_path)

        # Check output file exists
        assert output_path.exists()

        # Load and verify
        ds = xr.open_dataset(output_path)
        assert "precipitation" in ds.data_vars
        assert "temperature_min" in ds.data_vars
        ds.close()

    def test_convert_with_metadata(self, mock_mat_file, tmp_path):
        """Test conversion with additional metadata."""
        output_path = tmp_path / "converted_meteo_metadata.nc"
        metadata = {"test": "value"}

        convert_mat_to_netcdf(mock_mat_file, output_path, add_metadata=metadata)

        ds = xr.open_dataset(output_path)
        assert ds.attrs["test"] == "value"
        ds.close()


class TestMeteoDataValidation:
    """Tests for meteo data validation and error handling."""

    def test_empty_stations(self):
        """Test MeteoData with empty station lists."""
        empty_stations = {
            "precipitation": [],
            "temperature_min": [],
        }
        meteo_data = MeteoData(empty_stations)

        assert meteo_data.start_date is None
        assert meteo_data.end_date is None

    def test_to_netcdf_no_data_raises(self, tmp_path):
        """Test that saving empty data raises error."""
        empty_stations = {"precipitation": []}
        meteo_data = MeteoData(empty_stations)
        output_path = tmp_path / "empty_meteo.nc"

        with pytest.raises(ValueError, match="No valid meteorological data"):
            meteo_data.to_netcdf(output_path)

    def test_csv_reader_not_implemented(self):
        """Test that CSV reader raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            MeteoData.from_csv("test.csv", {})


class TestNetCDFMeteoReader:
    """Tests for NetCDF meteo data reader."""

    @pytest.fixture
    def sample_netcdf(self, mock_mat_file, tmp_path):
        """Create a sample NetCDF file from mock MAT data."""
        # Load mock data and save to NetCDF
        meteo_data = MeteoData.from_mat(mock_mat_file)
        nc_path = tmp_path / "test_meteo.nc"
        meteo_data.to_netcdf(nc_path)
        return nc_path

    def test_reader_initialization(self, sample_netcdf):
        """Test NetCDFMeteoReader initialization."""
        reader = NetCDFMeteoReader(sample_netcdf)
        assert reader.file_path == sample_netcdf

    def test_reader_initialization_nonexistent_file(self):
        """Test that reader raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            NetCDFMeteoReader("nonexistent.nc")

    def test_read_netcdf_file(self, sample_netcdf):
        """Test reading NetCDF meteo data file."""
        reader = NetCDFMeteoReader(sample_netcdf)
        meteo_data = reader.read()

        assert isinstance(meteo_data, MeteoData)
        assert len(meteo_data.variables) > 0

        # Check expected variables from mock file
        assert "precipitation" in meteo_data.variables
        assert "temperature_min" in meteo_data.variables

    def test_from_netcdf_classmethod(self, sample_netcdf):
        """Test MeteoData.from_netcdf() classmethod."""
        meteo_data = MeteoData.from_netcdf(sample_netcdf)

        assert isinstance(meteo_data, MeteoData)
        assert "precipitation" in meteo_data.variables
        assert "temperature_min" in meteo_data.variables

    def test_netcdf_station_structure(self, sample_netcdf):
        """Test that station data loaded from NetCDF has correct structure."""
        meteo_data = MeteoData.from_netcdf(sample_netcdf)

        # Check each variable has stations
        for var_name, stations in meteo_data.stations.items():
            assert len(stations) > 0, f"No stations for {var_name}"

            # Check first station structure
            station = stations[0]
            assert "code" in station
            assert "x" in station
            assert "y" in station
            assert "elevation" in station
            assert "name" in station
            assert "time" in station
            assert "data" in station

            # Check data types
            assert isinstance(station["code"], (int, np.integer))
            assert isinstance(station["x"], (float, np.floating))
            assert isinstance(station["y"], (float, np.floating))
            assert isinstance(station["elevation"], (float, np.floating))
            assert isinstance(station["name"], str)
            assert isinstance(station["time"], pd.DatetimeIndex)
            assert isinstance(station["data"], np.ndarray)

    def test_netcdf_round_trip(self, mock_mat_file, tmp_path):
        """Test round-trip: MAT -> MeteoData -> NetCDF -> MeteoData."""
        # Load from MAT
        meteo_original = MeteoData.from_mat(mock_mat_file)

        # Save to NetCDF
        nc_path = tmp_path / "roundtrip.nc"
        meteo_original.to_netcdf(nc_path)

        # Load from NetCDF
        meteo_reloaded = MeteoData.from_netcdf(nc_path)

        # Check variables match
        assert set(meteo_original.variables) == set(meteo_reloaded.variables)

        # Check number of stations matches
        for var in meteo_original.variables:
            assert len(meteo_original.stations[var]) == len(meteo_reloaded.stations[var])

        # Check date ranges match (approximately, due to NaN filtering)
        assert meteo_reloaded.start_date is not None
        assert meteo_reloaded.end_date is not None

    def test_netcdf_preserves_station_metadata(self, mock_mat_file, tmp_path):
        """Test that station metadata is preserved through NetCDF round-trip."""
        # Load from MAT
        meteo_original = MeteoData.from_mat(mock_mat_file)

        # Save to NetCDF and reload
        nc_path = tmp_path / "metadata_test.nc"
        meteo_original.to_netcdf(nc_path)
        meteo_reloaded = MeteoData.from_netcdf(nc_path)

        # Check first precipitation station metadata
        orig_station = meteo_original.stations["precipitation"][0]
        reload_station = meteo_reloaded.stations["precipitation"][0]

        assert orig_station["code"] == reload_station["code"]
        assert np.isclose(orig_station["x"], reload_station["x"])
        assert np.isclose(orig_station["y"], reload_station["y"])
        assert np.isclose(orig_station["elevation"], reload_station["elevation"])

    def test_netcdf_handles_missing_variables(self, tmp_path):
        """Test that reader handles NetCDF files with subset of variables."""
        # Create a minimal NetCDF with only one variable
        time = pd.date_range("2023-01-01", periods=5, freq="h")
        ds = xr.Dataset(
            {
                "precipitation": (
                    ["time", "station"],
                    np.random.rand(5, 1),
                    {"long_name": "precipitation", "units": "mm"},
                )
            },
            coords={
                "time": time,
                "station": [0],
                "station_code": (["station"], [1001]),
                "x": (["station"], [1600000.0]),
                "y": (["station"], [4800000.0]),
                "elevation": (["station"], [500.0]),
                "station_name": (["station"], ["Test"]),
            },
        )

        nc_path = tmp_path / "minimal.nc"
        ds.to_netcdf(nc_path)
        ds.close()

        # Load with NetCDFMeteoReader
        meteo_data = MeteoData.from_netcdf(nc_path)

        # Should only have precipitation
        assert meteo_data.variables == ["precipitation"]
        assert len(meteo_data.stations["precipitation"]) == 1
