"""Tests for mobidic.io.meteo module."""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from mobidic.io.meteo import MeteoWriter, _get_variable_longname, _get_variable_units


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
def sample_grids(sample_grid_metadata):
    """Create sample meteorological grids."""
    nrows, ncols = sample_grid_metadata["shape"]
    return {
        "precipitation": np.random.rand(nrows, ncols) * 0.001,  # m/s
        "temperature": np.random.rand(nrows, ncols) * 20.0 + 10.0,  # degC
        "pet": np.random.rand(nrows, ncols) * 0.0001,  # m/s
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "meteo_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


class TestMeteoWriterInitialization:
    """Test MeteoWriter initialization."""

    def test_basic_initialization(self, sample_grid_metadata, temp_output_dir):
        """Test basic initialization."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation", "temperature"],
        )

        assert writer.output_path == output_path
        assert writer.variables == ["precipitation", "temperature"]
        assert writer.nrows == 10
        assert writer.ncols == 15
        assert len(writer.x) == 15
        assert len(writer.y) == 10
        assert len(writer.time_buffer) == 0
        assert len(writer.data_buffer) == 2

    def test_directory_creation(self, sample_grid_metadata, tmp_path):
        """Test automatic directory creation."""
        output_path = tmp_path / "nested" / "dir" / "meteo.nc"
        MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )
        assert output_path.parent.exists()

    def test_with_metadata(self, sample_grid_metadata, temp_output_dir):
        """Test initialization with additional metadata."""
        output_path = temp_output_dir / "meteo.nc"
        add_metadata = {"basin": "Arno", "run": "test"}
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
            add_metadata=add_metadata,
        )
        assert writer.add_metadata == add_metadata

    def test_coordinate_arrays(self, sample_grid_metadata, temp_output_dir):
        """Test coordinate array generation."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        # Check x coordinates
        expected_x = sample_grid_metadata["xllcorner"] + np.arange(15) * 100.0
        np.testing.assert_array_almost_equal(writer.x, expected_x)

        # Check y coordinates
        expected_y = sample_grid_metadata["yllcorner"] + np.arange(10) * 100.0
        np.testing.assert_array_almost_equal(writer.y, expected_y)

    def test_single_variable(self, sample_grid_metadata, temp_output_dir):
        """Test with single variable."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )
        assert len(writer.data_buffer) == 1
        assert "precipitation" in writer.data_buffer

    def test_multiple_variables(self, sample_grid_metadata, temp_output_dir):
        """Test with multiple variables."""
        output_path = temp_output_dir / "meteo.nc"
        variables = ["precipitation", "temperature", "pet", "humidity"]
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=variables,
        )
        assert len(writer.data_buffer) == 4
        for var in variables:
            assert var in writer.data_buffer
            assert writer.data_buffer[var] == []


class TestMeteoWriterAppend:
    """Test MeteoWriter append method."""

    def test_append_single_timestep(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test appending single timestep."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation", "temperature"],
        )

        time = datetime(2023, 1, 1, 0, 0)
        writer.append(
            time,
            precipitation=sample_grids["precipitation"],
            temperature=sample_grids["temperature"],
        )

        assert len(writer.time_buffer) == 1
        assert writer.time_buffer[0] == time
        assert len(writer.data_buffer["precipitation"]) == 1
        assert len(writer.data_buffer["temperature"]) == 1

    def test_append_multiple_timesteps(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test appending multiple timesteps."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        start_time = datetime(2023, 1, 1)
        for i in range(10):
            time = start_time + timedelta(hours=i)
            writer.append(time, precipitation=sample_grids["precipitation"])

        assert len(writer.time_buffer) == 10
        assert len(writer.data_buffer["precipitation"]) == 10

    def test_append_missing_variable(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test error when missing required variable."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation", "temperature"],
        )

        time = datetime(2023, 1, 1)
        with pytest.raises(ValueError, match="Missing required variable 'temperature'"):
            writer.append(time, precipitation=sample_grids["precipitation"])

    def test_append_wrong_shape(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test error when grid has wrong shape."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        time = datetime(2023, 1, 1)
        wrong_shape_grid = np.random.rand(5, 5)  # Wrong shape
        with pytest.raises(ValueError, match="incorrect shape"):
            writer.append(time, precipitation=wrong_shape_grid)

    def test_append_creates_copy(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test that append creates copy of data (not reference)."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        time = datetime(2023, 1, 1)
        original_grid = sample_grids["precipitation"].copy()
        writer.append(time, precipitation=original_grid)

        # Modify original
        original_grid[:] = 999.0

        # Check that buffered data is unchanged
        assert not np.allclose(writer.data_buffer["precipitation"][0], 999.0)


class TestMeteoWriterClose:
    """Test MeteoWriter close method."""

    def test_close_writes_netcdf(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test that close writes NetCDF file."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation", "temperature"],
        )

        # Add some data
        for i in range(5):
            time = datetime(2023, 1, 1) + timedelta(hours=i)
            writer.append(
                time,
                precipitation=sample_grids["precipitation"],
                temperature=sample_grids["temperature"],
            )

        # Close and write
        writer.close()

        # Check file exists
        assert output_path.exists()

        # Load and validate
        ds = xr.open_dataset(output_path)
        assert "precipitation" in ds.variables
        assert "temperature" in ds.variables
        assert "time" in ds.coords
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert "crs" in ds.variables
        assert len(ds.time) == 5
        ds.close()

    def test_close_with_empty_buffer(self, sample_grid_metadata, temp_output_dir):
        """Test close with empty buffer (no data appended)."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        # Close without appending any data
        writer.close()

        # File should not be created
        assert not output_path.exists()

    def test_close_clears_buffers(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test that close clears buffers."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        time = datetime(2023, 1, 1)
        writer.append(time, precipitation=sample_grids["precipitation"])

        writer.close()

        # Buffers should be cleared
        assert len(writer.time_buffer) == 0
        assert len(writer.data_buffer) == 0

    def test_unit_conversion_precipitation(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test unit conversion from m/s to mm/h for precipitation."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        # Input: m/s
        precip_m_per_s = np.ones((10, 15)) * 0.001  # 0.001 m/s
        time = datetime(2023, 1, 1)
        writer.append(time, precipitation=precip_m_per_s)

        writer.close()

        # Load and check conversion to mm/h
        ds = xr.open_dataset(output_path)
        # 0.001 m/s * 1000 mm/m * 3600 s/h = 3.6 mm/h
        expected_mm_per_h = 0.001 * 1000.0 * 3600.0
        np.testing.assert_array_almost_equal(ds["precipitation"].values[0], expected_mm_per_h)
        assert ds["precipitation"].attrs["units"] == "mm h-1"
        ds.close()

    def test_unit_conversion_pet(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test unit conversion from m/s to mm/h for PET."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["pet"],
        )

        # Input: m/s
        pet_m_per_s = np.ones((10, 15)) * 0.0001  # 0.0001 m/s
        time = datetime(2023, 1, 1)
        writer.append(time, pet=pet_m_per_s)

        writer.close()

        # Load and check conversion to mm/h
        ds = xr.open_dataset(output_path)
        expected_mm_per_h = 0.0001 * 1000.0 * 3600.0
        np.testing.assert_array_almost_equal(ds["pet"].values[0], expected_mm_per_h)
        assert ds["pet"].attrs["units"] == "mm h-1"
        ds.close()

    def test_no_unit_conversion_temperature(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test no unit conversion for temperature."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["temperature"],
        )

        # Input: degC
        temp_degc = np.ones((10, 15)) * 15.0
        time = datetime(2023, 1, 1)
        writer.append(time, temperature=temp_degc)

        writer.close()

        # Load and check no conversion
        ds = xr.open_dataset(output_path)
        np.testing.assert_array_almost_equal(ds["temperature"].values[0], 15.0)
        assert ds["temperature"].attrs["units"] == "degC"
        ds.close()

    def test_metadata_attributes(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test metadata attributes in NetCDF."""
        output_path = temp_output_dir / "meteo.nc"
        add_metadata = {"basin": "Arno", "simulation_id": "test_001"}
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
            add_metadata=add_metadata,
        )

        time = datetime(2023, 1, 1)
        writer.append(time, precipitation=sample_grids["precipitation"])
        writer.close()

        # Load and check metadata
        ds = xr.open_dataset(output_path)
        assert ds.attrs["title"] == "MOBIDIC meteorological forcing data"
        assert ds.attrs["Conventions"] == "CF-1.12"
        assert ds.attrs["basin"] == "Arno"
        assert ds.attrs["simulation_id"] == "test_001"
        assert "history" in ds.attrs
        assert "date_created" in ds.attrs
        ds.close()

    def test_crs_grid_mapping(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test CRS grid mapping in NetCDF."""
        output_path = temp_output_dir / "meteo.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        time = datetime(2023, 1, 1)
        writer.append(time, precipitation=sample_grids["precipitation"])
        writer.close()

        # Load and check CRS
        ds = xr.open_dataset(output_path)
        assert "crs" in ds.variables
        assert "grid_mapping_name" in ds["crs"].attrs
        assert ds["precipitation"].attrs["grid_mapping"] == "crs"
        ds.close()


class TestMeteoWriterContextManager:
    """Test MeteoWriter as context manager."""

    def test_context_manager_success(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test context manager with successful execution."""
        output_path = temp_output_dir / "meteo.nc"

        with MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        ) as writer:
            time = datetime(2023, 1, 1)
            writer.append(time, precipitation=sample_grids["precipitation"])

        # File should be written
        assert output_path.exists()

        # Validate contents
        ds = xr.open_dataset(output_path)
        assert "precipitation" in ds.variables
        ds.close()

    def test_context_manager_with_exception(self, sample_grid_metadata, temp_output_dir):
        """Test context manager with exception (should not write)."""
        output_path = temp_output_dir / "meteo.nc"

        with pytest.raises(ValueError):
            with MeteoWriter(
                output_path=output_path,
                grid_metadata=sample_grid_metadata,
                variables=["precipitation"],
            ):
                # Deliberately cause an error
                raise ValueError("Test error")

        # File should not be created due to exception
        assert not output_path.exists()

    def test_context_manager_multiple_timesteps(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test context manager with multiple timesteps."""
        output_path = temp_output_dir / "meteo.nc"

        with MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation", "temperature"],
        ) as writer:
            for i in range(24):
                time = datetime(2023, 1, 1) + timedelta(hours=i)
                writer.append(
                    time,
                    precipitation=sample_grids["precipitation"],
                    temperature=sample_grids["temperature"],
                )

        # Validate
        ds = xr.open_dataset(output_path)
        assert len(ds.time) == 24
        assert "precipitation" in ds.variables
        assert "temperature" in ds.variables
        ds.close()


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_variable_longname_known_variables(self):
        """Test _get_variable_longname for known variables."""
        assert _get_variable_longname("precipitation") == "precipitation rate"
        assert _get_variable_longname("pet") == "potential evapotranspiration rate"
        assert _get_variable_longname("temperature") == "air temperature"
        assert _get_variable_longname("temperature_min") == "minimum air temperature"
        assert _get_variable_longname("temperature_max") == "maximum air temperature"
        assert _get_variable_longname("humidity") == "relative humidity"
        assert _get_variable_longname("wind_speed") == "wind speed"
        assert _get_variable_longname("radiation") == "surface downwelling shortwave flux"

    def test_get_variable_longname_unknown_variable(self):
        """Test _get_variable_longname for unknown variable."""
        assert _get_variable_longname("unknown_var") == "unknown_var"
        assert _get_variable_longname("custom_variable") == "custom_variable"

    def test_get_variable_units_known_variables(self):
        """Test _get_variable_units for known variables."""
        assert _get_variable_units("precipitation") == "mm h-1"
        assert _get_variable_units("pet") == "mm h-1"
        assert _get_variable_units("temperature") == "degC"
        assert _get_variable_units("temperature_min") == "degC"
        assert _get_variable_units("temperature_max") == "degC"
        assert _get_variable_units("humidity") == "%"
        assert _get_variable_units("wind_speed") == "m s-1"
        assert _get_variable_units("radiation") == "W m-2"

    def test_get_variable_units_unknown_variable(self):
        """Test _get_variable_units for unknown variable."""
        assert _get_variable_units("unknown_var") == "unknown"
        assert _get_variable_units("custom_variable") == "unknown"


class TestMeteoWriterIntegration:
    """Integration tests for MeteoWriter."""

    def test_complete_workflow(self, sample_grid_metadata, temp_output_dir):
        """Test complete workflow with realistic data."""
        output_path = temp_output_dir / "meteo_complete.nc"
        nrows, ncols = sample_grid_metadata["shape"]

        # Simulate 48 hours of data
        with MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation", "temperature", "pet"],
            add_metadata={"basin": "TestBasin", "period": "2023-01-01 to 2023-01-02"},
        ) as writer:
            start_time = datetime(2023, 1, 1, 0, 0)
            for hour in range(48):
                time = start_time + timedelta(hours=hour)

                # Generate realistic data
                precip = np.random.rand(nrows, ncols) * 0.0005  # m/s
                temp = 15.0 + 5.0 * np.sin(hour / 24.0 * 2 * np.pi)  # Daily cycle
                temp_grid = np.full((nrows, ncols), temp)
                pet = np.random.rand(nrows, ncols) * 0.0001

                writer.append(time, precipitation=precip, temperature=temp_grid, pet=pet)

        # Validate output
        ds = xr.open_dataset(output_path)

        # Check dimensions
        assert len(ds.time) == 48
        assert ds.precipitation.shape == (48, nrows, ncols)
        assert ds.temperature.shape == (48, nrows, ncols)
        assert ds.pet.shape == (48, nrows, ncols)

        # Check units
        assert ds.precipitation.units == "mm h-1"
        assert ds.pet.units == "mm h-1"
        assert ds.temperature.units == "degC"

        # Check metadata
        assert ds.attrs["basin"] == "TestBasin"
        assert ds.attrs["Conventions"] == "CF-1.12"

        # Check coordinates
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert "crs" in ds.variables

        ds.close()

    def test_pathlib_path_input(self, sample_grid_metadata, sample_grids, temp_output_dir):
        """Test that Path objects work as input."""
        output_path = Path(temp_output_dir) / "meteo_pathlib.nc"
        writer = MeteoWriter(
            output_path=output_path,
            grid_metadata=sample_grid_metadata,
            variables=["precipitation"],
        )

        time = datetime(2023, 1, 1)
        writer.append(time, precipitation=sample_grids["precipitation"])
        writer.close()

        assert output_path.exists()
        assert isinstance(writer.output_path, Path)
