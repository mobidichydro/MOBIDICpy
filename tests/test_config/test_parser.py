"""Tests for configuration parser."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mobidic.config import MOBIDICConfig, load_config
from mobidic.config.parser import save_config


@pytest.fixture
def sample_config_path():
    """Path to sample configuration."""
    return Path(__file__).parent.parent.parent / "examples" / "sample_config.yaml"


@pytest.fixture
def minimal_config_dict():
    """Minimal valid configuration dictionary."""
    return {
        "basin": {
            "id": "TestBasin",
            "paramset_id": "TestScenario",
            "baricenter": {"lon": 10.0, "lat": 45.0},
        },
        "paths": {
            "meteodata": "test/meteodata.nc",
            "gisdata": "test/gisdata.nc",
            "states": "test/states/",
            "output": "test/outputs/",
        },
        "vector_files": {"river_network": {"shp": "test/network.shp", "id_field": "REACH_ID"}},
        "raster_files": {
            "dtm": "test/dtm.tif",
            "flow_dir": "test/flowdir.tif",
            "flow_acc": "test/flowacc.tif",
            "Wc0": "test/wc0.tif",
            "Wg0": "test/wg0.tif",
            "ks": "test/ks.tif",
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
            "resample": 1,
            "soil_scheme": "Bucket",
            "energy_balance": "None",
        },
        "output_states": {
            "discharge": True,
            "reservoir_states": True,
            "soil_capillary": True,
            "soil_gravitational": True,
            "surface_temperature": False,
            "ground_temperature": False,
            "aquifer_head": False,
            "et_prec": False,
        },
    }


class TestMOBIDICConfig:
    """Tests for MOBIDICConfig model."""

    def test_minimal_valid_config(self, minimal_config_dict):
        """Test that minimal valid configuration is accepted."""
        config = MOBIDICConfig(**minimal_config_dict)
        assert config.basin.id == "TestBasin"
        assert config.simulation.timestep == 900

    def test_invalid_flow_dir_type(self, minimal_config_dict):
        """Test that invalid flow direction type is rejected."""
        minimal_config_dict["raster_settings"]["flow_dir_type"] = "Invalid"
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_negative_timestep(self, minimal_config_dict):
        """Test that negative timestep is rejected."""
        minimal_config_dict["simulation"]["timestep"] = -100
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_invalid_saturation(self, minimal_config_dict):
        """Test that saturation outside [0, 1] is rejected."""
        minimal_config_dict["initial_conditions"] = {"Wcsat": 1.5}
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_invalid_albedo(self, minimal_config_dict):
        """Test that albedo outside [0, 1] is rejected."""
        minimal_config_dict["parameters"]["energy"]["Alb"] = 1.5
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_nbr_less_than_one(self, minimal_config_dict):
        """Test that NBr <= 1 is rejected."""
        minimal_config_dict["parameters"]["routing"]["NBr"] = 0.5
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_reach_selection_file_without_path(self, minimal_config_dict):
        """Test that reach_selection='file' without sel_file is rejected."""
        minimal_config_dict["output_report_settings"] = {"reach_selection": "file"}
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_reach_selection_list_without_list(self, minimal_config_dict):
        """Test that reach_selection='list' without sel_list is rejected."""
        minimal_config_dict["output_report_settings"] = {"reach_selection": "list"}
        with pytest.raises(ValidationError):
            MOBIDICConfig(**minimal_config_dict)

    def test_optional_fields_defaults(self, minimal_config_dict):
        """Test that optional fields have correct defaults."""
        config = MOBIDICConfig(**minimal_config_dict)
        assert config.initial_conditions.Ws == 0.0
        assert config.initial_conditions.Wcsat == 0.3
        assert config.parameters.multipliers.ks_factor == 1.0
        assert config.advanced.log_level == "INFO"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_sample_config(self, sample_config_path):
        """Test loading the sample example configuration."""
        if not sample_config_path.exists():
            pytest.skip(f"Sample config not found at {sample_config_path}")

        config = load_config(sample_config_path)
        assert config.basin.id == "Basin"
        assert config.basin.paramset_id == "Event_Scenario_1"
        assert config.basin.baricenter.lon == 11.0
        assert config.basin.baricenter.lat == 44.0
        assert config.simulation.timestep == 900
        assert config.raster_settings.flow_dir_type == "Grass"

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Test that loading invalid YAML raises YAMLError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("{ invalid: yaml: content:")

        with pytest.raises(yaml.YAMLError):
            load_config(invalid_yaml)

    def test_load_invalid_config(self, tmp_path):
        """Test that loading config with validation errors raises ValueError."""
        invalid_config = tmp_path / "invalid_config.yaml"
        invalid_config.write_text(
            """
basin:
  id: Test
  # Missing required fields
"""
        )

        with pytest.raises(ValueError):
            load_config(invalid_config)


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_and_reload(self, minimal_config_dict, tmp_path):
        """Test that saved config can be reloaded."""
        # Create config
        config = MOBIDICConfig(**minimal_config_dict)

        # Save to file
        output_path = tmp_path / "test_config.yaml"
        save_config(config, output_path)

        # Reload and verify
        reloaded_config = load_config(output_path)
        assert reloaded_config.basin.id == config.basin.id
        assert reloaded_config.simulation.timestep == config.simulation.timestep

    def test_roundtrip_sample_config(self, sample_config_path, tmp_path):
        """Test roundtrip save/load of sample configuration."""
        if not sample_config_path.exists():
            pytest.skip(f"Sample config not found at {sample_config_path}")

        # Load original
        original_config = load_config(sample_config_path)

        # Save and reload
        output_path = tmp_path / "sample_copy.yaml"
        save_config(original_config, output_path)
        reloaded_config = load_config(output_path)

        # Verify key fields match
        assert reloaded_config.basin.id == original_config.basin.id
        assert reloaded_config.basin.paramset_id == original_config.basin.paramset_id
        assert reloaded_config.simulation.timestep == original_config.simulation.timestep
        assert reloaded_config.parameters.soil.gamma == original_config.parameters.soil.gamma
