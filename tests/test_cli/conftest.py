"""Shared fixtures for CLI tests."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def minimal_config_dict():
    """Minimal valid MOBIDIC configuration dictionary (station forcing)."""
    return {
        "basin": {
            "id": "TestBasin",
            "paramset_id": "TestScenario",
            "baricenter": {"lon": 10.0, "lat": 45.0},
        },
        "paths": {
            "meteodata": "test/meteodata.nc",
            "gisdata": "test/gisdata.nc",
            "network": "test/network.parquet",
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
            "timestep": 900,
            "decimation": 1,
            "soil_scheme": "Bucket",
            "energy_balance": "None",
        },
    }


@pytest.fixture
def config_file(tmp_path, minimal_config_dict) -> Path:
    """Write the minimal config to a YAML file and return its path."""
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(minimal_config_dict, f)
    return path
