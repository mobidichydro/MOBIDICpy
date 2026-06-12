"""Tests for the PEST++ forward model wrapper."""

from pathlib import Path

import yaml

from mobidic.calibration.parameter_mapping import apply_optimal_parameters
from mobidic.config import load_config


def _minimal_config_dict() -> dict:
    """Return a minimal-but-valid MOBIDIC config dict for testing."""
    return {
        "basin": {
            "id": "TestBasin",
            "baricenter": {"lon": 10.0, "lat": 45.0},
        },
        "paths": {
            "meteodata": "meteodata.nc",
            "gisdata": "gisdata.nc",
            "network": "network.parquet",
            "states": "states/",
            "output": "outputs/",
        },
        "vector_files": {"river_network": {"shp": "network.shp"}},
        "raster_files": {
            "dtm": "dtm.tif",
            "flow_dir": "flowdir.tif",
            "flow_acc": "flowacc.tif",
            "Wc0": "wc0.tif",
            "Wg0": "wg0.tif",
            "ks": "ks.tif",
            "CLC": "clc.tif",
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
                "Kc_CLC_map": "kc/custom_kc_mapping.csv",  # Relative path
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


class TestForwardModelPathResolution:
    """Regression tests for issue #48: relative paths must resolve against the
    base config directory, not the worker directory where the forward model runs.
    """

    def test_kc_clc_map_resolves_against_base_config_dir(self, tmp_path, monkeypatch):
        """The relative Kc_CLC_map path must resolve to the base config's dir even
        when the process runs from a different (worker) directory."""
        # Base config lives in its own directory with the referenced CSV next to it.
        base_dir = tmp_path / "case"
        (base_dir / "kc").mkdir(parents=True)
        kc_csv = base_dir / "kc" / "custom_kc_mapping.csv"
        kc_csv.write_text("clc_code,kc_jan\n111,0.5\n", encoding="utf-8")

        base_config_path = base_dir / "config.yaml"
        with open(base_config_path, "w", encoding="utf-8") as f:
            yaml.dump(_minimal_config_dict(), f)

        # Simulate a PEST++ worker running from a separate directory.
        worker_dir = tmp_path / "worker_42"
        worker_dir.mkdir()
        monkeypatch.chdir(worker_dir)

        # This mirrors forward_model Step 2: load the base config and apply params.
        config = load_config(base_config_path)
        apply_optimal_parameters(config, {"parameters.soil.ks": 2.5})

        resolved = Path(config.parameters.soil.Kc_CLC_map)
        assert resolved.is_absolute()
        assert resolved == kc_csv.resolve()
        assert resolved.exists()
        # Sanity: the in-memory parameter update was applied.
        assert config.parameters.soil.ks == 2.5

    def test_paths_independent_of_cwd(self, tmp_path, monkeypatch):
        """Resolved paths must not depend on the current working directory."""
        base_dir = tmp_path / "case"
        (base_dir / "kc").mkdir(parents=True)
        (base_dir / "kc" / "custom_kc_mapping.csv").write_text("clc_code,kc_jan\n111,0.5\n", encoding="utf-8")
        base_config_path = base_dir / "config.yaml"
        with open(base_config_path, "w", encoding="utf-8") as f:
            yaml.dump(_minimal_config_dict(), f)

        monkeypatch.chdir(tmp_path)
        kc_a = Path(load_config(base_config_path).parameters.soil.Kc_CLC_map)

        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        kc_b = Path(load_config(base_config_path).parameters.soil.Kc_CLC_map)

        assert kc_a == kc_b
