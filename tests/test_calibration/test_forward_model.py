"""Tests for the PEST++ forward model wrapper."""

from pathlib import Path

import pytest
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


class TestPrepareSimulationDoesNotMutateCaller:
    """A caller that reuses one GISData for many realizations must get it back intact.

    ``prepare_simulation()`` drops the grids superseded by a calibrated scalar and
    rebinds ``network``; done in place, the next unrelated simulation would silently
    fall back to the scalar config values and to the last realization's routing.
    """

    def _gisdata(self, tmp_path):
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import LineString

        from mobidic.preprocessing.preprocessor import GISData

        network = gpd.GeoDataFrame(
            {
                "mobidic_id": [0, 1],
                "strahler_order": [1, 2],
                "length_m": [1000.0, 2000.0],
                "width_m": [1.0, 2.83],
                "lag_time_s": [193.0, 386.0],
                "n_manning": [0.03, 0.03],
                "geometry": [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            }
        )
        grids = {"alpha": np.full((2, 2), 1e-5), "beta": np.full((2, 2), 2e-5), "ks": np.full((2, 2), 3e-5)}
        return GISData(
            grids=grids,
            metadata={"shape": (2, 2)},
            network=network,
            hillslope_reach_map=np.zeros((2, 2)),
            config=None,
        )

    def test_caller_gisdata_is_unchanged(self, tmp_path):
        from mobidic.calibration.forward_model import prepare_simulation

        base_config_path = tmp_path / "config.yaml"
        with open(base_config_path, "w", encoding="utf-8") as f:
            yaml.dump(_minimal_config_dict(), f)

        gisdata = self._gisdata(tmp_path)
        original_grids = set(gisdata.grids)
        original_network = gisdata.network
        original_lag = gisdata.network["lag_time_s"].tolist()

        _, gis = prepare_simulation(
            base_config_path=base_config_path,
            param_updates={"parameters.soil.alpha": 9e-5, "parameters.routing.wcel": 2.0},
            gisdata=gisdata,
            routing_params_calibrated=True,
        )

        # The caller's object still has every grid and its original routing table.
        assert set(gisdata.grids) == original_grids
        assert gisdata.network is original_network
        assert gisdata.network["lag_time_s"].tolist() == original_lag

        # The returned view carries the calibrated changes.
        assert "alpha" not in gis.grids
        assert "beta" in gis.grids  # not calibrated -> kept
        assert gis.network["lag_time_s"].tolist() == [500.0, 1000.0]  # length_m / wcel


# ---- Data-assimilation interface ----


class TestPestValueFormatting:
    """PEST++ fails the whole run on a denormal double, so none may be written.

    The linear routing recursion Q(t+dt) = C3*Q(t) + C4*qL decays exponentially
    towards zero, so a reach with no inflow drifts into the subnormal range and
    stays there. Observed in practice as:
        InstructionFile error ... casting '2.9954111345e-314' to double yielded
        denormal value for free instruction: '!ST_Q_1013!'
    """

    def test_denormals_are_flushed_to_zero(self):
        import numpy as np

        from mobidic.calibration.forward_model import _pest_value

        for denormal in (2.9954111345e-314, 8.0976127055e-315, 5e-324, -1e-320):
            written = _pest_value(denormal)
            assert float(written) == 0.0
            # The real criterion: what PEST++ parses back must be normal or zero.
            assert float(written) == 0.0 or abs(float(written)) >= np.finfo(np.float64).tiny

    def test_normal_values_are_untouched(self):
        from mobidic.calibration.forward_model import _pest_value

        for value in (0.0, 47.83, 1.0e-300, 471.0, 1.0e6):
            assert float(_pest_value(value)) == pytest.approx(value)

    def test_written_output_never_contains_a_denormal(self, tmp_path):
        import json

        import geopandas as gpd
        import numpy as np
        from shapely.geometry import LineString

        from mobidic.calibration.forward_model import _write_model_output

        network = gpd.GeoDataFrame(
            {"mobidic_id": [0, 1], "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])]}
        )
        discharge = np.array([[1.0, 3.0e-314], [2.0, 1.0e-320]])
        out = tmp_path / "model_output.csv"

        _write_model_output(
            discharge_ts=discharge,
            sim_times=[],
            observation_reaches=[],
            network=network,
            output_path=out,
            obs_data_json=json.dumps([{"name": "Q1", "reach_id": 1}]),
            slot_indices=[0, 1],
            extra_outputs=[("st_q_0001", 4.0e-315)],
        )

        values = [float(line.split(",")[1]) for line in out.read_text(encoding="utf-8").strip().split("\n")[1:]]
        assert all(v == 0.0 or abs(v) >= np.finfo(np.float64).tiny for v in values)


class TestDAReservedKeys:
    def test_reserved_keys_are_recognised(self):
        from mobidic.calibration.forward_model import _is_reserved_da_key

        assert _is_reserved_da_key("__cycle__")
        assert _is_reserved_da_key("__state_id__")
        assert _is_reserved_da_key("__state__.q.0278")
        assert not _is_reserved_da_key("parameters.soil.ks")
        assert not _is_reserved_da_key("parameters.multipliers.ks_factor")

    def test_reserved_keys_are_never_applied_to_the_config(self, tmp_path):
        """apply_optimal_parameters would raise on a reserved name; it must never see one."""
        raw = {
            "parameters.soil.ks": 2.5,
            "__cycle__": 3.0,
            "__state_id__": 12345.0,
        }
        from mobidic.calibration.forward_model import _is_reserved_da_key

        param_updates = {k: v for k, v in raw.items() if not _is_reserved_da_key(k)}
        assert param_updates == {"parameters.soil.ks": 2.5}

        base_config_path = tmp_path / "config.yaml"
        with open(base_config_path, "w", encoding="utf-8") as f:
            yaml.dump(_minimal_config_dict(), f)
        config = load_config(base_config_path)
        apply_optimal_parameters(config, param_updates)
        assert config.parameters.soil.ks == 2.5


class TestDAModelOutput:
    def test_slot_indices_and_extra_outputs_are_written(self, tmp_path):
        import json

        import geopandas as gpd
        import numpy as np
        from shapely.geometry import LineString

        from mobidic.calibration.forward_model import _write_model_output

        network = gpd.GeoDataFrame(
            {"mobidic_id": [0, 1], "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])]}
        )
        discharge = np.arange(20, dtype=float).reshape(10, 2)
        out = tmp_path / "model_output.csv"

        _write_model_output(
            discharge_ts=discharge,
            sim_times=[],
            observation_reaches=[],
            network=network,
            output_path=out,
            obs_data_json=json.dumps([{"name": "Q1", "reach_id": 1}]),
            slot_indices=[6, 7, 8, 9],
            extra_outputs=[("st_state_id", 123456.0)],
        )

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0] == "obs_name,value"
        assert [line.split(",")[0] for line in lines[1:]] == [
            "Q1_0000",
            "Q1_0001",
            "Q1_0002",
            "Q1_0003",
            "st_state_id",
        ]
        # Reach 1 is column 1: values 13, 15, 17, 19 at rows 6..9
        assert float(lines[1].split(",")[1]) == 13.0
        assert float(lines[4].split(",")[1]) == 19.0

    def test_state_identifier_survives_the_written_precision(self, tmp_path):
        import json

        import geopandas as gpd
        import numpy as np
        from shapely.geometry import LineString

        from mobidic.calibration.forward_model import _write_model_output

        network = gpd.GeoDataFrame({"mobidic_id": [0], "geometry": [LineString([(0, 0), (1, 0)])]})
        out = tmp_path / "model_output.csv"

        for state_id in (0, 1, 999_999, 123_456):
            _write_model_output(
                discharge_ts=np.zeros((1, 1)),
                sim_times=[],
                observation_reaches=[],
                network=network,
                output_path=out,
                obs_data_json=json.dumps([{"name": "Q1", "reach_id": 0}]),
                slot_indices=[0],
                extra_outputs=[("st_state_id", float(state_id))],
            )
            written = out.read_text(encoding="utf-8").strip().split("\n")[-1].split(",")[1]
            assert int(float(written)) == state_id
