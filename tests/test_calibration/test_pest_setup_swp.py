"""Tests for the pestpp-swp (parameter sweep) path in PestSetup."""

import pytest

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.pest_setup import PEST_TOOL_MAP, PestSetup


def _make_swp_config(sweep_csv: str, **overrides) -> CalibrationConfig:
    defaults = {
        "mobidic_config": "Arno.yaml",
        "pest_tool": "swp",
        "pest_options": {"sweep_parameter_csv_file": sweep_csv},
        "parameters": [
            {
                "name": "ks_factor",
                "parameter_key": "parameters.multipliers.ks_factor",
                "initial_value": 1.0,
                "lower_bound": 0.01,
                "upper_bound": 100.0,
                "transform": "log",
            },
            {
                "name": "wcel",
                "parameter_key": "parameters.routing.wcel",
                "initial_value": 5.0,
                "lower_bound": 1.0,
                "upper_bound": 10.0,
                "transform": "none",
            },
        ],
        "observations": [
            {
                "name": "Q1",
                "obs_file": "obs.csv",
                "reach_id": 1,
                "value_column": "Q",
                "metrics": [{"metric": "nse", "target": 1.0, "weight": 1.0}],
            }
        ],
    }
    defaults.update(overrides)
    return CalibrationConfig(**defaults)


def _write_sweep_csv(path, columns):
    header = ",".join(["real_name", *columns])
    rows = [header, "0," + ",".join("1.0" for _ in columns), "base," + ",".join("1.0" for _ in columns)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_tool_map_has_swp():
    assert PEST_TOOL_MAP["swp"] == "pestpp-swp"


def test_pest_exe_is_swp():
    cfg = _make_swp_config("ensemble.par.csv")
    setup = PestSetup(cfg, base_path=".")
    assert setup._pest_exe == "pestpp-swp"


def test_prepare_sweep_file_copies_and_sets_name(tmp_path):
    sweep = tmp_path / "ensemble.par.csv"
    _write_sweep_csv(sweep, ["ks_factor", "wcel"])

    cfg = _make_swp_config("ensemble.par.csv")
    setup = PestSetup(cfg, base_path=tmp_path)

    wd = tmp_path / "wd"
    wd.mkdir()
    setup._prepare_sweep_file(wd)

    assert setup._sweep_csv_name == "ensemble.par.csv"
    assert (wd / "ensemble.par.csv").exists()


def test_prepare_sweep_file_missing_file_raises(tmp_path):
    cfg = _make_swp_config("does_not_exist.par.csv")
    setup = PestSetup(cfg, base_path=tmp_path)
    wd = tmp_path / "wd"
    wd.mkdir()
    with pytest.raises(FileNotFoundError, match="sweep_parameter_csv_file not found"):
        setup._prepare_sweep_file(wd)


def test_prepare_sweep_file_missing_column_raises(tmp_path):
    sweep = tmp_path / "ensemble.par.csv"
    # 'wcel' column is missing
    _write_sweep_csv(sweep, ["ks_factor"])

    cfg = _make_swp_config("ensemble.par.csv")
    setup = PestSetup(cfg, base_path=tmp_path)
    wd = tmp_path / "wd"
    wd.mkdir()
    with pytest.raises(ValueError, match="missing columns for calibration parameter"):
        setup._prepare_sweep_file(wd)


def test_prepare_sweep_file_column_check_case_insensitive(tmp_path):
    sweep = tmp_path / "ensemble.par.csv"
    _write_sweep_csv(sweep, ["KS_FACTOR", "WCEL"])

    cfg = _make_swp_config("ensemble.par.csv")
    setup = PestSetup(cfg, base_path=tmp_path)
    wd = tmp_path / "wd"
    wd.mkdir()
    # Should not raise despite different case
    setup._prepare_sweep_file(wd)
    assert setup._sweep_csv_name == "ensemble.par.csv"


def test_build_pst_sets_sweep_option(tmp_path):
    pytest.importorskip("pyemu")

    sweep = tmp_path / "ensemble.par.csv"
    _write_sweep_csv(sweep, ["ks_factor", "wcel"])

    cfg = _make_swp_config("ensemble.par.csv")
    setup = PestSetup(cfg, base_path=tmp_path)
    wd = tmp_path / "wd"
    wd.mkdir()
    setup._prepare_sweep_file(wd)

    # Minimal observation bookkeeping: rely solely on the metric pseudo-observation
    setup._obs_data = []
    setup._n_obs_per_group = {}

    pst = setup._build_pst(obs_names=[], wd=wd)
    assert pst.pestpp_options.get("sweep_parameter_csv_file") == "ensemble.par.csv"
