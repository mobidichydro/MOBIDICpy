"""Tests for the pestpp-swp (parameter sweep) path in CalibrationResults."""

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.results import CalibrationResults


def _make_swp_config(**overrides) -> CalibrationConfig:
    defaults = {
        "mobidic_config": "Arno.yaml",
        "pest_tool": "swp",
        "pest_options": {"sweep_parameter_csv_file": "ensemble.par.csv"},
        "parameters": [
            {
                "name": "ks_factor",
                "parameter_key": "parameters.multipliers.ks_factor",
                "initial_value": 1.0,
                "lower_bound": 0.01,
                "upper_bound": 100.0,
                "transform": "log",
            }
        ],
        "observations": [{"name": "Q1", "obs_file": "obs.csv", "reach_id": 1, "value_column": "Q"}],
    }
    defaults.update(overrides)
    return CalibrationConfig(**defaults)


def test_get_sweep_results_reads_csv(tmp_path):
    cfg = _make_swp_config()  # case_name defaults to "sweep"
    (tmp_path / "sweep_out.csv").write_text(
        "run_id,input_run_id,failed_flag,phi,Q1_0000,Q1_0001\n0,0,0,1.5,0.2,0.3\n1,1,0,2.5,0.4,0.5\n",
        encoding="utf-8",
    )

    results = CalibrationResults(master_dir=tmp_path, calib_config=cfg)
    df = results.get_sweep_results()

    assert df is not None
    assert len(df) == 2
    assert "phi" in df.columns


def test_get_sweep_results_missing_file_returns_none(tmp_path):
    cfg = _make_swp_config()
    results = CalibrationResults(master_dir=tmp_path, calib_config=cfg)
    assert results.get_sweep_results() is None


def test_get_optimal_parameters_empty_for_swp(tmp_path):
    cfg = _make_swp_config()
    results = CalibrationResults(master_dir=tmp_path, calib_config=cfg)
    assert results.get_optimal_parameters() == {}
