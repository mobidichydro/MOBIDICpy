"""Tests for PEST++ instruction file generation."""

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.instruction import generate_instruction_file, _make_obs_names


def _make_config(with_metrics=False):
    metrics = None
    if with_metrics:
        metrics = [
            {"metric": "nse", "target": 1.0, "weight": 10.0},
            {"metric": "peak_error", "target": 0.0, "weight": 8.0},
        ]

    return CalibrationConfig(
        mobidic_config="Arno.yaml",
        parameters=[
            {
                "name": "ks_factor",
                "parameter_key": "parameters.multipliers.ks_factor",
                "initial_value": 1.0,
                "lower_bound": 0.01,
                "upper_bound": 100.0,
            }
        ],
        observations=[
            {
                "name": "Q_329",
                "obs_file": "obs.csv",
                "reach_id": 329,
                "value_column": "Q_329",
                "metrics": metrics,
            }
        ],
    )


class TestMakeObsNames:
    def test_time_series_only(self):
        cc = _make_config()
        names = _make_obs_names(cc, {"Q_329": 5})
        assert len(names) == 5
        assert names[0] == "Q_329_0000"
        assert names[4] == "Q_329_0004"

    def test_with_metrics(self):
        cc = _make_config(with_metrics=True)
        names = _make_obs_names(cc, {"Q_329": 3})
        assert len(names) == 5  # 3 time-series + 2 metrics
        assert names[3] == "Q_329_nse"
        assert names[4] == "Q_329_peak_error"

    def test_zero_observations(self):
        cc = _make_config()
        names = _make_obs_names(cc, {"Q_329": 0})
        assert len(names) == 0


class TestGenerateInstructionFile:
    def test_generates_valid_ins(self, tmp_path):
        cc = _make_config()
        ins_path, obs_names = generate_instruction_file(cc, {"Q_329": 3}, tmp_path / "model_output.csv.ins")

        content = ins_path.read_text()
        lines = content.strip().split("\n")

        # First line: pif marker
        assert lines[0] == "pif ~"

        # Second line: skip header
        assert lines[1] == "l1"

        # Observation lines
        assert len(obs_names) == 3
        assert "!Q_329_0000!" in lines[2]
        assert "!Q_329_0001!" in lines[3]
        assert "!Q_329_0002!" in lines[4]

    def test_with_metrics(self, tmp_path):
        cc = _make_config(with_metrics=True)
        ins_path, obs_names = generate_instruction_file(cc, {"Q_329": 2}, tmp_path / "output.ins")

        assert len(obs_names) == 4  # 2 time-series + 2 metrics
        content = ins_path.read_text()
        assert "!Q_329_nse!" in content
        assert "!Q_329_peak_error!" in content
