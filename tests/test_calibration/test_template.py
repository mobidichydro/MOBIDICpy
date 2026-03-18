"""Tests for PEST++ template file generation."""

import pytest

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.template import generate_model_input_csv, generate_template_file


def _make_config():
    return CalibrationConfig(
        mobidic_config="Arno.yaml",
        parameters=[
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
                "initial_value": 3.0,
                "lower_bound": 0.5,
                "upper_bound": 10.0,
                "transform": "none",
            },
        ],
        observations=[
            {
                "name": "Q_329",
                "obs_file": "obs.csv",
                "reach_id": 329,
                "value_column": "Q_329",
            }
        ],
    )


class TestGenerateTemplateFile:
    def test_generates_valid_tpl(self, tmp_path):
        cc = _make_config()
        tpl_path = generate_template_file(cc, tmp_path / "model_input.csv.tpl")

        content = tpl_path.read_text()
        lines = content.strip().split("\n")

        # First line: ptf marker
        assert lines[0] == "ptf ~"

        # Second line: CSV header
        assert lines[1] == "parameter_key,value"

        # Parameter lines
        assert "parameters.multipliers.ks_factor" in lines[2]
        assert "~ ks_factor" in lines[2]
        assert "~" in lines[2]

        assert "parameters.routing.wcel" in lines[3]
        assert "~ wcel" in lines[3]

    def test_number_of_lines(self, tmp_path):
        cc = _make_config()
        tpl_path = generate_template_file(cc, tmp_path / "model_input.csv.tpl")
        lines = tpl_path.read_text().strip().split("\n")
        # Header (ptf ~) + CSV header + 2 parameters = 4 lines
        assert len(lines) == 4


class TestGenerateModelInputCsv:
    def test_generates_csv_with_initial_values(self, tmp_path):
        cc = _make_config()
        csv_path = generate_model_input_csv(cc, tmp_path / "model_input.csv")

        content = csv_path.read_text()
        lines = content.strip().split("\n")

        assert lines[0] == "parameter_key,value"
        assert "parameters.multipliers.ks_factor,1.0" in lines[1]
        assert "parameters.routing.wcel,3.0" in lines[2]

    def test_csv_readable_as_parameter_mapping(self, tmp_path):
        from mobidic.calibration.parameter_mapping import read_model_input_csv

        cc = _make_config()
        csv_path = generate_model_input_csv(cc, tmp_path / "model_input.csv")

        params = read_model_input_csv(csv_path)
        assert params["parameters.multipliers.ks_factor"] == pytest.approx(1.0)
        assert params["parameters.routing.wcel"] == pytest.approx(3.0)
