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


# ---- Data-assimilation extra rows ----


class TestExtraParameters:
    def _extras(self):
        from mobidic.calibration.template import ExtraParameter

        return [
            ExtraParameter("__cycle__", "cycle_num", 0.0, width=12),
            ExtraParameter("__state_id__", "sp_state_id", -1.0, width=20),
        ]

    def test_template_appends_reserved_rows(self, tmp_path):
        path = generate_template_file(_make_config(), tmp_path / "model_input.csv.tpl", extra_parameters=self._extras())
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        assert lines[0] == "ptf ~"
        assert lines[1] == "parameter_key,value"
        assert lines[-2].startswith("__cycle__,~ cycle_num")
        assert lines[-1].startswith("__state_id__,~ sp_state_id")

    def test_state_field_is_wide_enough_to_avoid_truncation(self, tmp_path):
        path = generate_template_file(_make_config(), tmp_path / "model_input.csv.tpl", extra_parameters=self._extras())
        state_line = path.read_text(encoding="utf-8").strip().split("\n")[-2]

        marker = state_line.split(",", 1)[1]
        assert marker.startswith("~") and marker.endswith("~")
        # PEST fits the value to the field width; 12 characters is the minimum
        # that keeps a six-digit identifier exact.
        assert len(marker) >= 12

    def test_narrow_field_is_rejected(self, tmp_path):
        from mobidic.calibration.template import ExtraParameter

        with pytest.raises(ValueError, match="at least 12"):
            generate_template_file(
                _make_config(),
                tmp_path / "model_input.csv.tpl",
                extra_parameters=[ExtraParameter("__state_id__", "sp_state_id", -1.0, width=4)],
            )

    def test_model_input_csv_carries_the_initial_values(self, tmp_path):
        path = generate_model_input_csv(_make_config(), tmp_path / "model_input.csv", extra_parameters=self._extras())
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        assert lines[-2] == "__cycle__,0.0"
        assert lines[-1] == "__state_id__,-1.0"

    def test_template_and_csv_stay_in_sync(self, tmp_path):
        from mobidic.calibration.parameter_mapping import read_model_input_csv

        extras = self._extras()
        tpl = generate_template_file(_make_config(), tmp_path / "model_input.csv.tpl", extra_parameters=extras)
        csv = generate_model_input_csv(_make_config(), tmp_path / "model_input.csv", extra_parameters=extras)

        tpl_keys = [line.split(",", 1)[0] for line in tpl.read_text(encoding="utf-8").strip().split("\n")[2:]]
        assert list(read_model_input_csv(csv)) == tpl_keys
