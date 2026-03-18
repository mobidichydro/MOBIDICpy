"""Tests for CalibrationConfig Pydantic models."""

import pytest
from pydantic import ValidationError

from mobidic.calibration.config import (
    CalibrationConfig,
    CalibrationParameter,
    CalibrationPeriod,
    MetricConfig,
    ObservationGroup,
    ParallelConfig,
)


# ---- CalibrationParameter tests ----


class TestCalibrationParameter:
    def test_valid_parameter(self):
        p = CalibrationParameter(
            name="ks_factor",
            parameter_key="parameters.multipliers.ks_factor",
            initial_value=1.0,
            lower_bound=0.01,
            upper_bound=100.0,
            transform="log",
            par_group="soil",
        )
        assert p.name == "ks_factor"
        assert p.transform == "log"
        assert p.scale == 1.0
        assert p.offset == 0.0

    def test_name_with_spaces_rejected(self):
        with pytest.raises(ValidationError, match="cannot contain spaces"):
            CalibrationParameter(
                name="ks factor",
                parameter_key="parameters.multipliers.ks_factor",
                initial_value=1.0,
                lower_bound=0.01,
                upper_bound=100.0,
            )

    def test_lower_bound_greater_than_upper_rejected(self):
        with pytest.raises(ValidationError, match="lower_bound"):
            CalibrationParameter(
                name="test",
                parameter_key="a.b",
                initial_value=1.0,
                lower_bound=100.0,
                upper_bound=1.0,
            )

    def test_initial_value_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError, match="initial_value"):
            CalibrationParameter(
                name="test",
                parameter_key="a.b",
                initial_value=200.0,
                lower_bound=0.01,
                upper_bound=100.0,
            )

    def test_log_transform_requires_positive_lower_bound(self):
        with pytest.raises(ValidationError, match="positive"):
            CalibrationParameter(
                name="test",
                parameter_key="a.b",
                initial_value=0.5,
                lower_bound=-1.0,
                upper_bound=100.0,
                transform="log",
            )

    def test_fixed_transform_accepted(self):
        p = CalibrationParameter(
            name="fixed_param",
            parameter_key="a.b",
            initial_value=5.0,
            lower_bound=1.0,
            upper_bound=10.0,
            transform="fixed",
        )
        assert p.transform == "fixed"


# ---- MetricConfig tests ----


class TestMetricConfig:
    def test_valid_metric(self):
        m = MetricConfig(metric="nse", target=1.0, weight=10.0)
        assert m.metric == "nse"

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValidationError, match="Unsupported metric"):
            MetricConfig(metric="invalid_metric", target=1.0)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            MetricConfig(metric="nse", target=1.0, weight=-1.0)

    def test_all_supported_metrics(self):
        for metric in ["nse", "nse_log", "pbias", "peak_error", "rmse", "kge"]:
            m = MetricConfig(metric=metric, target=0.0)
            assert m.metric == metric


# ---- ObservationGroup tests ----


class TestObservationGroup:
    def test_valid_observation_group(self):
        og = ObservationGroup(
            name="Q_329",
            obs_file="observations/Q_329.csv",
            reach_id=329,
            weight=1.0,
            value_column="Q_329",
        )
        assert og.name == "Q_329"
        assert og.time_column == "time"  # default

    def test_with_metrics(self):
        og = ObservationGroup(
            name="Q_329",
            obs_file="observations/Q_329.csv",
            reach_id=329,
            value_column="Q_329",
            metrics=[
                MetricConfig(metric="nse", target=1.0, weight=10.0),
                MetricConfig(metric="peak_error", target=0.0, weight=8.0),
            ],
        )
        assert len(og.metrics) == 2

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            ObservationGroup(
                name="Q_329",
                obs_file="obs.csv",
                reach_id=329,
                weight=-0.5,
                value_column="Q",
            )


# ---- CalibrationConfig tests ----


class TestCalibrationConfig:
    def _make_minimal_config(self, **overrides):
        defaults = {
            "mobidic_config": "Arno.yaml",
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
            "observations": [
                {
                    "name": "Q_329",
                    "obs_file": "observations/Q_329.csv",
                    "reach_id": 329,
                    "value_column": "Q_329",
                }
            ],
        }
        defaults.update(overrides)
        return CalibrationConfig(**defaults)

    def test_minimal_valid_config(self):
        cc = self._make_minimal_config()
        assert cc.pest_tool == "glm"
        assert cc.working_dir == "pest_run"
        assert cc.parallel.port == 4004
        assert cc.use_raster_forcing is False

    def test_duplicate_parameter_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate parameter names"):
            self._make_minimal_config(
                parameters=[
                    {"name": "p1", "parameter_key": "a.b", "initial_value": 1, "lower_bound": 0.1, "upper_bound": 10},
                    {"name": "p1", "parameter_key": "c.d", "initial_value": 2, "lower_bound": 0.1, "upper_bound": 10},
                ]
            )

    def test_duplicate_observation_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate observation group names"):
            self._make_minimal_config(
                observations=[
                    {"name": "Q1", "obs_file": "a.csv", "reach_id": 1, "value_column": "Q"},
                    {"name": "Q1", "obs_file": "b.csv", "reach_id": 2, "value_column": "Q"},
                ]
            )

    def test_empty_parameters_rejected(self):
        with pytest.raises(ValidationError):
            self._make_minimal_config(parameters=[])

    def test_empty_observations_rejected(self):
        with pytest.raises(ValidationError):
            self._make_minimal_config(observations=[])

    def test_all_pest_tools_accepted(self):
        for tool in ["glm", "ies", "sen", "da", "opt", "mou", "sqp"]:
            cc = self._make_minimal_config(pest_tool=tool)
            assert cc.pest_tool == tool

    def test_invalid_pest_tool_rejected(self):
        with pytest.raises(ValidationError):
            self._make_minimal_config(pest_tool="invalid")

    def test_calibration_period(self):
        cc = self._make_minimal_config(calibration_period={"start_date": "2023-11-01", "end_date": "2023-11-30"})
        assert cc.calibration_period.start_date == "2023-11-01"
        assert cc.calibration_period.end_date == "2023-11-30"

    def test_calibration_period_start_after_end_rejected(self):
        with pytest.raises(ValidationError, match="must be before"):
            self._make_minimal_config(calibration_period={"start_date": "2023-12-01", "end_date": "2023-11-01"})

    def test_calibration_period_equal_dates_rejected(self):
        with pytest.raises(ValidationError, match="must be before"):
            self._make_minimal_config(calibration_period={"start_date": "2023-11-01", "end_date": "2023-11-01"})

    def test_calibration_period_invalid_date_rejected(self):
        with pytest.raises(ValidationError, match="Invalid date format"):
            self._make_minimal_config(calibration_period={"start_date": "not-a-date", "end_date": "2023-11-30"})

    def test_pest_options(self):
        cc = self._make_minimal_config(pest_options={"noptmax": 30, "ies_num_reals": 50})
        assert cc.pest_options["noptmax"] == 30
        assert cc.pest_options["ies_num_reals"] == 50

    def test_use_raster_forcing(self):
        cc = self._make_minimal_config(use_raster_forcing=True)
        assert cc.use_raster_forcing is True

    def test_simulation_period(self):
        cc = self._make_minimal_config(
            simulation_period={"start_date": "2023-01-01", "end_date": "2023-11-30"},
            calibration_period={"start_date": "2023-11-01", "end_date": "2023-11-30"},
        )
        assert cc.simulation_period.start_date == "2023-01-01"
        assert cc.calibration_period.start_date == "2023-11-01"

    def test_simulation_period_without_calibration_period(self):
        cc = self._make_minimal_config(
            simulation_period={"start_date": "2023-01-01", "end_date": "2023-12-31"},
        )
        assert cc.simulation_period is not None
        assert cc.calibration_period is None

    def test_calibration_before_simulation_start_rejected(self):
        with pytest.raises(ValidationError, match="must be >= simulation_period.start_date"):
            self._make_minimal_config(
                simulation_period={"start_date": "2023-06-01", "end_date": "2023-11-30"},
                calibration_period={"start_date": "2023-01-01", "end_date": "2023-11-30"},
            )

    def test_calibration_after_simulation_end_rejected(self):
        with pytest.raises(ValidationError, match="must be <= simulation_period.end_date"):
            self._make_minimal_config(
                simulation_period={"start_date": "2023-01-01", "end_date": "2023-06-30"},
                calibration_period={"start_date": "2023-01-01", "end_date": "2023-11-30"},
            )


# ---- CalibrationPeriod tests ----


class TestCalibrationPeriod:
    def test_valid_period(self):
        cp = CalibrationPeriod(start_date="2023-11-01", end_date="2023-11-30")
        assert cp.start_date == "2023-11-01"
        assert cp.end_date == "2023-11-30"

    def test_valid_period_with_time(self):
        cp = CalibrationPeriod(start_date="2023-11-01 06:00:00", end_date="2023-11-30 18:00:00")
        assert cp.start_date == "2023-11-01 06:00:00"

    def test_start_after_end_rejected(self):
        with pytest.raises(ValidationError, match="must be before"):
            CalibrationPeriod(start_date="2024-01-01", end_date="2023-01-01")

    def test_equal_dates_rejected(self):
        with pytest.raises(ValidationError, match="must be before"):
            CalibrationPeriod(start_date="2023-11-01", end_date="2023-11-01")

    def test_invalid_date_format_rejected(self):
        with pytest.raises(ValidationError, match="Invalid date format"):
            CalibrationPeriod(start_date="31/12/2023", end_date="2024-01-01")

    def test_garbage_date_rejected(self):
        with pytest.raises(ValidationError, match="Invalid date format"):
            CalibrationPeriod(start_date="not-a-date", end_date="2024-01-01")


# ---- ParallelConfig tests ----


class TestParallelConfig:
    def test_defaults(self):
        pc = ParallelConfig()
        assert pc.num_workers is None
        assert pc.port == 4004
        assert pc.manager_ip is None

    def test_invalid_port_rejected(self):
        with pytest.raises(ValidationError, match="Port"):
            ParallelConfig(port=80)

    def test_cluster_mode(self):
        pc = ParallelConfig(manager_ip="192.168.1.100", port=5000, num_workers=16)
        assert pc.manager_ip == "192.168.1.100"
