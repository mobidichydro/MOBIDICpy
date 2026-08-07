"""Tests for CalibrationConfig Pydantic models."""

import pytest
from pydantic import ValidationError

from mobidic.calibration.config import (
    CalibrationConfig,
    CalibrationParameter,
    CalibrationPeriod,
    DAConfig,
    DAStateConfig,
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

    def test_name_is_lower_cased(self):
        """PEST++ is case-insensitive, but pyemu only lower-cases some of its names."""
        p = CalibrationParameter(
            name="Wc_factor",
            parameter_key="parameters.multipliers.Wc_factor",
            initial_value=1.0,
            lower_bound=0.1,
            upper_bound=10.0,
        )
        assert p.name == "wc_factor"
        # The path into the MOBIDIC schema keeps its case
        assert p.parameter_key == "parameters.multipliers.Wc_factor"

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
                parameter_key="parameters.multipliers.ks_factor",
                initial_value=1.0,
                lower_bound=100.0,
                upper_bound=1.0,
            )

    def test_initial_value_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError, match="initial_value"):
            CalibrationParameter(
                name="test",
                parameter_key="parameters.multipliers.ks_factor",
                initial_value=200.0,
                lower_bound=0.01,
                upper_bound=100.0,
            )

    def test_log_transform_requires_positive_lower_bound(self):
        with pytest.raises(ValidationError, match="positive"):
            CalibrationParameter(
                name="test",
                parameter_key="parameters.multipliers.ks_factor",
                initial_value=0.5,
                lower_bound=-1.0,
                upper_bound=100.0,
                transform="log",
            )

    def test_fixed_transform_accepted(self):
        p = CalibrationParameter(
            name="fixed_param",
            parameter_key="parameters.routing.wcel",
            initial_value=5.0,
            lower_bound=1.0,
            upper_bound=10.0,
            transform="fixed",
        )
        assert p.transform == "fixed"

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValidationError, match="not a field of"):
            CalibrationParameter(
                name="test",
                parameter_key="not_a_section.foo",
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            )

    def test_unknown_nested_key_rejected(self):
        with pytest.raises(ValidationError, match="not a field of"):
            CalibrationParameter(
                name="test",
                parameter_key="parameters.multipliers.does_not_exist",
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            )

    def test_nested_model_leaf_rejected(self):
        with pytest.raises(ValidationError, match="nested model"):
            CalibrationParameter(
                name="test",
                parameter_key="parameters.multipliers",
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            )

    def test_non_numeric_leaf_rejected(self):
        with pytest.raises(ValidationError, match="numeric"):
            CalibrationParameter(
                name="test",
                parameter_key="parameters.routing.method",
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            )

    def test_empty_path_rejected(self):
        with pytest.raises(ValidationError, match="valid dot-notation path"):
            CalibrationParameter(
                name="test",
                parameter_key="",
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            )

    def test_trailing_dot_rejected(self):
        with pytest.raises(ValidationError, match="valid dot-notation path"):
            CalibrationParameter(
                name="test",
                parameter_key="parameters.multipliers.",
                initial_value=1.0,
                lower_bound=0.1,
                upper_bound=10.0,
            )


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

    def test_hydroerr_metric_names_accepted(self):
        """Names sourced from HydroErr (beyond the original hardcoded list) must validate."""
        for metric in ["kge_2009", "kge_2012", "mle", "mae", "r_squared", "pearson_r", "ve", "d"]:
            m = MetricConfig(metric=metric, target=1.0)
            assert m.metric == metric

    def test_validator_uses_registry(self):
        """The validator must accept every name in METRIC_REGISTRY (no drift)."""
        from mobidic.calibration.metrics import METRIC_REGISTRY

        for metric in METRIC_REGISTRY:
            MetricConfig(metric=metric, target=0.0)


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
                    {
                        "name": "p1",
                        "parameter_key": "parameters.multipliers.ks_factor",
                        "initial_value": 1,
                        "lower_bound": 0.1,
                        "upper_bound": 10,
                    },
                    {
                        "name": "p1",
                        "parameter_key": "parameters.routing.wcel",
                        "initial_value": 2,
                        "lower_bound": 0.1,
                        "upper_bound": 10,
                    },
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
        for tool in ["glm", "ies", "sen", "swp", "da", "opt", "mou", "sqp"]:
            overrides = {"pest_tool": tool}
            if tool == "swp":
                overrides["pest_options"] = {"sweep_parameter_csv_file": "ensemble.par.csv"}
            if tool == "da":
                # Sequential DA is the default mode and needs cycle machinery;
                # batch DA is the plain ensemble-smoother setup.
                overrides["da"] = {"mode": "batch"}
            cc = self._make_minimal_config(**overrides)
            assert cc.pest_tool == tool

    def test_invalid_pest_tool_rejected(self):
        with pytest.raises(ValidationError):
            self._make_minimal_config(pest_tool="invalid")

    def test_case_name_default_glm(self):
        cc = self._make_minimal_config(pest_tool="glm")
        assert cc.case_name == "calibration"

    def test_case_name_default_sen(self):
        cc = self._make_minimal_config(pest_tool="sen")
        assert cc.case_name == "sensitivity"

    def test_case_name_default_swp(self):
        cc = self._make_minimal_config(pest_tool="swp", pest_options={"sweep_parameter_csv_file": "ensemble.par.csv"})
        assert cc.case_name == "sweep"

    def test_case_name_explicit_overrides_default(self):
        cc = self._make_minimal_config(
            pest_tool="swp",
            pest_options={"sweep_parameter_csv_file": "ensemble.par.csv"},
            case_name="myrun",
        )
        assert cc.case_name == "myrun"

    def test_swp_requires_sweep_file(self):
        with pytest.raises(ValidationError, match="sweep_parameter_csv_file is required"):
            self._make_minimal_config(pest_tool="swp")

    def test_swp_requires_sweep_file_empty_options(self):
        with pytest.raises(ValidationError, match="sweep_parameter_csv_file is required"):
            self._make_minimal_config(pest_tool="swp", pest_options={"noptmax": 5})

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

    def test_invalid_num_workers_rejected(self):
        with pytest.raises(ValidationError, match="num_workers"):
            ParallelConfig(num_workers=-1)

        with pytest.raises(ValidationError, match="num_workers"):
            ParallelConfig(num_workers=0)

    def test_num_workers_exceeding_cpus_warns(self, recwarn):
        import os

        available = os.cpu_count() or 1
        # Should not raise, but a loguru warning is emitted (not captured by recwarn)
        pc = ParallelConfig(num_workers=available + 1000)
        assert pc.num_workers == available + 1000

    def test_cluster_mode(self):
        pc = ParallelConfig(manager_ip="192.168.1.100", port=5000, num_workers=1)
        assert pc.manager_ip == "192.168.1.100"


# ---- PESTPP-DA configuration tests ----


class TestDAConfig:
    def test_defaults(self):
        da = DAConfig()
        assert da.mode == "sequential"
        assert da.assimilate == "all"
        assert da.forecast_cycles == 0
        assert da.states.restart_from == "warmup"
        assert da.states.keep_cycles == 2

    def test_invalid_cycle_length_rejected(self):
        with pytest.raises(ValidationError, match="not a valid pandas offset"):
            DAConfig(cycle_length="six hours")

    def test_negative_cycle_length_rejected(self):
        with pytest.raises(ValidationError, match="must be positive"):
            DAConfig(cycle_length="-6h")

    def test_stop_cycle_before_hotstart_rejected(self):
        with pytest.raises(ValidationError, match="must be >= da.hotstart_cycle"):
            DAConfig(cycle_length="6h", hotstart_cycle=5, stop_cycle=2)

    def test_keep_cycles_minimum_is_two(self):
        with pytest.raises(ValidationError):
            DAStateConfig(keep_cycles=1)


class TestDAJointEstimation:
    """da.states.estimate selects formulation 2 and constrains the rest of the setup."""

    def test_estimate_is_empty_by_default(self):
        """An empty list is formulation 1: states transferred, never adjusted."""
        assert DAStateConfig().estimate == []
        assert DAStateConfig().estimate_reaches == "upstream"

    def test_discharge_is_accepted_with_state_files(self):
        states = DAStateConfig(restart_from="previous_cycle", estimate=["discharge"])
        assert states.estimate == ["discharge"]

    def test_estimate_requires_the_enkf_restart_mode(self):
        with pytest.raises(ValidationError, match="requires restart_from='previous_cycle'"):
            DAStateConfig(estimate=["discharge"])

    def test_soil_moisture_is_accepted(self):
        states = DAStateConfig(restart_from="previous_cycle", estimate=["discharge", "soil_moisture"])
        assert states.estimate == ["discharge", "soil_moisture"]

    def test_an_alias_overlapping_its_component_is_rejected(self):
        """'soil_moisture' already stands for both stores."""
        with pytest.raises(ValidationError, match="more than once"):
            DAStateConfig(restart_from="previous_cycle", estimate=["soil_moisture", "soil_capillary"])

    def test_saturation_bounds_must_be_a_usable_fraction(self):
        with pytest.raises(ValidationError, match="0 <= lower < upper <= 1"):
            DAStateConfig(restart_from="previous_cycle", saturation_bounds=(0.5, 0.2))
        with pytest.raises(ValidationError, match="0 <= lower < upper <= 1"):
            DAStateConfig(restart_from="previous_cycle", saturation_bounds=(0.0, 1.5))

    def test_min_zone_cells_defaults_to_no_merging(self):
        assert DAStateConfig().min_zone_cells == 1

    def test_unknown_state_variable_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown state variable"):
            DAStateConfig(restart_from="previous_cycle", estimate=["groundwater"])

    def test_duplicates_are_rejected(self):
        with pytest.raises(ValidationError, match="duplicates"):
            DAStateConfig(restart_from="previous_cycle", estimate=["discharge", "discharge"])

    def test_estimate_requires_a_warmup_period(self):
        """The warm-up supplies parval1 for every state parameter."""
        with pytest.raises(ValidationError, match="requires da.warmup_period"):
            DAConfig(
                cycle_length="6h",
                states={"restart_from": "previous_cycle", "estimate": ["discharge"]},
            )

    def test_estimate_with_a_warmup_period_is_accepted(self):
        da = DAConfig(
            cycle_length="6h",
            warmup_period={"start_date": "2023-10-25 00:00:00", "end_date": "2023-11-01 00:00:00"},
            states={"restart_from": "previous_cycle", "estimate": ["discharge"]},
        )
        assert da.states.estimate == ["discharge"]

    def test_bound_factor_must_exceed_one(self):
        with pytest.raises(ValidationError):
            DAStateConfig(bound_factor=1.0)

    def test_prior_std_must_be_positive(self):
        with pytest.raises(ValidationError):
            DAStateConfig(prior_std=0.0)


class TestCalibrationConfigDA:
    def _make(self, **overrides):
        defaults = {
            "mobidic_config": "Arno.yaml",
            "pest_tool": "da",
            "simulation_period": {
                "start_date": "2023-11-01 00:00:00",
                "end_date": "2023-11-04 23:45:00",
            },
            "da": {"cycle_length": "6h", "warmup_period": {"start_date": "2023-10-25", "end_date": "2023-11-01"}},
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

    def test_case_name_defaults_to_assimilation(self):
        assert self._make().case_name == "assimilation"

    def test_is_sequential_da(self):
        assert self._make().is_sequential_da is True
        assert self._make(da={"mode": "batch"}).is_sequential_da is False

    def test_sequential_without_cycle_length_rejected(self):
        with pytest.raises(ValidationError, match="da.cycle_length is required"):
            self._make(da={"warmup_period": {"start_date": "2023-10-25", "end_date": "2023-11-01"}})

    def test_sequential_without_simulation_period_rejected(self):
        with pytest.raises(ValidationError, match="simulation_period is required"):
            self._make(simulation_period=None)

    def test_warmup_required_when_restarting_from_warmup(self):
        with pytest.raises(ValidationError, match="da.warmup_period is required"):
            self._make(da={"cycle_length": "6h"})

    def test_warmup_optional_when_restarting_from_previous_cycle(self):
        cc = self._make(da={"cycle_length": "6h", "states": {"restart_from": "previous_cycle"}})
        assert cc.da.warmup_period is None

    def test_warmup_must_end_before_the_first_cycle(self):
        with pytest.raises(ValidationError, match="must be at or before"):
            self._make(
                da={
                    "cycle_length": "6h",
                    "warmup_period": {"start_date": "2023-10-25", "end_date": "2023-11-02"},
                }
            )

    def test_metrics_rejected_in_sequential_mode(self):
        with pytest.raises(ValidationError, match="Derived metrics are not supported"):
            self._make(
                observations=[
                    {
                        "name": "Q_329",
                        "obs_file": "observations/Q_329.csv",
                        "reach_id": 329,
                        "value_column": "Q_329",
                        "metrics": [{"metric": "nse", "target": 1.0}],
                    }
                ]
            )

    def test_metrics_allowed_in_batch_mode(self):
        cc = self._make(
            da={"mode": "batch"},
            observations=[
                {
                    "name": "Q_329",
                    "obs_file": "observations/Q_329.csv",
                    "reach_id": 329,
                    "value_column": "Q_329",
                    "metrics": [{"metric": "nse", "target": 1.0}],
                }
            ],
        )
        assert cc.observations[0].metrics is not None

    def test_pst_version_1_rejected_in_sequential_mode(self):
        with pytest.raises(ValidationError, match="pst_version must be 2"):
            self._make(pest_options={"pst_version": 1})

    def test_pst_version_2_accepted(self):
        assert self._make(pest_options={"pst_version": 2}).pest_options["pst_version"] == 2

    def test_da_section_with_another_tool_only_warns(self):
        cc = CalibrationConfig(
            mobidic_config="Arno.yaml",
            pest_tool="ies",
            da={"cycle_length": "6h"},
            parameters=[
                {
                    "name": "ks_factor",
                    "parameter_key": "parameters.multipliers.ks_factor",
                    "initial_value": 1.0,
                    "lower_bound": 0.01,
                    "upper_bound": 100.0,
                    "transform": "log",
                }
            ],
            observations=[
                {
                    "name": "Q_329",
                    "obs_file": "observations/Q_329.csv",
                    "reach_id": 329,
                    "value_column": "Q_329",
                }
            ],
        )
        assert cc.case_name == "calibration"
