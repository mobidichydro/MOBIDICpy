"""Tests for the PESTPP-DA cycle schedule and cycle tables."""

import pandas as pd
import pytest

from mobidic.calibration.config import ObservationGroup
from mobidic.calibration.da_cycles import (
    CYCLE_PARAM_NAME,
    assert_cycle_values_differ,
    build_cycle_schedule,
    build_observation_cycle_tables,
    build_parameter_cycle_table,
    read_cycle_metadata,
    schedule_from_metadata,
    write_cycle_metadata,
    write_cycle_tables,
)

DT = 900  # 15 minutes


def _group(name="Q1", weight=1.0):
    return ObservationGroup(name=name, obs_file="obs.csv", reach_id=1, value_column="Q", weight=weight)


def _frame(times, values):
    return pd.DataFrame({"time": pd.to_datetime(times), "value": values})


class TestBuildCycleSchedule:
    def test_windows_are_contiguous_without_overlap(self):
        # 8 timesteps of 15 min = 2 cycles of 1 h
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)

        assert schedule.n_cycles == 2
        assert schedule.n_steps_per_cycle == 4
        assert schedule.starts[0] == pd.Timestamp("2023-11-01 00:00:00")
        # Cycle 0 ends one timestep before cycle 1 starts: no timestep applied twice.
        assert schedule.ends[0] == pd.Timestamp("2023-11-01 00:45:00")
        assert schedule.starts[1] == pd.Timestamp("2023-11-01 01:00:00")
        assert schedule.ends[1] == pd.Timestamp("2023-11-01 01:45:00")

    def test_every_forcing_time_appears_exactly_once(self):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 05:45:00", "1h", DT)
        times = []
        for cycle in range(schedule.n_cycles):
            times.extend(schedule.slot_times(cycle))

        expected = pd.date_range("2023-11-01 00:00:00", "2023-11-01 05:45:00", freq=f"{DT}s")
        assert times == list(expected)
        assert len(set(times)) == len(times)

    def test_all_times_matches_slot_times(self):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 05:45:00", "1h", DT)
        expected = []
        for cycle in range(schedule.n_cycles):
            expected.extend(schedule.slot_times(cycle))
        assert list(schedule.all_times()) == expected

    def test_cycle_length_not_multiple_of_timestep_rejected(self):
        with pytest.raises(ValueError, match="not an integer multiple"):
            build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 05:45:00", "10min", DT)

    def test_window_too_short_rejected(self):
        with pytest.raises(ValueError, match="fewer than"):
            build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 00:30:00", "1h", DT)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError, match="before simulation start"):
            build_cycle_schedule("2023-11-01 02:00:00", "2023-11-01 01:00:00", "1h", DT)

    def test_trailing_partial_cycle_is_dropped(self):
        # 6 timesteps = 1 complete 4-step cycle + 2 leftover
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:15:00", "1h", DT)
        assert schedule.n_cycles == 1
        assert schedule.ends[-1] == pd.Timestamp("2023-11-01 00:45:00")


class TestObservationCycleTables:
    def _schedule(self):
        return build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)

    def test_names_are_rows_and_cycles_are_columns(self):
        schedule = self._schedule()
        obs, weight, n_slots = build_observation_cycle_tables(
            [_group()], {"Q1": _frame(list(schedule.all_times()), [1.0] * 8)}, schedule
        )

        assert list(obs.index) == [f"Q1_{i:04d}" for i in range(4)]
        assert list(obs.columns) == [0, 1]
        assert obs.shape == (4, 2)
        assert weight.shape == (4, 2)
        assert n_slots == {"Q1": 4}

    def test_values_follow_absolute_time_per_cycle(self):
        schedule = self._schedule()
        times = list(schedule.all_times())
        values = [float(i) for i in range(len(times))]
        obs, weight, _ = build_observation_cycle_tables([_group()], {"Q1": _frame(times, values)}, schedule)

        # Slot 0 of cycle 0 is t=0; slot 0 of cycle 1 is t=4 steps later.
        assert obs.loc["Q1_0000", 0] == 0.0
        assert obs.loc["Q1_0000", 1] == 4.0
        assert obs.loc["Q1_0003", 1] == 7.0
        assert (weight.to_numpy() == 1.0).all()

    def test_missing_observation_gives_nan_value_and_zero_weight(self):
        schedule = self._schedule()
        # Only the first cycle is observed
        times = list(schedule.slot_times(0))
        obs, weight, _ = build_observation_cycle_tables([_group()], {"Q1": _frame(times, [2.0] * 4)}, schedule)

        assert obs[0].notna().all()
        assert obs[1].isna().all()
        assert (weight[0] == 1.0).all()
        assert (weight[1] == 0.0).all()

    def test_assimilate_end_weights_only_the_last_slot(self):
        schedule = self._schedule()
        times = list(schedule.all_times())
        obs, weight, _ = build_observation_cycle_tables(
            [_group()], {"Q1": _frame(times, [1.0] * 8)}, schedule, assimilate="end"
        )

        assert (weight.loc["Q1_0003"] == 1.0).all()
        for slot in range(3):
            assert (weight.loc[f"Q1_{slot:04d}"] == 0.0).all()
        # Values are still reported for every slot
        assert obs.notna().all().all()

    def test_forecast_cycles_zero_the_trailing_weights(self):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 02:45:00", "1h", DT)
        times = list(schedule.all_times())
        _, weight, _ = build_observation_cycle_tables(
            [_group()], {"Q1": _frame(times, [1.0] * len(times))}, schedule, forecast_cycles=1
        )

        assert schedule.n_cycles == 3
        assert (weight[0] == 1.0).all()
        assert (weight[1] == 1.0).all()
        assert (weight[2] == 0.0).all()

    def test_group_weight_is_used(self):
        schedule = self._schedule()
        times = list(schedule.all_times())
        _, weight, _ = build_observation_cycle_tables([_group(weight=2.5)], {"Q1": _frame(times, [1.0] * 8)}, schedule)
        assert (weight.to_numpy() == 2.5).all()

    def test_invalid_assimilate_rejected(self):
        schedule = self._schedule()
        with pytest.raises(ValueError, match="assimilate must be"):
            build_observation_cycle_tables([_group()], {}, schedule, assimilate="sometimes")

    def test_too_many_forecast_cycles_rejected(self):
        schedule = self._schedule()
        with pytest.raises(ValueError, match="exceeds the number of cycles"):
            build_observation_cycle_tables([_group()], {}, schedule, forecast_cycles=5)


class TestParameterCycleTable:
    def test_one_row_per_parameter_one_column_per_cycle(self):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 02:45:00", "1h", DT)
        table = build_parameter_cycle_table(schedule)

        assert list(table.index) == [CYCLE_PARAM_NAME]
        assert list(table.columns) == [0, 1, 2]
        assert list(table.loc[CYCLE_PARAM_NAME]) == [0.0, 1.0, 2.0]

    def test_cycle_values_differ_between_consecutive_cycles(self):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 02:45:00", "1h", DT)
        assert_cycle_values_differ(build_parameter_cycle_table(schedule))

    def test_repeated_cycle_value_is_rejected(self):
        table = pd.DataFrame([[0.0, 0.0]], index=[CYCLE_PARAM_NAME], columns=[0, 1])
        with pytest.raises(ValueError, match="repeats between consecutive cycles"):
            assert_cycle_values_differ(table)


class TestCycleFiles:
    def test_write_cycle_tables_creates_all_three(self, tmp_path):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)
        obs, weight, _ = build_observation_cycle_tables([_group()], {}, schedule)
        par = build_parameter_cycle_table(schedule)

        names = write_cycle_tables(tmp_path, obs, weight, par)

        assert names["da_observation_cycle_table"] == "obs_cycle_table.csv"
        for basename in names.values():
            assert (tmp_path / basename).exists()

    def test_missing_observation_is_written_as_an_empty_cell(self, tmp_path):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)
        obs, weight, _ = build_observation_cycle_tables([_group()], {}, schedule)
        write_cycle_tables(tmp_path, obs, weight, build_parameter_cycle_table(schedule))

        text = (tmp_path / "obs_cycle_table.csv").read_text(encoding="utf-8")
        assert "Q1_0000,,\n" in text

    def test_cycle_metadata_roundtrip(self, tmp_path):
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 05:45:00", "1h", DT)
        path = write_cycle_metadata(schedule, tmp_path / "da_cycles.csv")

        df = read_cycle_metadata(path)
        assert list(df.index) == list(range(schedule.n_cycles))
        assert df.loc[0, "start_date"] == schedule.starts[0]
        assert df.loc[0, "end_date"] == schedule.ends[0]
        assert int(df.loc[0, "n_steps"]) == schedule.n_steps_per_cycle

        rebuilt = schedule_from_metadata(df)
        assert rebuilt == schedule


class TestWeightCycleTableErrorModel:
    def _schedule(self):
        return build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)

    def _relative_group(self):
        return ObservationGroup(
            name="Q1",
            obs_file="obs.csv",
            reach_id=1,
            value_column="Q",
            relative_error=0.1,
            min_error=5.0,
        )

    def test_weights_follow_the_observed_value(self):
        schedule = self._schedule()
        times = list(schedule.all_times())
        # Low flow in cycle 0, flood peak in cycle 1
        values = [10.0] * 4 + [400.0] * 4
        _, weight, _ = build_observation_cycle_tables([self._relative_group()], {"Q1": _frame(times, values)}, schedule)

        # sigma = max(0.1*10, 5) = 5 -> w = 0.2 ; sigma = 0.1*400 = 40 -> w = 0.025
        assert weight.loc["Q1_0000", 0] == pytest.approx(0.2)
        assert weight.loc["Q1_0000", 1] == pytest.approx(0.025)

    def test_unobserved_slots_still_get_zero_weight(self):
        schedule = self._schedule()
        times = list(schedule.slot_times(0))
        _, weight, _ = build_observation_cycle_tables(
            [self._relative_group()], {"Q1": _frame(times, [10.0] * 4)}, schedule
        )
        assert (weight[1] == 0.0).all()
