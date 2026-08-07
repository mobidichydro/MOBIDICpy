"""Tests for parsing PESTPP-DA per-cycle output."""

import pandas as pd
import pytest

from mobidic.calibration.config import CalibrationConfig
from mobidic.calibration.da_cycles import build_cycle_schedule, write_cycle_metadata
from mobidic.calibration.results import CalibrationResults

DT = 900
CASE = "assimilation"


def _make_config(**overrides) -> CalibrationConfig:
    defaults = {
        "mobidic_config": "Arno.yaml",
        "pest_tool": "da",
        "simulation_period": {"start_date": "2023-11-01 00:00:00", "end_date": "2023-11-01 01:45:00"},
        "da": {
            "cycle_length": "1h",
            "warmup_period": {"start_date": "2023-10-25 00:00:00", "end_date": "2023-11-01 00:00:00"},
        },
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


def _write_fixture(master_dir):
    """Two cycles x two iterations of a four-slot observation ensemble."""
    schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)
    write_cycle_metadata(schedule, master_dir / "da_cycles.csv")

    slots = [f"Q1_{i:04d}" for i in range(4)]
    for cycle in range(2):
        for iteration in range(2):
            # Value encodes cycle, iteration, slot and realization
            data = {slot: [10 * cycle + iteration + s / 100 + r for r in range(3)] for s, slot in enumerate(slots)}
            obs = pd.DataFrame(data, index=["real_0", "real_1", "base"])
            obs.to_csv(master_dir / f"{CASE}.{cycle}.{iteration}.obs.csv")

            par = pd.DataFrame(
                {"ks_factor": [1.0, 2.0, 3.0], "sp_q_00001": [0.5, 0.6, 0.7], "cycle_num": [cycle] * 3},
                index=["real_0", "real_1", "base"],
            )
            par.to_csv(master_dir / f"{CASE}.{cycle}.{iteration}.par.csv")

        pd.DataFrame({"ks_factor": [1.0]}, index=["base"]).to_csv(master_dir / f"{CASE}.global.{cycle}.pe.csv")
        pd.DataFrame({slots[0]: [1.0]}, index=["base"]).to_csv(master_dir / f"{CASE}.global.{cycle}.oe.csv")

    return schedule


@pytest.fixture
def results(tmp_path):
    _write_fixture(tmp_path)
    return CalibrationResults(master_dir=tmp_path, calib_config=_make_config())


class TestGetDaResults:
    def test_reads_every_cycle_and_iteration(self, results):
        out = results.get_da_results()

        assert set(out["observations"]) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert set(out["parameters"]) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert set(out["global_parameters"]) == {0, 1}
        assert set(out["global_observations"]) == {0, 1}

    def test_filters_by_cycle(self, results):
        out = results.get_da_results(cycle=1)
        assert set(out["observations"]) == {(1, 0), (1, 1)}
        assert set(out["global_parameters"]) == {1}

    def test_filters_by_iteration(self, results):
        out = results.get_da_results(iteration=0)
        assert set(out["observations"]) == {(0, 0), (1, 0)}

    def test_missing_output_returns_empty_dicts(self, tmp_path):
        res = CalibrationResults(master_dir=tmp_path, calib_config=_make_config())
        out = res.get_da_results()
        assert all(len(v) == 0 for v in out.values())

    def test_noptmax_zero_base_filenames_are_read(self, tmp_path):
        """With noptmax=0 PESTPP-DA names the file after the realization, not an iteration."""
        schedule = build_cycle_schedule("2023-11-01 00:00:00", "2023-11-01 01:45:00", "1h", DT)
        write_cycle_metadata(schedule, tmp_path / "da_cycles.csv")

        slots = [f"Q1_{i:04d}" for i in range(4)]
        for cycle in range(2):
            pd.DataFrame({slot: [float(cycle)] for slot in slots}, index=["base"]).to_csv(
                tmp_path / f"{CASE}.{cycle}.base.obs.csv"
            )

        res = CalibrationResults(master_dir=tmp_path, calib_config=_make_config())
        out = res.get_da_results()

        assert set(out["observations"]) == {(0, 0), (1, 0)}
        series = res.get_da_timeseries("Q1")
        assert list(series.columns) == ["base"]
        assert len(series) == 8


class TestGetDaTimeseries:
    def test_posterior_series_is_continuous_in_time(self, results):
        df = results.get_da_timeseries("Q1")

        assert list(df.columns) == ["real_0", "real_1", "base"]
        assert len(df) == 8  # 2 cycles x 4 slots
        assert df.index[0] == pd.Timestamp("2023-11-01 00:00:00")
        assert df.index[-1] == pd.Timestamp("2023-11-01 01:45:00")
        assert (df.index.to_series().diff().dropna() == pd.Timedelta(seconds=DT)).all()

    def test_posterior_uses_the_last_iteration(self, results):
        posterior = results.get_da_timeseries("Q1", posterior=True)
        prior = results.get_da_timeseries("Q1", posterior=False)

        # Cycle 0 slot 0, realization 0: iteration 1 vs iteration 0
        assert posterior.iloc[0, 0] == pytest.approx(1.0)
        assert prior.iloc[0, 0] == pytest.approx(0.0)

    def test_slots_are_ordered_within_a_cycle(self, results):
        df = results.get_da_timeseries("Q1", posterior=False)
        # Slot index is encoded in the hundredths digit
        assert df.iloc[0, 0] == pytest.approx(0.00)
        assert df.iloc[1, 0] == pytest.approx(0.01)
        assert df.iloc[3, 0] == pytest.approx(0.03)
        # Cycle 1 starts again at slot 0, offset by 10
        assert df.iloc[4, 0] == pytest.approx(10.00)

    def test_unknown_group_returns_none(self, results):
        assert results.get_da_timeseries("does_not_exist") is None

    def test_missing_cycle_metadata_returns_none(self, tmp_path):
        _write_fixture(tmp_path)
        (tmp_path / "da_cycles.csv").unlink()
        res = CalibrationResults(master_dir=tmp_path, calib_config=_make_config())
        assert res.get_da_timeseries("Q1") is None


class TestOptimalParameters:
    def test_sequential_da_returns_empty_without_raising(self, results):
        assert results.get_optimal_parameters() == {}

    def test_state_names_do_not_raise_in_batch_mode(self, tmp_path):
        cfg = _make_config(da={"mode": "batch"}, simulation_period=None)
        par_file = tmp_path / "calibration.1.par"
        par_file.write_text(
            "single point\nks_factor 2.5 1.0 0.0\ncycle_num 3.0 1.0 0.0\nsp_q_00001 0.4 1.0 0.0\n",
            encoding="utf-8",
        )
        res = CalibrationResults(master_dir=tmp_path, calib_config=cfg)

        optimal = res.get_optimal_parameters()
        assert optimal == {"parameters.multipliers.ks_factor": 2.5}
